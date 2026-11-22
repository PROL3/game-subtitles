"""Screen capture, ROI selection, preprocessing, and OCR."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import mss
import numpy as np
from rapidfuzz.fuzz import ratio
from rapidocr_onnxruntime import RapidOCR


ROI_FILE = Path("roi.json")


def select_roi(roi_file: Path = ROI_FILE, monitor: str = "all") -> dict:
    """Capture one monitor or the complete virtual desktop for ROI selection."""
    with mss.mss() as sct:
        if monitor == "all":
            target = sct.monitors[0]
        else:
            index = int(monitor)
            if index <= 0 or index >= len(sct.monitors):
                raise ValueError(f"Monitor {index} is unavailable. Connected monitors: 1-{len(sct.monitors) - 1}")
            target = sct.monitors[index]
        frame = np.asarray(sct.grab(target))[:, :, :3]

    cv2.namedWindow("Select subtitle ROI", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Select subtitle ROI", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    if not w or not h:
        raise RuntimeError("No ROI was selected.")
    roi = {"desktop": monitor == "all", "monitor": None if monitor == "all" else int(monitor),
           "left": int(x), "top": int(y), "width": int(w), "height": int(h)}
    roi_file.write_text(json.dumps(roi, indent=2), encoding="utf-8")
    return roi


def load_or_select_roi(select: bool = False, roi_file: Path = ROI_FILE, monitor: str = "all") -> dict:
    if select or not roi_file.exists():
        return select_roi(roi_file, monitor)
    return json.loads(roi_file.read_text(encoding="utf-8"))


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Keep bright white/yellow glyphs while suppressing moving backgrounds."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 135, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 45, 105]), np.array([48, 255, 255]))
    bright = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(cv2.bitwise_or(white, yellow), bright)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # The subtitle in the supplied frame is relatively small inside a wide ROI.
    # Upscaling substantially helps RapidOCR's detector find the glyphs.
    return cv2.resize(mask, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)


class SubtitleCapture:
    def __init__(self, roi: dict, on_text: Callable[[str], None], fps: float = 7.0):
        self.roi = roi
        self.on_text = on_text
        self.interval = 1.0 / fps
        self.ocr = RapidOCR()
        self.stop_event = threading.Event()
        self.ocr_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._last_text = ""
        self._candidate_text = ""
        self._candidate_count = 0
        self._last_change = 0.0

    @staticmethod
    def _plausible_ocr(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text)
        words = re.findall(r"[A-Za-z]{2,}", text)
        if len(letters) < 2 or not words:
            return False
        if len(letters) / max(len(re.sub(r"\s", "", text)), 1) < 0.55:
            return False
        if re.search(r"(.)\1\1\1", text, re.IGNORECASE):
            return False
        return True

    @staticmethod
    def _extract_text(result) -> str:
        # rapidocr-onnxruntime has returned both tuples and result objects across versions.
        texts = []
        if isinstance(result, tuple) and len(result) >= 2:
            texts = result[1] or []
        elif isinstance(result, list):
            if result and isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
                texts = [row[1] for row in result
                         if len(row) < 3 or float(row[2]) >= 0.50]
            else:
                texts = result
        return " ".join(str(t) for t in texts if t).strip()

    def _ocr_changed(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (96, 32), interpolation=cv2.INTER_AREA)
        if self._last_frame is None:
            self._last_frame = small
            self._last_change = time.monotonic()
            return True
        delta = float(cv2.absdiff(small, self._last_frame).mean())
        self._last_frame = small
        if delta < 2.0:
            return False
        now = time.monotonic()
        if now - self._last_change < 0.08:
            return False
        self._last_change = now
        return True

    def _read_text(self, frame: np.ndarray) -> str:
        # One OCR pass is the fast path. Retry on the color frame only if the
        # thresholded image produced no text.
        result, _ = self.ocr(preprocess(frame))
        text = " ".join(self._extract_text(result).split())
        if not text:
            result, _ = self.ocr(cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
            text = " ".join(self._extract_text(result).split())
        return text

    def run(self) -> None:
        ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        ocr_thread.start()
        with mss.mss() as sct:
            monitor = sct.monitors[0] if self.roi.get("desktop") else sct.monitors[int(self.roi.get("monitor", 1))]
            region = {"left": monitor["left"] + self.roi["left"], "top": monitor["top"] + self.roi["top"],
                      "width": self.roi["width"], "height": self.roi["height"]}
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = np.asarray(sct.grab(region))[:, :, :3]
                if self._ocr_changed(frame):
                    # Latest-only frame handoff: OCR never blocks screen capture.
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    self.ocr_event.set()
                self.stop_event.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def _ocr_worker(self) -> None:
        while not self.stop_event.is_set():
            self.ocr_event.wait(0.5)
            self.ocr_event.clear()
            if self.stop_event.is_set():
                return
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                continue
            try:
                text = self._read_text(frame)
                if len(text) >= 2 and self._plausible_ocr(text):
                    if self._candidate_text and ratio(text, self._candidate_text) > 85:
                        self._candidate_count += 1
                    else:
                        self._candidate_text = text
                        self._candidate_count = 1
                    if (self._candidate_count >= 2
                            and (not self._last_text or ratio(self._candidate_text, self._last_text) <= 85)):
                        self._last_text = self._candidate_text
                        print(f"OCR detected: {self._candidate_text}", flush=True)
                        self.on_text(self._candidate_text)
            except Exception as exc:
                print(f"OCR frame error: {exc}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        self.ocr_event.set()
