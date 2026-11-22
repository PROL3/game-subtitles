"""Low-latency local English subtitle -> Hebrew server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import threading
import time
from collections import OrderedDict
from queue import Empty, Full, Queue
from pathlib import Path

import cv2
import mss
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from rapidfuzz.fuzz import ratio
from rapidocr_onnxruntime import RapidOCR

from nllb_translator import NLLBTranslator


BASE_DIR = Path(__file__).parent
ROI_PATH = BASE_DIR / "roi.json"
INDEX_PATH = BASE_DIR / "index.html"


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def select_region(monitor_arg: str) -> dict:
    with mss.mss() as sct:
        if monitor_arg == "all":
            target, desktop = sct.monitors[0], True
        else:
            index = int(monitor_arg)
            if index <= 0 or index >= len(sct.monitors):
                raise ValueError(f"Monitor {index} unavailable; available monitors: 1-{len(sct.monitors)-1}")
            target, desktop = sct.monitors[index], False
        image = np.asarray(sct.grab(target))[:, :, :3]
    x, y, w, h = cv2.selectROI("Select subtitle ROI", image, showCrosshair=True)
    cv2.destroyAllWindows()
    if not w or not h:
        raise RuntimeError("No ROI selected")
    roi = {"desktop": desktop, "monitor": None if desktop else int(monitor_arg),
           "left": int(x), "top": int(y), "width": int(w), "height": int(h)}
    ROI_PATH.write_text(json.dumps(roi, indent=2), encoding="utf-8")
    return roi


def load_region(args) -> dict:
    if args.select_roi:
        return select_region(args.monitor)
    if args.roi_preset == "bottom-center":
        with mss.mss() as sct:
            monitor = sct.monitors[0] if args.monitor == "all" else sct.monitors[int(args.monitor)]
        width = int(monitor["width"] * 0.80)
        height = int(monitor["height"] * 0.24)
        return {"desktop": args.monitor == "all", "monitor": None if args.monitor == "all" else int(args.monitor),
                "left": int((monitor["width"] - width) / 2), "top": int(monitor["height"] * 0.70),
                "width": width, "height": height}
    if not ROI_PATH.exists():
        return select_region(args.monitor)
    return json.loads(ROI_PATH.read_text(encoding="utf-8"))


class TranslationCache:
    def __init__(self, max_size: int = 512):
        self.items: OrderedDict[str, str] = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.max_size:
                self.items.popitem(last=False)


class SubtitleEngine:
    def __init__(self, on_translation, fps: float, diff_threshold: float):
        self.on_translation = on_translation
        self.interval = 1.0 / fps
        self.diff_threshold = diff_threshold
        self.ocr = RapidOCR()
        self.cache = TranslationCache()
        self.stop = threading.Event()
        self.new_frame = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.previous_small: np.ndarray | None = None
        self.last_ocr = ""
        self.translator = NLLBTranslator()
        queue_size = max(1, int(os.getenv("TRANSLATION_QUEUE_SIZE", "4")))
        self.translation_queue: Queue[str | None] = Queue(maxsize=queue_size)

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 145, 255]))
        yellow = cv2.inRange(hsv, np.array([12, 40, 100]), np.array([50, 255, 255]))
        bright = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1]
        return cv2.resize(cv2.bitwise_or(cv2.bitwise_or(white, yellow), bright), None,
                          fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    def changed(self, frame: np.ndarray) -> bool:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 32), interpolation=cv2.INTER_AREA)
        if self.previous_small is None:
            self.previous_small = small
            return True
        mse = float(np.mean((small.astype(np.float32) - self.previous_small.astype(np.float32)) ** 2))
        self.previous_small = small
        return mse >= self.diff_threshold

    def read_text(self, frame: np.ndarray) -> str:
        result, _ = self.ocr(self.preprocess(frame))
        blocks = []
        for row in result or []:
            if len(row) >= 3 and float(row[2]) < 0.55:
                continue
            box, text = row[0], str(row[1]).strip()
            if text:
                box = np.asarray(box)
                blocks.append((float(box[:, 1].mean()), float(box[:, 0].mean()), text))
        blocks.sort(key=lambda item: (item[0], item[1]))
        return " ".join(item[2] for item in blocks)

    @staticmethod
    def plausible(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text)
        words = re.findall(r"[A-Za-z]{2,}", text)
        return bool(words and len(letters) >= 2 and len(letters) / max(len(re.sub(r"\s", "", text)), 1) >= 0.55)

    def translate(self, english: str) -> tuple[str, bool]:
        key = normalize(english)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        hebrew = self.translator.translate(english)
        self.cache.put(key, hebrew)
        return hebrew, False

    def enqueue_translation(self, text: str | None) -> None:
        """Queue a subtitle without blocking OCR; discard the oldest on overflow."""
        try:
            self.translation_queue.put_nowait(text)
        except Full:
            try:
                self.translation_queue.get_nowait()
            except Empty:
                pass
            try:
                self.translation_queue.put_nowait(text)
            except Full:
                pass

    def translation_loop(self) -> None:
        while not self.stop.is_set():
            try:
                text = self.translation_queue.get(timeout=0.5)
            except Empty:
                continue
            if text is None:
                self.translation_queue.task_done()
                return
            try:
                hebrew, cached = self.translate(text)
                if hebrew:
                    print(f"OCR: {text} -> HE: {hebrew}{' (cache)' if cached else ''}", flush=True)
                    self.on_translation(text, hebrew, cached)
            except Exception as exc:
                print(f"Translation error: {exc}", flush=True)
            finally:
                self.translation_queue.task_done()

    def capture_loop(self, roi: dict) -> None:
        with mss.mss() as sct:
            monitor = sct.monitors[0] if roi.get("desktop") else sct.monitors[int(roi.get("monitor", 1))]
            region = {"left": monitor["left"] + roi["left"], "top": monitor["top"] + roi["top"],
                      "width": roi["width"], "height": roi["height"]}
            while not self.stop.is_set():
                started = time.monotonic()
                frame = np.asarray(sct.grab(region))[:, :, :3]
                if self.changed(frame):
                    with self.lock:
                        self.latest_frame = frame.copy()
                    self.new_frame.set()
                self.stop.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def ocr_loop(self) -> None:
        while not self.stop.is_set():
            self.new_frame.wait(0.5)
            self.new_frame.clear()
            if self.stop.is_set():
                return
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                continue
            try:
                text = " ".join(self.read_text(frame).split())
                if len(text) < 2 or not self.plausible(text) or (self.last_ocr and ratio(text, self.last_ocr) > 85):
                    continue
                self.last_ocr = text
                self.enqueue_translation(text)
            except Exception as exc:
                print(f"OCR error: {exc}", flush=True)

    def start(self, roi: dict) -> None:
        threading.Thread(target=self.capture_loop, args=(roi,), daemon=True).start()
        threading.Thread(target=self.ocr_loop, daemon=True).start()
        threading.Thread(target=self.translation_loop, daemon=True).start()


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for client in tuple(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)


def local_url(port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    return f"http://{host}:{port}"


def build_app(args) -> FastAPI:
    app = FastAPI()
    hub = Hub()
    engine_holder = {"engine": None}

    @app.on_event("startup")
    async def startup():
        loop = asyncio.get_running_loop()
        engine = SubtitleEngine(
            lambda en, he, cached: asyncio.run_coroutine_threadsafe(
                hub.broadcast({"type": "subtitle", "english": en, "hebrew": he, "cached": cached}), loop),
            args.fps, args.diff_threshold,
        )
        roi = load_region(args)
        engine.start(roi)
        engine_holder["engine"] = engine
        print(f"Capture: {roi} at {args.fps:g} FPS", flush=True)

    @app.get("/")
    async def index():
        return FileResponse(INDEX_PATH)

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        hub.clients.add(socket)
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            hub.clients.discard(socket)

    @app.on_event("shutdown")
    async def shutdown():
        engine = engine_holder["engine"]
        if engine:
            engine.stop.set()
            engine.new_frame.set()
            engine.enqueue_translation(None)

    print(f"Mobile URL: {local_url(args.port)}", flush=True)
    return app


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-roi", action="store_true")
    parser.add_argument("--monitor", choices=["all", "1", "2", "3", "4"], default=os.getenv("CAPTURE_MONITOR", "all"))
    parser.add_argument("--roi-preset", choices=["bottom-center", "saved"], default=os.getenv("ROI_PRESET", "bottom-center"))
    parser.add_argument("--fps", type=float, default=float(os.getenv("CAPTURE_FPS", "8")))
    parser.add_argument("--diff-threshold", type=float, default=float(os.getenv("FRAME_DIFF_THRESHOLD", "3.0")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
