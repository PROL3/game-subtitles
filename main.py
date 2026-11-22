from __future__ import annotations

import argparse
import asyncio
import os
import socket
import threading
from collections import deque
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from capture import SubtitleCapture, load_or_select_roi
from translator import LatestOnlyTranslator


class State:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.previous: deque[str] = deque(maxlen=3)
        self.active = ""
        self.lock = asyncio.Lock()

    async def broadcast(self, payload: dict):
        dead = []
        for client in tuple(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)


def lan_url(port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    return f"http://{host}:{port}"


async def run(args):
    state = State()
    loop = asyncio.get_running_loop()

    async def on_start(line: str):
        if state.active:
            state.previous.append(state.active)
        state.active = ""
        await state.broadcast({"type": "start", "english": line, "history": list(state.previous)})

    async def on_token(token: str):
        state.active += token
        await state.broadcast({"type": "token", "token": token, "active": state.active})

    async def on_done(text: str):
        state.active = text
        await state.broadcast({"type": "done", "text": text, "history": list(state.previous)})

    translator = LatestOnlyTranslator(
        args.model, on_start, on_token, on_done, args.ollama_url,
        provider=args.provider, openai_model=args.openai_model, openai_api_key=args.openai_api_key,
    )
    capture = SubtitleCapture(
        load_or_select_roi(args.select_roi, Path(args.roi), args.monitor),
        lambda text: translator.submit(text),
        fps=float(os.getenv("CAPTURE_FPS", "8")),
    )
    capture_thread = threading.Thread(target=capture.run, daemon=True)
    # submit() is called by the capture thread; marshal task creation onto the event loop.
    original_submit = translator.submit
    translator.submit = lambda text: loop.call_soon_threadsafe(original_submit, text)
    capture_target = "all monitors" if capture.roi.get("desktop") else f"monitor {capture.roi.get('monitor', 1)}"
    print(f"Capture ROI: {capture_target}, left={capture.roi['left']} top={capture.roi['top']} width={capture.roi['width']} height={capture.roi['height']}", flush=True)
    capture_thread.start()

    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        state.clients.add(ws)
        await ws.send_json({"type": "state", "active": state.active, "history": list(state.previous)})
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            state.clients.discard(ws)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    print(f"Mobile subtitle URL: {lan_url(args.port)}", flush=True)
    print(f"Translation provider: {args.provider} ({args.openai_model if args.provider == 'openai' else args.model})", flush=True)
    try:
        await server.serve()
    finally:
        capture.stop()
        await translator.close()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local real-time game subtitle translator")
    parser.add_argument("--select-roi", action="store_true", help="Select the subtitle rectangle again")
    parser.add_argument("--monitor", choices=["all", "1", "2", "3", "4"], default=os.getenv("CAPTURE_MONITOR", "all"),
                        help="Monitor for ROI selection: all, 1, 2, ...")
    parser.add_argument("--roi", default="roi.json")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--provider", choices=["ollama", "openai"], default=None)
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    args.provider = args.provider or ("ollama" if os.getenv("OLLAMA_LOCAL", "on").lower() in {"1", "true", "yes", "on"} else "openai")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
