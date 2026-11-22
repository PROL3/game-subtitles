# Real-time game subtitle translator

Captures an English subtitle ROI from a PC window, OCRs it locally, translates the newest line to Hebrew with local NLLB-200/CTranslate2, and serves the result to phones on the LAN.

## Fast local NLLB server

The recommended implementation is now `server.py`. It uses RapidOCR plus
offline NLLB-200 `en->he` translation through CTranslate2, an LRU cache, latest-frame processing, and
the bottom-center subtitle preset:

```powershell
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python server.py
```

The first start downloads `facebook/nllb-200-distilled-600M` and converts it to
an INT8 CTranslate2 model. This is a one-time setup; after that translation
runs locally. OCR and translation run in separate threads with a small
bounded queue, so fast subtitle changes do not block capture while stale
backlog is discarded. The server prints the phone URL. To
select a manual ROI on a particular monitor:

```powershell
python server.py --select-roi --monitor 2
```

## Windows setup

Install Python 3.10 or 3.11 and Ollama, then pull the default model. RapidOCR's
current Windows wheel does not support Python 3.12+ (including 3.14), so check
which interpreters are installed with `py --list`:

```powershell
py --list
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
python main.py
```

Copy `.env.example` to `.env` and edit it to choose the translator:
`OLLAMA_LOCAL=on` uses local Ollama; `OLLAMA_LOCAL=off` uses OpenAI with
`OPENAI_API_KEY`. Run `python main.py --select-roi` when two monitors are
connected; the selector now shows the complete Windows virtual desktop.
To select only one display, use `--monitor 1` or `--monitor 2`:

```powershell
python main.py --select-roi --monitor 2
```

The selected monitor is saved in `roi.json` and reused on normal starts.

For automatic game-subtitle framing without a saved ROI, use the bottom-center
preset. Adjust `ROI_PRESET=bottom-center` and optionally select a monitor with
`CAPTURE_MONITOR=1`. Use `ROI_PRESET=saved` to use `roi.json` instead.

### Better translation with OpenAI

The translator supports both local Ollama and OpenAI. OpenAI is more accurate
but uses a paid API key; keep the key in an environment variable, not in the
source code:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python main.py --provider openai --openai-model gpt-5.4-mini
```

The current OpenAI path uses the Responses API with streaming and cancels the
active request when a newer subtitle arrives. The official API supports
streaming responses, and `gpt-5.4-mini` is positioned for lower-latency,
high-volume workloads. See the [OpenAI streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses)
and [model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

### Free local option

Ollama remains free and offline. For better Hebrew than the small 3B model,
try a larger local model if the PC has enough RAM:

```powershell
ollama pull qwen2.5:7b
python main.py --model qwen2.5:7b
```

The mobile page includes Screen Wake Lock and a local muted-video/NoSleep
fallback. Tap the on-screen button once after opening the page; this user
gesture is required by mobile autoplay policies.
Browsers generally require HTTPS (or localhost) to grant Wake Lock, so a plain
`http://10.x.x.x:8000` page cannot guarantee that the phone stays awake. For
reliable Wake Lock, use local HTTPS or increase the phone's screen timeout.

Run each command separately. If Python 3.11 is not installed, install it from
python.org, then rerun the commands above. If you already created `.venv` with
Python 3.14, remove that environment and recreate it with Python 3.11:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On the first run, draw the subtitle rectangle in the OpenCV window. The coordinates are saved to `roi.json`. Re-select with `python main.py --select-roi`. The terminal prints the exact phone URL; both devices must be on the same LAN and Windows Firewall must allow Python on private networks.

Useful options:

```text
--model MODEL       Ollama model (or OLLAMA_MODEL)
--ollama-url URL    Ollama base URL (or OLLAMA_URL)
--port 8000         Web server port
--select-roi        Reopen ROI selection
```

The capture loop runs at roughly 7 FPS, performs a cheap pixel-diff/debounce before OCR, and uses a latest-only asyncio task so stale translations are cancelled when dialogue changes.
