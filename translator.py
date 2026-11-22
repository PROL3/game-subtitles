"""Latest-only, streaming Ollama translator."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from typing import Awaitable, Callable

import httpx


SYSTEM_PROMPT = (
    "Translate only the newest English subtitle line into natural, fluent spoken Israeli Hebrew, "
    "fit for GTA dialogue and slang. Output ONLY the Hebrew translation: no notes, explanations, "
    "quotes, or prefixes. Keep proper nouns, brand names, and game-specific terms in English "
    "when that sounds natural. Keep names, brands, locations, and game terms such as "
    "Michael, Los Santos, and Remote Play in Latin English spelling inside the Hebrew sentence. "
    "Do not reverse, transliterate, or scramble those English terms."
)


class LatestOnlyTranslator:
    def __init__(self, model: str, on_start: Callable[[str], Awaitable[None]],
                 on_token: Callable[[str], Awaitable[None]], on_done: Callable[[str], Awaitable[None]],
                 base_url: str = "http://localhost:11434", provider: str = "ollama",
                 openai_model: str = "gpt-5.4-mini", openai_api_key: str | None = None):
        self.model = model
        self.provider = provider.lower()
        self.openai_model = openai_model
        self.on_start = on_start
        self.on_token = on_token
        self.on_done = on_done
        self.url = base_url.rstrip("/") + "/api/chat"
        self.history: deque[str] = deque(maxlen=3)
        self.active: asyncio.Task | None = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0))
        self.openai_api_key = openai_api_key

    def submit(self, line: str) -> None:
        if self.active and not self.active.done():
            self.active.cancel()
        self.history.append(line)
        self.active = asyncio.create_task(self._translate(line))

    @staticmethod
    def _meaningful_translation(text: str, source: str) -> bool:
        if not text or text.startswith("["):
            return False
        hebrew = re.findall(r"[\u0590-\u05FF]", text)
        source_words = re.findall(r"[A-Za-z]{2,}", source)
        # A normal translated subtitle should contain at least two Hebrew letters.
        # English game names are allowed, but an English-only response is rejected.
        if len(hebrew) < 2:
            return False
        if len(source_words) >= 2 and len(text.strip()) < 2:
            return False
        if re.search(r"(.)\1\1\1", text, re.IGNORECASE):
            return False
        return True

    async def _emit_done(self, source: str, output: str) -> None:
        if self._meaningful_translation(output.strip(), source):
            await self.on_done(output.strip())
        else:
            # Clear any partial stream rather than leaving meaningless text on screen.
            await self.on_done("")

    async def _translate(self, line: str) -> None:
        await self.on_start(line)
        context = list(self.history)[:-1]
        context_text = "\n".join(context) if context else "(none)"
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                     "Previous English context (do not translate):\n"
                     + context_text
                     + "\n\nNEWEST SUBTITLE TO TRANSLATE:\n<<<\n"
                     + line
                     + "\n>>>\n\nReturn only the Hebrew translation of the text between <<< and >>>."}]
        try:
            if self.provider == "openai":
                await self._translate_openai(line, context_text)
            else:
                await self._translate_ollama(line, messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.on_done(f"[Translation error: {exc}]")

    async def _translate_ollama(self, line: str, messages: list[dict]) -> None:
        output = ""
        try:
            async with self.client.stream("POST", self.url, json={
                "model": self.model,
                "stream": True,
                "messages": messages,
                "options": {"temperature": 0.1, "top_p": 0.8, "num_predict": 80},
            }) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Ollama HTTP {response.status_code}: {body}")
                async for raw in response.aiter_lines():
                    if not raw:
                        continue
                    data = json.loads(raw)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        output += token
                        await self.on_token(token)
                    if data.get("done"):
                        break
            if output.strip():
                await self._emit_done(line, output)
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

    async def _translate_openai(self, line: str, context_text: str) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OpenAI provider is not configured")
        prompt = (
            "Previous English context (do not translate):\n" + context_text +
            "\n\nNEWEST SUBTITLE TO TRANSLATE:\n<<<\n" + line +
            "\n>>>\n\nReturn only the Hebrew translation of the text between <<< and >>>."
        )
        output = ""
        async with self.client.stream(
            "POST", "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={
                "model": self.openai_model,
                "instructions": SYSTEM_PROMPT,
                "input": prompt,
                "stream": True,
                "max_output_tokens": 80,
                "reasoning": {"effort": "none"},
            },
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenAI HTTP {response.status_code}: {body}")
            async for raw in response.aiter_lines():
                if not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("type") == "response.output_text.delta":
                    token = event.get("delta", "")
                    output += token
                    await self.on_token(token)
        if output.strip():
            await self._emit_done(line, output)

    async def close(self) -> None:
        if self.active and not self.active.done():
            self.active.cancel()
            await asyncio.gather(self.active, return_exceptions=True)
        await self.client.aclose()
