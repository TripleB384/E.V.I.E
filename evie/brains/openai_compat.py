"""Brains that are an HTTP endpoint speaking the OpenAI chat schema.

Groq, Ollama, OpenRouter, GitHub Models and Google's OpenAI-compat endpoint all
accept the same request shape, so one class covers every one of them. Adding a
provider is a brains.yaml entry, not code.

These are the fast lane: a single completion call, no agent loop, no tools.
Use them for questions; use a CLI brain for work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from .base import (
    Brain,
    BrainExhausted,
    BrainRefused,
    BrainStatus,
    BrainUnavailable,
    Context,
    Health,
)


@dataclass
class HttpBrainSpec:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    agentic: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 120.0
    aliases: tuple[str, ...] = ()


class OpenAICompatBrain:
    def __init__(self, spec: HttpBrainSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.agentic = spec.agentic

    @property
    def _key(self) -> str | None:
        if not self.spec.api_key_env:
            return None
        return os.environ.get(self.spec.api_key_env)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if key := self._key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def stream(self, prompt: str, ctx: Context) -> AsyncIterator[str]:
        if self.spec.api_key_env and not self._key:
            raise BrainUnavailable(f"${self.spec.api_key_env} is not set")

        body: dict[str, object] = {
            "model": self.spec.model,
            "messages": ctx.as_messages() + [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": self.spec.temperature,
        }
        if self.spec.max_tokens:
            body["max_tokens"] = self.spec.max_tokens

        url = self.spec.base_url.rstrip("/") + "/chat/completions"
        timeout = httpx.Timeout(self.spec.timeout, connect=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, json=body, headers=self._headers()
                ) as resp:
                    if resp.status_code != 200:
                        raise _from_status(
                            resp.status_code, (await resp.aread()).decode(errors="replace")
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload in ("", "[DONE]"):
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        for choice in chunk.get("choices") or []:
                            text = (choice.get("delta") or {}).get("content")
                            if text:
                                yield text
        except httpx.ConnectError as exc:
            raise BrainUnavailable(f"cannot reach {self.spec.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise BrainUnavailable(f"{self.name} timed out") from exc

    async def health(self) -> BrainStatus:
        if self.spec.api_key_env and not self._key:
            return BrainStatus(Health.UNAUTHENTICATED, f"${self.spec.api_key_env} not set")
        url = self.spec.base_url.rstrip("/") + "/models"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return BrainStatus(Health.MISSING, f"unreachable: {type(exc).__name__}")
        if resp.status_code in (401, 403):
            return BrainStatus(Health.UNAUTHENTICATED, "key rejected")
        if resp.status_code == 429:
            return BrainStatus(Health.EXHAUSTED, "rate limited")
        if resp.status_code >= 400:
            # Not every provider exposes /models; reachable is good enough.
            return BrainStatus(Health.UNKNOWN, f"HTTP {resp.status_code} from /models")
        return BrainStatus(Health.OK, self.spec.model)


def _from_status(status: int, body: str) -> Exception:
    snippet = body.strip()[:400]
    if status == 429:
        return BrainExhausted(f"rate limited: {snippet}")
    if status in (401, 403):
        return BrainUnavailable(f"auth rejected: {snippet}")
    return BrainRefused(f"HTTP {status}: {snippet}")


_: type[Brain] = OpenAICompatBrain
