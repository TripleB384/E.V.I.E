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
import re
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

    def missing(self) -> str | None:
        if self.spec.api_key_env and not self._key:
            return f"${self.spec.api_key_env} is not set"
        return None

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


# Providers disagree about which status an invalid key deserves. Google
# returns 400 ("Please pass a valid API key"), most others 401. Classifying on
# the status alone made a bad Google key a BrainRefused -- fatal, no fallback
# -- so one wrong credential killed a request the chain could have answered.
# Two independent signals, checked in either order, because providers phrase
# this every possible way: Google says "Please pass a valid API key" (problem
# word first), most others say "invalid api key" (noun first). Requiring a
# fixed order missed Google entirely.
#
# Bare "token" is deliberately excluded from the nouns: "max_tokens is
# required" would otherwise read as an auth failure.
_AUTH_NOUN = re.compile(
    r"\b(?:api[ _-]?key|credential|authenticat\w+|authori[sz]\w+|"
    r"(?:access|bearer|auth)[ _-]token)\b",
    re.IGNORECASE,
)
_AUTH_PROBLEM = re.compile(
    r"\b(?:invalid|valid|expired|missing|required|incorrect|bad|malformed|"
    r"unauthori[sz]ed|rejected|denied|forbidden|provide|pass)\b",
    re.IGNORECASE,
)
_MODEL_TROUBLE = re.compile(
    r"\bmodel\b[^.]{0,80}?"
    r"\b(?:not found|no longer|unavailable|does not exist|doesn'?t exist|"
    r"deprecated|retired|unsupported|invalid|unknown|decommissioned)\b"
    r"|\b(?:unknown|invalid|unsupported|no such)\b[^.]{0,20}?\bmodel\b",
    re.IGNORECASE,
)
_QUOTA_TROUBLE = re.compile(
    r"\b(?:quota|rate.?limit|exhausted|too many requests|billing|"
    r"insufficient.{0,20}(?:credit|balance|fund))\b",
    re.IGNORECASE,
)


def _from_status(status: int, body: str) -> Exception:
    snippet = body.strip()[:400]

    if status == 429 or _QUOTA_TROUBLE.search(body):
        return BrainExhausted(f"out of quota: {snippet}")

    if status in (401, 403) or (_AUTH_NOUN.search(body) and _AUTH_PROBLEM.search(body)):
        # Worth being specific: the provider's own wording does not
        # distinguish a typo from an expired key from one pasted into the
        # wrong variable, and all three look identical from here.
        return BrainUnavailable(
            f"the API key was rejected — check it is the right provider's key, "
            f"has no stray quotes or spaces, and has not expired. "
            f"Provider said: {snippet}"
        )

    if status == 404 or _MODEL_TROUBLE.search(body):
        # Per-provider configuration, not a bad request. Google retiring
        # gemini-2.5-flash says nothing about whether Groq can answer, so
        # ending the turn here wastes a working fallback chain.
        return BrainUnavailable(
            f"this provider will not serve that model — update `model:` for "
            f"this brain in brains.yaml. Provider said: {snippet}"
        )

    return BrainRefused(f"HTTP {status}: {snippet}")


_: type[Brain] = OpenAICompatBrain
