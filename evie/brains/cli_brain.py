"""Brains that are a subprocess.

This is the trick that makes E.V.I.E. cheap: `claude -p`, `codex exec` and
`gemini -p` authenticate with a subscription you already pay for (or a free
tier), so a request costs nothing at the margin. They also carry a full agent
loop, which means these brains can read files, run commands and call MCP
servers -- they can actually do the work, not just talk about it.

Providers differ only in how they print their output, so the difference lives
in a `parser` field in brains.yaml rather than in a subclass per vendor.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import AsyncIterator

from .base import (
    Brain,
    BrainExhausted,
    BrainRefused,
    BrainStatus,
    BrainUnavailable,
    Context,
    Health,
)

# Substrings that mean "this brain is out of quota", not "this brain is broken".
_EXHAUSTED_HINTS = (
    "rate limit",
    "rate_limit",
    "quota",
    "resource_exhausted",
    "429",
    "usage limit",
    "too many requests",
)
_AUTH_HINTS = ("not authenticated", "unauthorized", "401", "please log in", "/login")


@dataclass
class CliBrainSpec:
    name: str
    command: list[str]
    agentic: bool = True
    parser: str = "plain"  # "plain" | "claude_stream_json"
    resume_flag: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: float = 300.0
    aliases: tuple[str, ...] = ()


class CliBrain:
    """Runs a CLI as a brain, streaming its stdout back as it arrives."""

    def __init__(self, spec: CliBrainSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.agentic = spec.agentic

    # -- command assembly ------------------------------------------------

    def _argv(self, prompt: str, ctx: Context) -> list[str]:
        argv = [a.replace("{prompt}", prompt) for a in self.spec.command]
        session = ctx.session_ids.get(self.name)
        if self.spec.resume_flag and session:
            argv += [self.spec.resume_flag, session]
        return argv

    def _prompt_for(self, prompt: str, ctx: Context) -> str:
        """Give a brain that has never seen this conversation enough to follow it.

        A CLI brain resuming its own native session already has the history, so
        we send the bare prompt. Anything else gets a compact replay.
        """
        if self.spec.resume_flag and ctx.session_ids.get(self.name):
            return prompt
        parts = [p for p in (ctx.system, ctx.handoff_summary(), prompt) if p]
        return "\n\n".join(parts)

    # -- streaming -------------------------------------------------------

    async def stream(self, prompt: str, ctx: Context) -> AsyncIterator[str]:
        if shutil.which(self.spec.command[0]) is None:
            raise BrainUnavailable(f"{self.spec.command[0]!r} is not on PATH")

        argv = self._argv(self._prompt_for(prompt, ctx), ctx)
        env = {**os.environ, **(self.spec.env or {})}
        cwd = os.path.expanduser(self.spec.cwd) if self.spec.cwd else None
        if cwd and not os.path.isdir(cwd):
            os.makedirs(cwd, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        parse = (
            self._parse_claude_stream_json
            if self.spec.parser == "claude_stream_json"
            else self._parse_plain
        )

        try:
            async for chunk in parse(proc.stdout, ctx):
                yield chunk
        except asyncio.CancelledError:
            proc.kill()
            raise

        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()

        if proc.returncode not in (0, None):
            stderr = (await proc.stderr.read()).decode(errors="replace")
            raise _classify(stderr or f"{self.name} exited {proc.returncode}")

    async def _parse_plain(self, stdout, ctx: Context) -> AsyncIterator[str]:
        while True:
            block = await stdout.read(256)
            if not block:
                return
            yield block.decode(errors="replace")

    async def _parse_claude_stream_json(self, stdout, ctx: Context) -> AsyncIterator[str]:
        """Parse `claude -p --output-format stream-json --include-partial-messages`.

        Text arrives as `text_delta` events. Subagent chatter carries a
        non-null `parent_tool_use_id`, so we drop it -- the user should hear the
        answer, not the assistant talking to itself.
        """
        async for raw in stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = msg.get("type")

            if kind == "stream_event":
                if msg.get("parent_tool_use_id") is not None:
                    continue
                delta = (msg.get("event") or {}).get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]

            elif kind == "result":
                if sid := msg.get("session_id"):
                    ctx.session_ids[self.name] = sid
                if msg.get("is_error") or msg.get("subtype") not in (None, "success"):
                    raise _classify(str(msg.get("result") or msg.get("subtype")))

            elif kind == "system" and msg.get("subtype") == "api_retry":
                if msg.get("error") == "rate_limit":
                    raise BrainExhausted(f"{self.name} is rate limited")

    # -- health ----------------------------------------------------------

    def describe(self) -> str:
        """What this brain is, without claiming a model id we do not have.

        Which model the CLI picks is its own business -- it comes from that
        tool's config and can change between calls -- so naming one here
        would be a guess dressed up as a fact.
        """
        return f"the {self.spec.command[0]} command-line tool"

    def missing(self) -> str | None:
        exe = self.spec.command[0]
        return None if shutil.which(exe) else f"{exe} is not installed"

    async def health(self) -> BrainStatus:
        """Report what we actually know, which is less than you'd like.

        Whether a subscription CLI is logged in can only be settled by running
        it, and running it costs tokens -- so a health check that promised
        "ready" would either be lying or quietly spending your quota. It says
        "installed" instead, and the first real call reports the truth.
        """
        exe = self.spec.command[0]
        if shutil.which(exe) is None:
            return BrainStatus(Health.MISSING, f"{exe} not installed")
        return BrainStatus(Health.UNVERIFIED, f"{exe} installed, login not checked")


def _classify(message: str) -> Exception:
    low = message.lower()
    if any(h in low for h in _EXHAUSTED_HINTS):
        return BrainExhausted(message.strip()[:400])
    if any(h in low for h in _AUTH_HINTS):
        return BrainUnavailable(message.strip()[:400])
    return BrainRefused(message.strip()[:400])


_: type[Brain] = CliBrain  # structural conformance check
