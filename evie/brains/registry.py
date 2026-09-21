"""Which brain is talking, and what happens when it stops.

Three things live here:

  * resolution   -- "gemini", "google", "the fast one" all name one brain
  * hot swap     -- change brains mid-conversation without losing the thread
  * the fallback -- when a free tier runs dry, move to the next brain and
                    keep going, instead of handing the user an error

The fallback chain is what makes free tiers genuinely usable: ride the quota
until it is gone, then move down the list without the user doing anything.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable

from .base import (
    Brain,
    BrainError,
    BrainExhausted,
    BrainStatus,
    BrainUnavailable,
    Context,
)


class UnknownBrain(KeyError):
    """No brain answers to that name.

    KeyError renders its message with repr(), which turns a helpful sentence
    into `'unknown brain ...'` -- quotes and all. Override it so the CLI can
    print the message as written.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown brain"


@dataclass
class SwapResult:
    changed: bool
    brain: str
    previous: str | None = None

    def spoken(self) -> str:
        if not self.changed:
            return f"Already running on {self.brain}."
        return f"Switched to {self.brain}."


@dataclass
class _Counter:
    """Per-brain request tally, so you can see free-tier headroom."""

    day: _dt.date = field(default_factory=_dt.date.today)
    used: int = 0

    def bump(self) -> int:
        today = _dt.date.today()
        if today != self.day:
            self.day, self.used = today, 0
        self.used += 1
        return self.used


class BrainRegistry:
    def __init__(
        self,
        brains: dict[str, Brain],
        *,
        default: str,
        quick: str | None = None,
        fallback: Iterable[str] = (),
        aliases: dict[str, str] | None = None,
        daily_limits: dict[str, int] | None = None,
    ) -> None:
        if default not in brains:
            raise UnknownBrain(f"default brain {default!r} is not defined")
        self._brains = brains
        self._default = default
        self._active = default
        self.quick = quick if quick in brains else None
        self.fallback = [n for n in fallback if n in brains]
        self.daily_limits = daily_limits or {}
        self._counts: dict[str, _Counter] = defaultdict(_Counter)
        # Set by the loader. "Which file am I actually reading" has caused
        # three separate debugging sessions on this project alone.
        self.sources: list = []

        self._aliases: dict[str, str] = {n.lower(): n for n in brains}
        for alias, target in (aliases or {}).items():
            if target in brains:
                self._aliases[alias.lower().strip()] = target

    # -- lookup ----------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name.lower().strip() in self._aliases

    def names(self) -> list[str]:
        return list(self._brains)

    def aliases_for(self, name: str) -> list[str]:
        return sorted(a for a, t in self._aliases.items() if t == name and a != name)

    def resolve(self, name: str) -> str:
        """Map any alias to a canonical brain name."""
        key = name.lower().strip()
        if key in self._aliases:
            return self._aliases[key]
        # Tolerate speech-to-text noise: "switch to grok" for "groq".
        for alias, target in self._aliases.items():
            if alias in key or key in alias:
                return target
        raise UnknownBrain(
            f"no brain called {name!r}. Available: {', '.join(self._brains)}"
        )

    def get(self, name: str | None = None) -> Brain:
        return self._brains[self.resolve(name) if name else self._active]

    @property
    def active(self) -> str:
        return self._active

    @property
    def active_brain(self) -> Brain:
        return self._brains[self._active]

    # -- swapping --------------------------------------------------------

    def use(self, name: str) -> SwapResult:
        target = self.resolve(name)
        if target == self._active:
            return SwapResult(changed=False, brain=target)
        previous, self._active = self._active, target
        return SwapResult(changed=True, brain=target, previous=previous)

    def reset(self) -> SwapResult:
        return self.use(self._default)

    # -- usage tallies ---------------------------------------------------

    def used_today(self, name: str) -> int:
        return self._counts[name].used if name in self._counts else 0

    def headroom(self, name: str) -> str:
        used = self.used_today(name)
        limit = self.daily_limits.get(name)
        return f"{used}/{limit}" if limit else str(used)

    # -- the chain -------------------------------------------------------

    def _chain(self, start: str) -> list[str]:
        ordered = [start] + [n for n in self.fallback if n != start]
        return [n for n in ordered if n in self._brains]

    async def stream(
        self, prompt: str, ctx: Context, *, brain: str | None = None
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream an answer, moving down the fallback chain if a brain gives out.

        Yields ``(kind, text)`` where kind is ``"text"`` for response content or
        ``"notice"`` for something the user should be told out loud, such as a
        brain having been swapped out from under them.

        A brain that has already emitted text is never abandoned mid-answer --
        switching there would splice two different replies together.
        """
        start = self.resolve(brain) if brain else self._active
        errors: list[str] = []

        for index, name in enumerate(self._chain(start)):
            impl = self._brains[name]
            produced = False

            if index > 0:
                yield "notice", f"{errors[-1]} Switching to {name}."

            try:
                self._counts[name].bump()
                async for chunk in impl.stream(prompt, ctx):
                    produced = True
                    yield "text", chunk
            except BrainError as exc:
                if produced or not exc.retryable_elsewhere:
                    raise
                errors.append(_explain(name, exc))
                continue

            if index > 0:
                self._active = name  # a fallback that worked becomes the new default
            return

        raise BrainUnavailable(
            "every brain failed: " + "; ".join(errors) if errors else "no brains configured"
        )

    # -- health ----------------------------------------------------------

    async def health(self) -> dict[str, BrainStatus]:
        return {name: await impl.health() for name, impl in self._brains.items()}


def _explain(name: str, exc: BrainError) -> str:
    """Say why, not just that.

    Spoken aloud a bare reason is right, but the same string ends up in the
    CLI error when every brain fails -- and "claude is unavailable" gives
    nobody anything to act on. Keep the detail the brain already produced.
    """
    reason = str(exc).strip()
    if isinstance(exc, BrainExhausted):
        headline = f"{name} is rate limited"
    elif isinstance(exc, BrainUnavailable):
        headline = f"{name} is unavailable"
    else:
        headline = f"{name} failed"
    return f"{headline} ({reason})." if reason and reason not in headline else f"{headline}."
