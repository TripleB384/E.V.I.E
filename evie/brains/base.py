"""The seam.

Every other part of E.V.I.E. depends on this module and nothing deeper. A brain
is anything that can turn a prompt into a stream of text. Whether that happens
by shelling out to a subscription-backed CLI or by POSTing to an HTTP endpoint
is an implementation detail the rest of the codebase never sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable


class BrainError(Exception):
    """Base for brain failures. Carries whether falling back is worth trying."""

    retryable_elsewhere = True


class BrainUnavailable(BrainError):
    """Not installed, not authenticated, or not listening."""


class BrainExhausted(BrainError):
    """Rate limited or out of quota. The usual reason to move down the chain."""


class BrainRefused(BrainError):
    """The brain answered, but with an error. Another brain probably fails too."""

    retryable_elsewhere = False


class Health(Enum):
    OK = "ok"
    UNAUTHENTICATED = "unauthenticated"
    MISSING = "missing"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


@dataclass
class BrainStatus:
    health: Health
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.health is Health.OK


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class Context:
    """What a brain needs to answer in character and in context.

    `transcript` is E.V.I.E.'s own canonical history, kept independently of any
    provider. It is what makes a mid-conversation brain swap survivable: the
    incoming brain has never seen the conversation, so we replay a compact form
    of this into it.
    """

    system: str = ""
    transcript: list[Turn] = field(default_factory=list)
    # brain name -> that provider's native session id, when it has one.
    session_ids: dict[str, str] = field(default_factory=dict)

    def recent(self, limit: int = 12) -> list[Turn]:
        return self.transcript[-limit:]

    def as_messages(self, limit: int = 12) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        msgs += [{"role": t.role, "content": t.text} for t in self.recent(limit)]
        return msgs

    def handoff_summary(self, limit: int = 6) -> str:
        """A compact replay for a brain that has never seen this conversation."""
        turns = self.recent(limit)
        if not turns:
            return ""
        lines = [f"{'You' if t.role == 'assistant' else 'User'}: {t.text}" for t in turns]
        return "Earlier in this conversation:\n" + "\n".join(lines)


@runtime_checkable
class Brain(Protocol):
    name: str
    agentic: bool  # can it read files, run commands, use tools?

    def stream(self, prompt: str, ctx: Context) -> AsyncIterator[str]:
        """Yield response text incrementally. Raise a BrainError on failure."""
        ...

    async def health(self) -> BrainStatus:
        ...
