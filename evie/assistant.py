"""E.V.I.E. herself, with no microphone attached.

Everything that decides *what she says* lives here; everything that decides
*how you hear it* lives in loop.py. Keeping them apart means the interesting
half -- routing, brain swapping, fallback, memory -- is testable without a
sound card, and a bug in the voice stack can never be mistaken for a bug in
the thinking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from .brains import BrainRegistry, Context, Turn
from .config import Settings, load_identity, load_registry
from .memory import Vault
from .router import Action, Decision, route


@dataclass
class Reply:
    """What came back from one turn."""

    text: str
    brain: str
    action: Action
    notices: list[str]


class Assistant:
    def __init__(
        self,
        registry: BrainRegistry,
        settings: Settings,
        vault: Vault | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.vault = vault
        self.ctx = Context(system=self._identity())

    @classmethod
    def load(cls, settings: Settings | None = None) -> "Assistant":
        settings = settings or Settings.load()
        vault = Vault(settings.vault)
        return cls(load_registry(), settings, vault if vault.exists else None)

    def _identity(self) -> str:
        if self.vault and self.vault.exists and (found := self.vault.identity()):
            return found
        return load_identity(self.settings)

    # -- one turn --------------------------------------------------------

    async def respond(self, said: str) -> AsyncIterator[tuple[str, str]]:
        """Handle one utterance.

        Yields ``(kind, text)``:
          ``text``   -- response content, streamed
          ``notice`` -- something to tell the user (a fallback happened)
          ``meta``   -- control signal for the caller: "stop" or "quit"

        Commands handled by E.V.I.E. herself never reach a model, so a brain
        swap is instant and free.
        """
        decision: Decision = route(said, self.registry)

        if decision.action is Action.QUIT:
            yield "text", "Goodbye."
            yield "meta", "quit"
            return

        if decision.action is Action.STOP:
            yield "meta", "stop"
            return

        if decision.action is Action.REPLY:
            yield "text", decision.text
            self._remember(said, decision.text, "evie")
            return

        spoken: list[str] = []
        used = decision.brain or self.registry.active
        async for kind, chunk in self.registry.stream(
            decision.text, self.ctx, brain=decision.brain
        ):
            if kind == "text":
                spoken.append(chunk)
                yield "text", chunk
            else:
                yield "notice", chunk
                used = self.registry.active

        self._remember(said, "".join(spoken), used)

    def _remember(self, said: str, replied: str, brain: str) -> None:
        self.ctx.transcript.append(Turn("user", said))
        if replied.strip():
            self.ctx.transcript.append(Turn("assistant", replied.strip()))
        limit = self.settings.transcript_turns * 2
        if len(self.ctx.transcript) > limit:
            self.ctx.transcript = self.ctx.transcript[-limit:]
        if self.vault:
            self.vault.log_turn(said, replied, brain)

    # -- convenience -----------------------------------------------------

    async def ask(self, said: str) -> Reply:
        """Collect a whole turn. For the CLI and for tests."""
        parts: list[str] = []
        notices: list[str] = []
        action = Action.ANSWER
        async for kind, chunk in self.respond(said):
            if kind == "text":
                parts.append(chunk)
            elif kind == "notice":
                notices.append(chunk)
            elif chunk == "quit":
                action = Action.QUIT
            elif chunk == "stop":
                action = Action.STOP
        return Reply("".join(parts).strip(), self.registry.active, action, notices)
