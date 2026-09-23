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
        self._base_identity = self._identity()
        self.ctx = Context(system=self._base_identity)

    @classmethod
    def load(cls, settings: Settings | None = None) -> "Assistant":
        settings = settings or Settings.load()
        vault = Vault(settings.vault)
        return cls(load_registry(), settings, vault if vault.exists else None)

    # EVIE.md tells her to write things into the vault. A brain with no file
    # access cannot, so it narrates the write instead -- one real reply ended
    # with a fabricated "*Vault note:* User asked about the current model."
    # Nothing was written; it was describing an action it could not take.
    _NO_HANDS = (
        "\n\nFor this reply you have no file access: you cannot read or write "
        "the vault. Answer from what you have been given. Never describe "
        "saving, noting or filing anything — if something is worth keeping, "
        "say so in one short clause and let it be written for you. No "
        "notes-to-self, no restating the question back, no parenthetical "
        "commentary about what was asked: the reply is spoken out loud, and "
        "bookkeeping read aloud is noise."
    )

    # A model asked what it is answers from its training data, which describes
    # the weights and knows nothing about this deployment. Groq's gpt-oss-120b
    # said "I'm running on OpenAI's GPT-4 model" three times in one session.
    # No regex fixes that, because the phrasings are unbounded -- the only
    # real fix is telling the brain the truth before it is asked.
    _WHOAMI = (
        "\n\n## Which brain you are, right now\n\n"
        "This turn is being answered by the brain named {name} — {what}. "
        "{routing}\n\n"
        "If you are asked which brain, model, AI or architecture you are, "
        "answer from the line above and nothing else. What you remember "
        "about your own identity describes the model, not this assistant, and "
        "saying it would be wrong. You are E.V.I.E. either way; {name} is "
        "only what is thinking for you at the moment."
    )

    def _routing_note(self, name: str) -> str:
        pinned = self.registry.pinned
        if pinned == name:
            return (
                "You were chosen by hand and stay until that is released, "
                "so do not offer to switch unless asked."
            )
        if pinned:
            # A pin that could not be honoured -- agentic work sent to a
            # brain with hands. Say so, or the next "which brain" answer
            # contradicts the last switch confirmation.
            return (
                f"{pinned} was picked by hand but cannot do this kind of work, "
                f"so this one turn came here instead."
            )
        return "No brain is pinned; this one was chosen for this request."

    # The vault was write-only until this existed. Canvas sync wrote sixteen
    # real deadlines into classes/upcoming.md, and "what's due this week"
    # still got "I'm not sure what's on your calendar yet" -- because nothing
    # ever read the file back. Routing was right, groq answered in under a
    # second; it just had nothing to answer from.
    _BRIEFING = "\n\n# What you know right now\n\n{block}\n\n{caveat}"
    _CAN_READ_MORE = (
        "That is a summary. The vault is your working directory, so open "
        "anything in it when you need more than this."
    )
    _THIS_IS_ALL = (
        "That is everything you have. If the answer is not above, say so "
        "rather than guessing — and say when it was last synced if that is "
        "why."
    )
    # Reading the whole list back took 17 seconds of unbroken speech, which
    # is a long time to stand there. The list is formatted, so it invites
    # being read out; it needs saying that speech is not a screen.
    _DONT_READ_THE_LIST = (
        "\n\nWhen more than about three of these would answer a question, "
        "do not read them all out. Say how many there are and the one or two "
        "that matter soonest, then stop and let them ask for the rest."
    )

    def _identity(self, brain: str | None = None) -> str:
        if self.vault and self.vault.exists and (found := self.vault.identity()):
            base = found
        else:
            base = load_identity(self.settings)
        if brain is None:
            return base

        agentic = self.registry.get(brain).agentic

        # Injected for every brain, not just toolless ones: a CLI brain could
        # read these files itself, but that is a round trip to learn something
        # that fits in a few hundred characters, and `claude` already costs
        # 5-11s a call.
        if self.vault and self.vault.exists and (block := self.vault.briefing()):
            base += self._BRIEFING.format(
                block=block,
                caveat=self._CAN_READ_MORE if agentic else self._THIS_IS_ALL,
            )
            base += self._DONT_READ_THE_LIST

        base += self._WHOAMI.format(
            name=brain,
            what=self.registry.describe(brain),
            routing=self._routing_note(brain),
        )
        return base if agentic else base + self._NO_HANDS

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

        # Rebuilt every turn, because both halves change per turn: which
        # brain is answering, and whether it has hands. The same conversation
        # may be answered by a CLI brain with file tools and then by an HTTP
        # brain with none.
        self.ctx.system = self._identity(used)

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
