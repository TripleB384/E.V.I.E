"""One turn, end to end, with no audio and no network."""

import pytest

from evie.assistant import Assistant
from evie.brains import BrainExhausted, BrainRegistry, EchoBrain
from evie.config import Settings
from evie.memory import Vault
from evie.router import Action
from tests.test_brains import FakeBrain


@pytest.fixture
def assistant(tmp_path):
    settings = Settings(vault=tmp_path / "vault")
    reg = BrainRegistry(
        {"claude": FakeBrain("claude", text="Thursday at midnight."),
         "groq": FakeBrain("groq", text="From groq.")},
        default="claude",
        fallback=["claude", "groq"],
        aliases={"gemini": "groq"},
    )
    return Assistant(reg, settings, Vault(settings.vault).ensure("test"))


class TestTurns:
    async def test_answers_a_question(self, assistant):
        reply = await assistant.ask("when is the project due")
        assert reply.text == "Thursday at midnight."
        assert reply.action is Action.ANSWER

    async def test_remembers_the_turn(self, assistant):
        await assistant.ask("when is the project due")
        roles = [t.role for t in assistant.ctx.transcript]
        assert roles == ["user", "assistant"]

    async def test_writes_to_the_vault(self, assistant):
        await assistant.ask("when is the project due")
        log = assistant.vault.today().read_text()
        assert "when is the project due" in log
        assert "Thursday at midnight." in log

    async def test_trims_the_transcript(self, assistant):
        assistant.settings.transcript_turns = 2
        for i in range(5):
            await assistant.ask(f"question {i}")
        assert len(assistant.ctx.transcript) == 4

    async def test_quit_is_a_control_signal(self, assistant):
        reply = await assistant.ask("goodbye")
        assert reply.action is Action.QUIT


class TestSwapMidConversation:
    async def test_swap_costs_no_model_call(self, assistant):
        claude = assistant.registry.get("claude")
        before = claude.calls
        reply = await assistant.ask("switch to gemini")
        assert "Switched" in reply.text
        assert claude.calls == before

    async def test_context_survives_the_swap(self, assistant):
        await assistant.ask("the project is a compiler")
        await assistant.ask("switch to gemini")
        await assistant.ask("what is it again")

        groq = assistant.registry.get("groq")
        # The incoming brain has never seen the conversation, so it must have
        # been handed a replay of it.
        assert groq.calls == 1
        assert "compiler" in assistant.ctx.handoff_summary()

    async def test_fallback_is_announced(self, tmp_path):
        settings = Settings(vault=tmp_path / "v")
        reg = BrainRegistry(
            {"claude": FakeBrain("claude", fail=BrainExhausted("429")),
             "groq": FakeBrain("groq", text="Still here.")},
            default="claude",
            fallback=["claude", "groq"],
        )
        a = Assistant(reg, settings, None)
        reply = await a.ask("hello")
        assert reply.text == "Still here."
        assert reply.notices and "rate limited" in reply.notices[0]


class TestIdentity:
    async def test_identity_reaches_the_brain(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Sam")
        (vault.root / "EVIE.md").write_text("Always answer in exactly three words.")
        reg = BrainRegistry({"echo": EchoBrain()}, default="echo")
        a = Assistant(reg, Settings(vault=vault.root), vault)
        assert "three words" in a.ctx.system
