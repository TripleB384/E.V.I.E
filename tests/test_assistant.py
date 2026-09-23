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


class TestToolAwareIdentity:
    """A brain with no file access should not narrate file writes.

    One real reply ended with "*Vault note:* User asked about the current
    model." Nothing was written — EVIE.md instructs her to keep things in the
    vault, and a brain without hands described doing so instead.
    """

    def _assistant(self, tmp_path):
        from evie.brains import BrainRegistry
        from tests.test_brains import FakeBrain

        vault = Vault(tmp_path / "v").ensure("test")
        (vault.root / "EVIE.md").write_text("You are E.V.I.E. Keep notes in the vault.")

        hands = FakeBrain("claude", agentic=True)
        no_hands = FakeBrain("groq", agentic=False)
        reg = BrainRegistry(
            {"claude": hands, "groq": no_hands}, default="groq",
            tiers={"simple": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault)

    async def test_a_toolless_brain_is_told_it_has_no_hands(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("what's the capital of Peru")
        sent = a.registry.get("groq").prompts[0]
        assert "no file access" in a.ctx.system
        assert "Never describe saving" in a.ctx.system
        assert sent, "the brain was still called"

    async def test_an_agentic_brain_gets_the_identity_unaltered(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("organize my notes")
        assert "no file access" not in a.ctx.system
        assert "Keep notes in the vault" in a.ctx.system

    async def test_it_switches_back_and_forth_within_one_conversation(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("organize my notes")            # agentic -> claude
        assert "no file access" not in a.ctx.system
        await a.ask("what's the capital of Peru")   # simple -> groq
        assert "no file access" in a.ctx.system


class TestSheKnowsWhichBrainSheIs:
    """A model asked what it is answers from its training data.

    In one session Groq's gpt-oss-120b said "I'm running on OpenAI's GPT-4
    model", then "I'm powered by the GPT-4 architecture", then named GPT-4 a
    third time — confidently, and wrong every time. No regex fixes that: the
    phrasings are unbounded, so some questions will always reach a model.
    What can be fixed is what the model knows when one does.
    """

    def _assistant(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("test")
        (vault.root / "EVIE.md").write_text("You are E.V.I.E.")
        reg = BrainRegistry(
            {"claude": FakeBrain("claude", agentic=True, model="claude-cli"),
             "groq": FakeBrain("groq", model="openai/gpt-oss-120b")},
            default="groq",
            tiers={"simple": "groq", "normal": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault)

    async def test_the_answering_brain_is_named_in_its_own_prompt(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("what's the capital of Peru")
        assert "groq" in a.ctx.system
        assert "openai/gpt-oss-120b" in a.ctx.system

    async def test_it_is_told_not_to_answer_from_training_data(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("what's the capital of Peru")
        assert "answer from the line above" in a.ctx.system

    async def test_the_name_follows_the_brain_that_actually_answered(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("organize my notes")          # agentic -> claude
        assert "claude-cli" in a.ctx.system
        assert "openai/gpt-oss-120b" not in a.ctx.system
        await a.ask("what's the capital of Peru")  # simple -> groq
        assert "openai/gpt-oss-120b" in a.ctx.system
        assert "claude-cli" not in a.ctx.system

    async def test_a_pinned_brain_is_told_it_was_picked_by_hand(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("switch to claude")
        await a.ask("hello")
        assert "chosen by hand" in a.ctx.system

    async def test_an_unpinned_brain_is_told_it_was_routed(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("hello")
        assert "No brain is pinned" in a.ctx.system

    async def test_it_is_told_it_cannot_switch_brains_itself(self, tmp_path):
        """The other half of the same problem. "No switch to Claude" missed the
        intercept, reached groq, and groq replied "I'm staying right here on
        the current model" -- a routing decision it had no part in, stated as
        settled. Widening the regex catches that phrasing; this is what
        catches the next one."""
        a = self._assistant(tmp_path)
        await a.ask("what's the capital of Peru")
        assert "cannot change which brain answers" in a.ctx.system
        assert "Never confirm a switch" in a.ctx.system

    async def test_a_pin_that_could_not_be_honoured_says_so(self, tmp_path):
        """Otherwise the answer to "which brain" contradicts the switch
        confirmation the user heard two turns ago."""
        a = self._assistant(tmp_path)
        await a.ask("switch to groq")
        await a.ask("organize my notes")   # groq has no hands; claude takes it
        assert "groq was picked by hand but cannot do this kind of work" in a.ctx.system

    async def test_a_brain_with_no_describe_still_gets_a_name(self, tmp_path):
        """`describe()` postdates the Brain protocol; an older one must not
        break the turn."""
        class Older:
            name = "old"
            agentic = False

            def missing(self):
                return None

            async def stream(self, prompt, ctx):
                yield "hi"

            async def health(self):
                raise NotImplementedError

        reg = BrainRegistry({"old": Older()}, default="old")
        a = Assistant(reg, Settings(vault=tmp_path / "v"), None)
        await a.ask("hello")
        assert "the brain named old — old." in a.ctx.system


class TestSheCanSeeTheVault:
    """Sixteen real Canvas deadlines sat in the vault and "what's due this
    week" still answered "I'm not sure what's on your calendar yet".

    Routing was right — groq answered in under a second, exactly as designed.
    Nothing had ever read the file back, so it had nothing to answer from.
    That made the whole vault-instead-of-MCP decision worthless in practice
    while looking correct in the logs.
    """

    def _assistant(self, tmp_path, **brains):
        vault = Vault(tmp_path / "v").ensure("Isaac")
        (vault.root / "EVIE.md").write_text("You are E.V.I.E.")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").write_text(
            "- [ ] **Thu 25 Sep, 11:59 pm** — Directed Writing draft (AICE ENG LANG)\n"
        )
        reg = BrainRegistry(
            brains or {"groq": FakeBrain("groq", agentic=False)},
            default=next(iter(brains or {"groq": None})),
        )
        return Assistant(reg, Settings(vault=vault.root), vault)

    async def test_a_toolless_brain_is_handed_the_deadlines(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("what's due this week")
        assert "Directed Writing draft" in a.ctx.system

    async def test_it_knows_what_day_it_is(self, tmp_path):
        """"This week" is unanswerable without a clock, and a model has none."""
        a = self._assistant(tmp_path)
        await a.ask("what's due this week")
        assert "Today is" in a.ctx.system

    async def test_an_agentic_brain_gets_it_too(self, tmp_path):
        """It could open the file itself, but that is a round trip to learn
        something that fits in a few hundred characters, and claude already
        costs 5-11s a call."""
        a = self._assistant(tmp_path, claude=FakeBrain("claude", agentic=True))
        await a.ask("what's due this week")
        assert "Directed Writing draft" in a.ctx.system

    async def test_the_two_are_told_different_things_about_it(self, tmp_path):
        toolless = self._assistant(tmp_path)
        await toolless.ask("hello")
        assert "everything you have" in toolless.ctx.system

        agentic = self._assistant(tmp_path, claude=FakeBrain("claude", agentic=True))
        await agentic.ask("hello")
        assert "working directory" in agentic.ctx.system

    async def test_no_vault_is_not_a_crash(self, tmp_path):
        reg = BrainRegistry({"groq": FakeBrain("groq")}, default="groq")
        a = Assistant(reg, Settings(vault=tmp_path / "nothing"), None)
        reply = await a.ask("hello")
        assert reply.text, "she should still answer"
        assert "What you know right now" not in a.ctx.system

    async def test_it_refreshes_between_turns(self, tmp_path):
        """A sync that happens mid-conversation has to be visible on the next
        turn, not after a restart."""
        a = self._assistant(tmp_path)
        await a.ask("hello")
        assert "Lab 4 writeup" not in a.ctx.system

        (a.vault.root / "classes" / "upcoming.md").write_text(
            "- [ ] **Fri 26 Sep, 9:00 am** — Lab 4 writeup (AI301)\n"
        )
        await a.ask("hello again")
        assert "Lab 4 writeup" in a.ctx.system

    async def test_it_is_told_not_to_add_notes_to_itself(self, tmp_path):
        """One reply ended "(Note: user asked about weekly due items.)" —
        bookkeeping read aloud is noise."""
        a = self._assistant(tmp_path)
        await a.ask("hello")
        assert "notes-to-self" in a.ctx.system


class TestSheDoesNotReadTheWholeList:
    """Asked what was due, she spoke seven deadlines in full: 17.4 seconds of
    unbroken speech. EVIE.md already says to lead with the answer and offer
    the detail, but the briefing hands her a formatted list, and a formatted
    list invites being read out."""

    def _assistant(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Isaac")
        (vault.root / "EVIE.md").write_text("You are E.V.I.E.")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").write_text("- [ ] **Thu 25 Sep** — Lab 4\n")
        reg = BrainRegistry({"groq": FakeBrain("groq")}, default="groq")
        return Assistant(reg, Settings(vault=vault.root), vault)

    async def test_she_is_told_to_summarise_a_long_list(self, tmp_path):
        a = self._assistant(tmp_path)
        await a.ask("what's due this week")
        assert "do not read them all out" in a.ctx.system
        assert "let them ask for the rest" in a.ctx.system

    async def test_the_advice_only_rides_along_with_a_vault(self, tmp_path):
        """With nothing to list, the instruction is noise in every prompt."""
        reg = BrainRegistry({"groq": FakeBrain("groq")}, default="groq")
        a = Assistant(reg, Settings(vault=tmp_path / "none"), None)
        await a.ask("hello")
        assert "do not read them all out" not in a.ctx.system


class TestSheCanRunTheSkillsInTheVault:
    """Wiring, not matching -- the matching is pinned in test_skills.py.

    Without this hand-off the router never learns a skill exists, every test
    over there still passes, and a real "do my weekly deadline sweep" goes to
    the brain with no hands.
    """

    def _assistant(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("test")
        reg = BrainRegistry(
            {"claude": FakeBrain("claude", agentic=True),
             "groq": FakeBrain("groq", agentic=False)},
            default="groq",
            tiers={"simple": "groq", "normal": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault), vault

    async def test_naming_a_skill_reaches_the_brain_with_hands(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)

        await a.ask("do my weekly deadline sweep")
        assert a.registry.get("claude").prompts, "claude was never called"
        assert not a.registry.get("groq").prompts

    async def test_an_ordinary_question_still_goes_to_the_cheap_brain(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)

        await a.ask("what's the capital of Peru")
        assert a.registry.get("groq").prompts
        assert not a.registry.get("claude").prompts

    async def test_a_skill_installed_mid_session_is_heard(self, tmp_path):
        """Read per turn, not cached: `evie skills sync` in another terminal
        should not need a restart to take effect."""
        from evie import skills

        a, vault = self._assistant(tmp_path)
        await a.ask("do my weekly deadline sweep")
        assert a.registry.get("groq").prompts, "no skills yet, so the cheap brain"

        skills.sync(vault.root)
        await a.ask("do my weekly deadline sweep")
        assert a.registry.get("claude").prompts

    async def test_no_vault_is_not_an_error(self, tmp_path):
        reg = BrainRegistry({"groq": FakeBrain("groq")}, default="groq")
        a = Assistant(reg, Settings(vault=tmp_path / "nope"), None)
        reply = await a.ask("do my weekly deadline sweep")
        assert reply.text


class TestAFallbackBrainGetsItsOwnIdentity:
    """`ctx.system` was built once, for the brain we *expected* to answer.

    Live: claude was logged out, groq picked up, and groq was handed claude's
    identity -- named claude, told the vault was its working directory, never
    told it had no hands. So it did not say it could not run the job; it
    produced 46 seconds of formatted markdown that looked like the job.

    That is the GPT-4 bug through a different door. `_WHOAMI` exists because a
    model asked what it is answers from training data, and a fallen-back turn
    put the *wrong* answer straight into the prompt instead.
    """

    def _pair(self, tmp_path, **kw):
        vault = Vault(tmp_path / "v").ensure("test")
        (vault.root / "EVIE.md").write_text("You are E.V.I.E.")
        claude = FakeBrain("claude", agentic=True, model="claude-cli",
                           fail=BrainExhausted("out of quota"))
        groq = FakeBrain("groq", agentic=False, model="openai/gpt-oss-120b")
        reg = BrainRegistry(
            {"claude": claude, "groq": groq},
            default="claude",
            fallback=["claude", "groq"],
            tiers={"simple": "claude", "normal": "claude", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault), vault

    async def test_it_is_named_as_itself(self, tmp_path):
        a, _ = self._pair(tmp_path)
        reply = await a.ask("what's the capital of Peru")
        assert reply.notices, "the chain did not actually fall back"
        assert "named groq" in a.ctx.system
        assert "named claude" not in a.ctx.system

    async def test_it_is_described_as_itself(self, tmp_path):
        a, _ = self._pair(tmp_path)
        await a.ask("what's the capital of Peru")
        assert "openai/gpt-oss-120b" in a.ctx.system
        assert "claude-cli" not in a.ctx.system

    async def test_it_is_told_it_has_no_hands(self, tmp_path):
        """The one that did the damage: it inherited claude's agentic framing
        and believed it could open files."""
        a, _ = self._pair(tmp_path)
        await a.ask("what's the capital of Peru")
        assert "no file access" in a.ctx.system
        assert "working directory" not in a.ctx.system

    async def test_the_brain_saw_it_at_the_time_not_just_afterwards(self, tmp_path):
        """Asserting on `ctx.system` after the turn would pass even if the
        prompt were rebuilt too late to reach anyone."""
        a, _ = self._pair(tmp_path)
        seen = {}
        groq = a.registry.get("groq")
        original = groq.stream

        async def spy(prompt, ctx):
            seen["system"] = ctx.system
            async for chunk in original(prompt, ctx):
                yield chunk

        groq.stream = spy
        await a.ask("what's the capital of Peru")
        assert "named groq" in seen["system"]
        assert "no file access" in seen["system"]

    async def test_a_turn_with_no_fallback_is_unchanged(self, tmp_path):
        a, _ = self._pair(tmp_path)
        a.registry.get("claude").fail = None
        await a.ask("what's the capital of Peru")
        assert "named claude" in a.ctx.system
        assert "no file access" not in a.ctx.system


class TestASkillThatCannotRunSaysSo:
    """groq was asked for the deadline sweep, could not read the log or write
    the plan, and returned a formatted list built from the briefing it already
    had. It looked like a sweep and was not one."""

    def _pair(self, tmp_path, claude_works=False):
        vault = Vault(tmp_path / "v").ensure("test")
        claude = FakeBrain("claude", agentic=True)
        if not claude_works:
            claude.fail = BrainExhausted("out of quota")
        groq = FakeBrain("groq", agentic=False)
        reg = BrainRegistry(
            {"claude": claude, "groq": groq},
            default="groq",
            fallback=["claude", "groq"],
            tiers={"simple": "groq", "normal": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault), vault

    async def test_a_toolless_brain_is_told_the_skill_did_not_run(self, tmp_path):
        from evie import skills

        a, vault = self._pair(tmp_path)
        skills.sync(vault.root)
        await a.ask("do my weekly deadline sweep")
        assert "weekly-deadline-sweep" in a.ctx.system
        assert "did not run" in a.ctx.system

    async def test_a_brain_with_hands_gets_no_such_caveat(self, tmp_path):
        from evie import skills

        a, vault = self._pair(tmp_path, claude_works=True)
        skills.sync(vault.root)
        await a.ask("do my weekly deadline sweep")
        assert a.registry.get("claude").prompts, "claude never ran it"
        assert "did not run" not in a.ctx.system

    async def test_an_ordinary_question_never_gets_it(self, tmp_path):
        from evie import skills

        a, vault = self._pair(tmp_path)
        skills.sync(vault.root)
        await a.ask("what's the capital of Peru")
        assert "did not run" not in a.ctx.system


class TestTheBrainThatOfferedGetsTheFollowUp:
    """`registry.active` is not the brain that just answered.

    It only moves on a fallback, so a turn the tiers sent to claude leaves it
    pointing at groq -- which is how "Yes." after claude's "want me to go
    through the rest?" landed on a brain that could not open the file claude
    had written.
    """

    def _assistant(self, tmp_path, claude_text="Here it is. Want the rest?"):
        vault = Vault(tmp_path / "v").ensure("test")
        reg = BrainRegistry(
            {"claude": FakeBrain("claude", agentic=True, text=claude_text),
             "groq": FakeBrain("groq", agentic=False, text="From groq.")},
            default="groq",
            tiers={"simple": "groq", "normal": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault), vault

    async def test_yes_goes_back_to_claude(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)

        await a.ask("do my weekly deadline sweep")
        assert a._last_brain == "claude"
        await a.ask("Yes.")
        assert a.registry.get("claude").calls == 2
        assert a.registry.get("groq").calls == 0

    async def test_agreement_with_no_question_asked_does_not(self, tmp_path):
        """The control. Routing every "yeah" to an agentic brain would cost
        half a minute and real plan allowance to answer nothing."""
        from evie import skills

        a, vault = self._assistant(tmp_path, claude_text="Here is the plan.")
        skills.sync(vault.root)

        await a.ask("do my weekly deadline sweep")
        await a.ask("Yes.")
        assert a.registry.get("groq").calls == 1

    async def test_an_explicit_continuation_needs_no_question(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path, claude_text="Here is the plan.")
        skills.sync(vault.root)

        await a.ask("do my weekly deadline sweep")
        await a.ask("tell me the rest")
        assert a.registry.get("claude").calls == 2

    async def test_the_first_turn_of_a_session_has_nobody_to_follow(self, tmp_path):
        a, _ = self._assistant(tmp_path)
        assert a._last_brain is None
        await a.ask("go on")
        assert a.registry.get("groq").calls == 1

    async def test_an_intercept_does_not_become_the_last_brain(self, tmp_path):
        """A swap confirmation is spoken by E.V.I.E. herself, so there is no
        brain behind it to follow up with."""
        a, _ = self._assistant(tmp_path)
        await a.ask("what's the capital of Peru")
        await a.ask("switch to claude")
        assert a._last_brain == "groq"


class TestSheSaysSomethingBeforeALongJob:
    """Thirty-eight seconds between the key going up and the first word. The
    work was real; from where you are standing it is a hang."""

    def _assistant(self, tmp_path, claude_fails=False):
        vault = Vault(tmp_path / "v").ensure("test")
        claude = FakeBrain("claude", agentic=True)
        if claude_fails:
            claude.fail = BrainExhausted("out of quota")
        reg = BrainRegistry(
            {"claude": claude, "groq": FakeBrain("groq", agentic=False)},
            default="groq",
            fallback=["claude", "groq"],
            tiers={"simple": "groq", "normal": "groq", "agentic": "claude"},
        )
        return Assistant(reg, Settings(vault=vault.root), vault), vault

    async def test_a_skill_run_is_announced(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)
        reply = await a.ask("do my weekly deadline sweep")
        assert any("weekly deadline sweep" in n for n in reply.notices)
        assert not any("-" in n.split("Running the ")[1][:24]
                       for n in reply.notices if "Running the" in n), \
            "the slug should be spoken as words"

    async def test_it_comes_before_the_answer(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)
        kinds = [kind async for kind, _ in a.respond("do my weekly deadline sweep")]
        assert kinds[0] == "notice", "nothing was said before the wait"

    async def test_an_ordinary_question_is_not_announced(self, tmp_path):
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)
        reply = await a.ask("what's the capital of Peru")
        assert not any("Running" in n for n in reply.notices)

    async def test_a_skill_that_then_fails_is_corrected_out_loud(self, tmp_path):
        """A login that has expired cannot be known about before the call, so
        she does announce it. What matters is that the correction follows in
        the same breath: each line was true when it was said."""
        from evie import skills

        a, vault = self._assistant(tmp_path, claude_fails=True)
        skills.sync(vault.root)
        reply = await a.ask("do my weekly deadline sweep")
        assert "Running the" in reply.notices[0]
        assert "Switching to groq" in reply.notices[1]
        assert "did not run" in a.ctx.system, "and the brain knows not to fake it"

    async def test_a_parked_brain_is_never_promised(self, tmp_path):
        """Parked means no key and no attempt. That is knowable without a
        call, so promising it would be a promise we know we cannot keep."""
        from evie import skills

        a, vault = self._assistant(tmp_path)
        skills.sync(vault.root)
        a.registry.parked["claude"] = "$ANTHROPIC_API_KEY is not set"
        reply = await a.ask("do my weekly deadline sweep")
        assert not any("Running the" in n for n in reply.notices)
