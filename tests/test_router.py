"""Voice commands must never reach a model.

That's the point of the router: saying "switch to Gemini" should cost nothing
and happen instantly. If one of these leaks through to a brain, the user pays
tokens and waits a second to do something local.
"""

import pytest

from evie.brains import BrainRegistry, EchoBrain
from evie.router import (
    Action,
    complexity,
    is_quick,
    route,
    strip_address,
    wants_agentic,
)


class Agentic(EchoBrain):
    agentic = True


@pytest.fixture
def reg_full():
    """The shipped roster, so name resolution is tested against real aliases."""
    from evie.config import PACKAGE_DEFAULTS, load_registry

    return load_registry(PACKAGE_DEFAULTS / "brains.yaml")


@pytest.fixture
def reg():
    return BrainRegistry(
        {"claude": Agentic("claude"), "gemini_cli": EchoBrain("gemini_cli"),
         "groq": EchoBrain("groq"), "ollama": EchoBrain("ollama")},
        default="claude",
        quick="groq",
        aliases={"gemini": "gemini_cli", "google": "gemini_cli",
                 "local": "ollama", "fast": "groq"},
    )


class TestAddress:
    @pytest.mark.parametrize(
        "said", ["Evie, what time is it", "hey Evie what time is it",
                 "evie: what time is it", "Eevee, what time is it"]
    )
    def test_strips_her_name(self, said):
        assert strip_address(said) == "what time is it"

    def test_leaves_other_text_alone(self):
        assert strip_address("what time is it") == "what time is it"


class TestSwapCommands:
    @pytest.mark.parametrize(
        "said,expected",
        [
            ("switch to Gemini", "gemini_cli"),
            ("Evie, switch to gemini", "gemini_cli"),
            ("use google", "gemini_cli"),
            ("switch over to the local model", "ollama"),
            ("go to groq", "groq"),
            ("please switch to gemini please", "gemini_cli"),
        ],
    )
    def test_swaps_without_calling_a_model(self, reg, said, expected):
        decision = route(said, reg)
        assert decision.action is Action.REPLY
        assert reg.active == expected
        assert "Switched" in decision.text

    def test_unknown_target_falls_through_to_a_model(self, reg):
        # "use the smallest font" is a request, not a brain swap.
        decision = route("use the smallest font", reg)
        assert decision.action is Action.ANSWER
        assert reg.active == "claude"

    def test_reports_active_brain(self, reg):
        assert "claude" in route("who are you running on", reg).text
        assert route("which model are you using", reg).action is Action.REPLY

    def test_lists_brains(self, reg):
        text = route("list your brains", reg).text
        assert "groq" in text and "claude" in text

    def test_go_back_resets(self, reg):
        reg.use("groq")
        route("go back", reg)
        assert reg.active == "claude"


class TestControl:
    @pytest.mark.parametrize("said", ["stop", "never mind", "shut up", "quiet"])
    def test_stop(self, reg, said):
        assert route(said, reg).action is Action.STOP

    @pytest.mark.parametrize("said", ["quit", "goodbye", "good night", "exit"])
    def test_quit(self, reg, said):
        assert route(said, reg).action is Action.QUIT

    def test_empty_input(self, reg):
        assert route("Evie,", reg).action is Action.REPLY


class TestIntentRouting:
    @pytest.mark.parametrize(
        "said",
        ["what's the capital of Peru", "how long is a marathon", "who wrote Dune",
         "explain recursion"],
    )
    def test_short_questions_go_to_the_quick_brain(self, reg, said):
        assert route(said, reg).brain == "groq"

    @pytest.mark.parametrize(
        "said",
        ["organize my CS notes", "check my deadlines for this week",
         "fix the bug in parser.py", "summarize my lecture notes",
         "save that to a file"],
    )
    def test_real_work_needs_a_brain_with_hands(self, reg, said):
        assert wants_agentic(said)
        decision = route(said, reg)
        # Either the active brain is already agentic, or one was chosen.
        chosen = decision.brain or reg.active
        assert reg.get(chosen).agentic

    def test_task_phrasing_beats_question_phrasing(self, reg):
        # Starts like a question but is plainly a task.
        assert not is_quick("what should I do, organize my notes for me")

    def test_switches_to_an_agentic_brain_when_active_has_no_hands(self, reg):
        reg.use("groq")
        decision = route("read my syllabus", reg)
        assert decision.brain == "claude"
        assert decision.tier == "agentic"

    def test_a_pinned_brain_survives_a_short_question(self, reg):
        """Picking a brain by hand must stick. Otherwise the very next
        one-liner routes back to the simple tier and undoes the choice."""
        reg.use("claude")
        assert reg.pinned == "claude"
        assert route("what's the capital of Peru", reg).brain == "claude"

    def test_a_pin_yields_only_to_work_it_cannot_do(self, reg):
        """Honouring a toolless pin on a file task would not respect the
        choice, it would fail at the file access instead."""
        reg.use("groq")
        assert route("summarize my lecture notes", reg).brain == "claude"
        assert route("tell me about Rome", reg).brain == "groq", "still pinned"

    def test_auto_releases_the_pin(self, reg):
        reg.use("claude")
        assert "Choosing" in route("auto", reg).text
        assert reg.pinned is None
        assert route("what's the capital of Peru", reg).brain == "groq"

    def test_a_rambling_question_is_not_the_simple_tier(self, reg):
        assert route("what " + "really " * 25 + "happened", reg).tier != "simple"


class TestComplexity:
    """Classifying difficulty before calling a model, because asking a model
    how hard something is costs as much as answering it."""

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "said",
        ["hey", "what time is it", "who wrote Dune", "how far is the moon",
         "what's 12 times 12"],
    )
    def test_lookups_are_simple(self, said):
        assert complexity(said) == "simple"

    @_pytest.mark.parametrize(
        "said",
        ["compare Rust and Go for a web backend",
         "analyze why our churn went up last quarter",
         "help me decide between two pricing models",
         "walk me through how a compiler does register allocation",
         "what's the best way to structure a seed round",
         "why does the parser fail on nested blocks",
         "write me an essay on the Treaty of Versailles",
         "write a draft of the investor email",
         "draft a cold email to the accelerator",
         "debug this recursion, it overflows on deep input"],
    )
    def test_real_thinking_is_hard(self, said):
        assert complexity(said) == "hard"

    @_pytest.mark.parametrize(
        "said",
        ["organize my CS notes", "check my deadlines", "read my syllabus",
         "summarize my lecture notes", "fix the bug in parser.py"],
    )
    def test_touching_things_is_agentic(self, said):
        # Checked before "hard": "summarize my lecture notes" reads like
        # thinking work but is really a file task.
        assert complexity(said) == "agentic"

    def test_length_alone_lifts_a_request(self):
        rambling = " ".join(["so I was thinking about the thing we discussed"] * 4)
        assert complexity(rambling) == "hard"

    def test_everything_else_is_normal(self):
        # A statement, not a lookup and not a request for analysis.
        assert complexity("I'm thinking about switching my major") == "normal"


class TestTierRouting:
    def test_each_tier_reaches_its_brain(self, reg):
        reg.tiers = {"simple": "groq", "normal": "gemini_cli",
                     "hard": "claude", "agentic": "claude"}
        assert reg.for_tier("simple") == "groq"
        assert reg.for_tier("hard") == "claude"

    def test_a_parked_tier_degrades_instead_of_failing(self, reg):
        """An unconfigured tier must not break routing -- you should not have
        to edit the tier table every time an API key appears or expires."""
        reg.tiers = {"simple": "groq", "hard": "claude"}
        reg.parked["claude"] = "$ANTHROPIC not set"
        assert reg.for_tier("hard") in reg.usable()

    def test_agentic_never_lands_on_a_brain_without_hands(self, reg):
        reg.tiers = {"agentic": "groq"}  # misconfigured: groq has no tools
        assert reg.get(reg.for_tier("agentic")).agentic


class TestHowWhisperActuallyHears:
    """Transcription returns phonemes, not spelling.

    Every string here came out of real speech-to-text for the word "Evie".
    Two of them reached a model, which then explained it could not change
    models -- a baffling answer to a question the router should have handled
    itself, for free.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "said,expected",
        [
            ("EV switch to Gemini", "gemini_cli"),
            ("Eevee switched to Gemini.", "gemini_cli"),
            ("E.V. switch to claude", "claude"),
            ("Eve, use groq", "groq"),
            ("Evie switching to claude", "claude"),
            ("evie change to groq", "groq"),
            ("Ivy, let's use groq", "groq"),
            ("hey evie swap to claude", "claude"),
            ("Evie, go to claude", "claude"),
        ],
    )
    def test_mishearings_and_inflections_all_switch(self, reg, said, expected):
        decision = route(said, reg)
        assert decision.action is Action.REPLY, f"{said!r} reached a model"
        assert reg.active == expected

    @_pytest.mark.parametrize(
        "said",
        ["even though I tried that", "the eve of the election was tense",
         "everything is fine", "evening plans are set", "ever since Tuesday"],
    )
    def test_ordinary_words_are_not_her_name(self, said):
        assert strip_address(said) == said

    def test_use_the_smallest_font_is_not_a_brain_swap(self, reg):
        before = reg.active
        assert route("use the smallest font", reg).action is Action.ANSWER
        assert reg.active == before


class TestTranscriptionSplitsWords:
    """Speech-to-text does not preserve word boundaries.

    "OpenRouter" came back as "open route". Substring matching could not
    bridge it -- "router" is not inside "open route" -- so the command went to
    a model, which replied "Switched to OpenRouter" while nothing switched.
    The user only found out two turns later.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "heard,expected",
        [
            ("open route", "openrouter"),
            ("open router", "openrouter"),
            ("OpenRouter", "openrouter"),
            ("gemini a p i", "gemini_api"),
            ("git hub", "github"),
        ],
    )
    def test_split_and_joined_names_resolve(self, reg_full, heard, expected):
        assert reg_full.resolve(heard) == expected

    def test_a_split_name_switches_for_real(self, reg_full):
        decision = route("EV switch to open route.", reg_full)
        assert decision.action is Action.REPLY, "must not reach a model"
        assert reg_full.active == "openrouter"
        assert reg_full.pinned == "openrouter"

    def test_a_near_miss_is_refused_rather_than_guessed(self, reg_full):
        """"clod" scores 0.89 against "cloud" and 0.60 against "claude", so
        edit-distance matching would route a mishearing of Claude to Ollama
        Cloud. Falling through to a model is merely unhelpful; switching to
        the wrong brain is wrong."""
        from evie.brains.registry import UnknownBrain

        with self._pytest.raises(UnknownBrain):
            reg_full.resolve("clod")


class TestWhoIsAnswering:
    import pytest as _pytest

    @_pytest.mark.parametrize(
        "said",
        ["which brain did you use to answer that question?",
         "what brain did you use to answer the weather question?",
         "which model are you using",
         "what brain are you on",
         "who are you running on",
         "are you still on claude"],
    )
    def test_every_phrasing_is_intercepted(self, reg, said):
        # One of these reached a model, which invented an account of its own
        # routing. The router knows the answer for free.
        assert route(said, reg).action is Action.REPLY


class TestSheClaimsSwitchesThatNeverHappened:
    """Every line here was said out loud in one voice session.

    Three of them reached a model, and the model answered as if it were the
    router: "Got it—switching over to the Grok brain now" and "Switched to
    auto routing", with nothing switched either time. A confirmation that
    isn't true is worse than no answer, because you stop checking.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "said",
        [
            "Eevee, can you tell me which brain / AI model you're using right now",
            "my bad I meant which brain you're using",
            "and then tell me which brain you're using",
            "so what AI are you",
            "what brain are you on",
        ],
    )
    def test_the_question_need_not_start_the_utterance(self, reg, said):
        assert route(said, reg).action is Action.REPLY, f"{said!r} reached a model"

    def test_grok_is_groq(self, reg_full):
        """The most common mishearing in the system, and prefix matching
        cannot reach it: a substituted final letter is not a prefix."""
        assert reg_full.resolve("grok") == "groq"
        assert reg_full.resolve("Grok") == "groq"

    def test_switching_to_grok_switches_for_real(self, reg_full):
        reg_full.use("claude")
        decision = route("Eevee switched to Grok.", reg_full)
        assert decision.action is Action.REPLY, "must not reach a model"
        assert reg_full.active == "groq"

    @_pytest.mark.parametrize(
        "said",
        ["if you go back to auto routing.", "reset to auto",
         "so just go back to automatic", "okay you choose"],
    )
    def test_auto_tolerates_a_lead_in(self, reg, said):
        reg.use("claude")
        assert "Choosing" in route(said, reg).text
        assert reg.pinned is None

    @_pytest.mark.parametrize("said", ["switch back to claude", "go back to claude"])
    def test_going_back_to_a_named_brain_is_a_switch_not_a_reset(self, reg, said):
        """`_RESET` used to swallow the brain name: "switch back to Claude"
        matched "switch back" and reset to the default instead."""
        reg.use("groq")
        route(said, reg)
        assert reg.active == "claude"

    @_pytest.mark.parametrize(
        "said",
        ["what is the best model for our pricing",
         "explain why you should switch back to Claude",
         "what is the resale value of that model"],
    )
    def test_ordinary_questions_are_not_stolen_from_a_model(self, reg, said):
        # Widening the intercepts must not start eating real questions:
        # answering "I'm running on claude" to any of these is nonsense.
        assert route(said, reg).action is Action.ANSWER
