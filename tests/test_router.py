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
