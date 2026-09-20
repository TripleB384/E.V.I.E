"""Voice commands must never reach a model.

That's the point of the router: saying "switch to Gemini" should cost nothing
and happen instantly. If one of these leaks through to a brain, the user pays
tokens and waits a second to do something local.
"""

import pytest

from evie.brains import BrainRegistry, EchoBrain
from evie.router import Action, is_quick, route, strip_address, wants_agentic


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
         "write a draft of the investor email", "fix the bug in parser.py",
         "summarize my lecture notes"],
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

    def test_long_questions_stay_on_the_main_brain(self, reg):
        long_q = "what " + "really " * 25 + "happened"
        assert route(long_q, reg).brain is None
