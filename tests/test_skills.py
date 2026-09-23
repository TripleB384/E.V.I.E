"""Skills: markdown jobs an agentic brain runs from the vault.

Two things are worth pinning here, and neither is the markdown.

The first is that install is an *overlay*. Copying a shipped file into the
vault once, at `evie init`, is a mistake this project has already made twice
-- with `brains.yaml` and with `config.yaml` -- and both times it silently
froze someone's setup on the day they ran init. A skill that has been adopted
must survive a sync; one that has not must pick up improvements.

The second is routing. A skill is invoked by *name*, and every other router
pattern matches verbs, so "do my weekly deadline sweep" scored as a short
simple question and went to the cheap brain -- which has no file access and
does not know the skill exists.
"""

from pathlib import Path

import pytest

from evie import skills
from evie.brains import BrainRegistry, EchoBrain
from evie.memory import Vault
from evie.router import Action, names_a_skill, route


class Agentic(EchoBrain):
    agentic = True


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "v").ensure("test")


@pytest.fixture
def reg():
    return BrainRegistry(
        {"claude": Agentic("claude"), "groq": EchoBrain("groq")},
        default="groq",
        tiers={"simple": "groq", "normal": "groq", "hard": "groq",
               "agentic": "claude"},
    )


class TestWhatShips:
    def test_there_are_some(self):
        assert skills.shipped(), "no skills in evie/defaults/skills"

    def test_every_one_parses_and_names_itself(self):
        for skill in skills.shipped():
            assert skill.name == skill.path.parent.name
            assert skill.description, f"{skill.name} has no description:"

    def test_every_one_can_be_asked_for_by_voice(self):
        """A skill with no trigger can only be reached by a phrasing that
        happens to hit `_TASK_VERBS`, which is the gap this closes."""
        for skill in skills.shipped():
            assert skill.triggers, f"{skill.name} has no evie:triggers line"

    def test_no_trigger_is_a_single_word(self):
        """A one-word trigger steals every sentence containing that word: a
        skill named "notes" would send "what did you say about my notes" to a
        5-11s Claude Code session."""
        for skill in skills.shipped():
            for phrase in skill.triggers:
                assert len(phrase.split()) >= skills.MIN_TRIGGER_WORDS, phrase

    def test_they_are_all_managed_to_begin_with(self):
        assert all(s.managed for s in skills.shipped())

    def test_the_frontmatter_carries_only_what_claude_code_expects(self):
        """`name` and `description` are the documented keys. The evie markers
        live in the body deliberately, so an unknown frontmatter key can never
        make a skill fail to load."""
        for skill in skills.shipped():
            lines = skill.path.read_text().splitlines()
            assert lines[0] == "---"
            end = lines.index("---", 1)
            keys = {ln.split(":", 1)[0] for ln in lines[1:end] if ":" in ln}
            assert keys == {"name", "description"}, f"{skill.name}: {keys}"


class TestSyncIsAnOverlayNotASnapshot:
    def test_it_installs_them(self, vault):
        report = skills.sync(vault.root)
        assert report.added == [s.name for s in skills.shipped()]
        assert len(skills.installed(vault.root)) == len(skills.shipped())

    def test_running_it_twice_changes_nothing(self, vault):
        skills.sync(vault.root)
        again = skills.sync(vault.root)
        assert again.changed == 0
        assert len(again.unchanged) == len(skills.shipped())

    def test_an_improved_skill_reaches_someone_who_already_synced(self, vault):
        """The whole point. A one-time copy freezes on the day you ran init,
        and every later fix is invisible -- which has happened twice here."""
        skills.sync(vault.root)
        one = skills.installed(vault.root)[0]
        one.path.write_text(one.path.read_text() + "\nstale\n")

        report = skills.sync(vault.root)
        assert one.name in report.updated
        assert "stale" not in one.path.read_text()

    def test_a_skill_you_adopted_is_never_overwritten(self, vault):
        """Removing the managed marker is how you say "this is mine now".
        Overwriting it would throw away work with no warning."""
        skills.sync(vault.root)
        mine = skills.installed(vault.root)[0]
        mine.path.write_text(
            mine.path.read_text().replace(skills.MANAGED, "") + "\nmy own notes\n"
        )

        report = skills.sync(vault.root)
        assert mine.name in report.yours
        assert mine.name not in report.updated
        assert "my own notes" in mine.path.read_text()

    def test_a_skill_of_your_own_is_left_entirely_alone(self, vault):
        mine = vault.skills_dir() / "my-thing" / "SKILL.md"
        mine.parent.mkdir(parents=True)
        mine.write_text("---\nname: my-thing\ndescription: mine\n---\n")

        skills.sync(vault.root)
        assert mine.read_text().endswith("---\n")
        assert "my-thing" in [s.name for s in skills.installed(vault.root)]

    def test_it_creates_the_directory_if_init_predates_skills(self, tmp_path):
        """An existing vault has no `.claude/skills`, and sync is how it gets
        one -- otherwise this only works for people who init again."""
        root = tmp_path / "old"
        (root / "classes").mkdir(parents=True)
        assert skills.sync(root).added
        assert (root / skills.SKILLS_DIR).is_dir()


class TestTheVaultHandsThemToTheRouter:
    def test_triggers_are_empty_before_a_sync(self, vault):
        assert vault.skill_triggers() == {}

    def test_triggers_appear_after_one(self, vault):
        skills.sync(vault.root)
        found = vault.skill_triggers()
        assert set(found) == {s.name for s in skills.shipped()}
        assert all(found.values())

    def test_a_missing_vault_is_not_an_error(self, tmp_path):
        assert Vault(tmp_path / "nope").skill_triggers() == {}


class TestInvokingOneByName:
    """Six of eight natural phrasings reached the cheap brain before this.

    `_TASK_VERBS` catches "check my deadlines" and "run the sweep" because
    those carry a verb it knows. "do my weekly deadline sweep" carries the
    skill's *name*, scores as a short simple question, and went to groq --
    which has no hands and would answer about the sweep instead of running it.
    """

    SAID = [
        "do my weekly deadline sweep",
        "run the weekly deadline sweep",
        "sweep my deadlines for the week",
        "clean up my notes",
        "tidy my notes for AICE English",
        "start the competitor tracker",
        "what should I work on",
    ]

    @pytest.mark.parametrize("said", SAID)
    def test_it_reaches_a_brain_with_hands(self, vault, reg, said):
        skills.sync(vault.root)
        decision = route(said, reg, vault.skill_triggers())
        assert decision.action is Action.ANSWER
        assert reg.get(decision.brain).agentic, f"{said!r} went to {decision.brain}"
        assert decision.tier == "agentic"

    def test_the_matched_skill_is_reported(self, vault, reg):
        skills.sync(vault.root)
        decision = route("do my weekly deadline sweep", reg, vault.skill_triggers())
        assert decision.skill == "weekly-deadline-sweep"

    def test_a_run_together_hearing_still_matches(self):
        """Speech-to-text does not preserve word boundaries -- "open route"
        for "OpenRouter" is the same failure, one layer up."""
        found = {"weekly-deadline-sweep": ("deadline sweep",)}
        assert names_a_skill("do my deadlinesweep", found)
        assert names_a_skill("do my Deadline Sweep.", found)

    def test_an_ordinary_question_is_not_a_skill(self, vault, reg):
        skills.sync(vault.root)
        found = vault.skill_triggers()
        for said in ("what's the capital of Peru", "hello", "what time is it"):
            assert route(said, reg, found).skill is None

    def test_a_switch_command_still_wins(self, vault, reg):
        """Order matters: a skill check ahead of the swap intercepts would
        make "switch to claude" cost a model call."""
        skills.sync(vault.root)
        decision = route("switch to claude", reg, vault.skill_triggers())
        assert decision.action is Action.REPLY

    def test_no_skills_means_the_old_behaviour(self, reg):
        assert route("do my weekly deadline sweep", reg, {}).tier == "simple"
        assert route("do my weekly deadline sweep", reg).tier == "simple"
