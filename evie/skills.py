"""Jobs she can run, written as markdown and kept in the vault.

A skill is a file describing a piece of recurring work -- sweep this week's
deadlines, tidy the notes for a course -- that an agentic brain reads and
carries out. `claude` already runs with the vault as its working directory, so
a file at `<vault>/.claude/skills/<name>/SKILL.md` needs no wiring at all: it
is found the same way it would be in any repo.

Two things here are not obvious.

**Install is an overlay, never a snapshot.** Copying a shipped file into the
vault once, at `evie init`, is a mistake this project has already made twice:
the copy wins forever, so every later improvement is invisible to anyone who
ran init before it. So a shipped skill carries a marker line, and `sync`
rewrites only files that still have it. Delete the line and the file is yours;
sync says it skipped it and moves on.

**A skill is invoked by name, and the router matches verbs.** "check my
deadlines" is already routed to a brain with hands by `_TASK_VERBS`, but "do
my weekly deadline sweep" is a five-word question that scores as *simple* and
goes to the cheap brain -- which has no file access and no idea the skill
exists, so it answers *about* the sweep instead of running it. `triggers()`
is what the router uses to recognise the name.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import PACKAGE_DEFAULTS

# The path inside the vault. Matches where Claude Code looks in any project,
# which is the whole reason this needs no wiring.
SKILLS_DIR = Path(".claude") / "skills"

SHIPPED = PACKAGE_DEFAULTS / "skills"

# Deliberately markers in the body rather than extra frontmatter keys: the
# frontmatter is read by Claude Code and should carry only what it expects.
# The vault also already uses HTML-comment markers for the Canvas block, so
# this is the same idea in the same place.
MANAGED = "<!-- evie:managed -->"
_TRIGGERS = re.compile(r"<!--\s*evie:triggers\s+(.+?)\s*-->", re.IGNORECASE)

# A one-word trigger steals every sentence containing that word: a skill named
# "notes" would route "what did you say about my notes" to a 5-11s Claude Code
# session. Two words is the cheapest rule that makes a collision unlikely.
MIN_TRIGGER_WORDS = 2


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    triggers: tuple[str, ...]
    managed: bool

    @property
    def description(self) -> str:
        """The `description:` line, which is what a brain matches on."""
        for line in self.path.read_text().splitlines():
            if line.startswith("description:"):
                return line.split(":", 1)[1].strip()
        return ""


def read(path: Path) -> Skill | None:
    """One SKILL.md, or None if it is not one."""
    if not path.is_file():
        return None
    body = path.read_text()
    triggers = tuple(
        phrase
        for raw in _TRIGGERS.findall(body)
        for phrase in (p.strip() for p in raw.split("|"))
        if len(phrase.split()) >= MIN_TRIGGER_WORDS
    )
    return Skill(
        name=path.parent.name,
        path=path,
        triggers=triggers,
        managed=MANAGED in body,
    )


def _all_in(root: Path) -> list[Skill]:
    if not root.is_dir():
        return []
    found = (read(d / "SKILL.md") for d in sorted(root.iterdir()) if d.is_dir())
    return [s for s in found if s is not None]


def shipped() -> list[Skill]:
    return _all_in(SHIPPED)


def installed(vault_root: Path) -> list[Skill]:
    return _all_in(Path(vault_root) / SKILLS_DIR)


def triggers(vault_root: Path) -> dict[str, tuple[str, ...]]:
    """What to say to run each installed skill. For the router."""
    return {s.name: s.triggers for s in installed(vault_root) if s.triggers}


@dataclass
class SyncReport:
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    yours: list[str]

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated)


def sync(vault_root: Path) -> SyncReport:
    """Bring the shipped skills into the vault without touching yours.

    A file that has lost its managed marker is one someone has adopted and
    edited, so it is left exactly as it is and reported -- overwriting it
    would throw away work with no warning.
    """
    target = Path(vault_root) / SKILLS_DIR
    report = SyncReport([], [], [], [])

    for skill in shipped():
        dest = target / skill.name / "SKILL.md"
        wanted = skill.path.read_text()

        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(skill.path, dest)
            report.added.append(skill.name)
            continue

        current = read(dest)
        if current and not current.managed:
            report.yours.append(skill.name)
        elif dest.read_text() == wanted:
            report.unchanged.append(skill.name)
        else:
            dest.write_text(wanted)
            report.updated.append(skill.name)

    return report
