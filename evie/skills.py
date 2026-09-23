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
from dataclasses import dataclass, field
from pathlib import Path

from .config import PACKAGE_DEFAULTS

# The path inside the vault. Matches where Claude Code looks in any project,
# which is the whole reason this needs no wiring.
SKILLS_DIR = Path(".claude") / "skills"
# Sub-agents are the same problem wearing different frontmatter: markdown that
# has to reach the vault and survive being edited by hand. Claude Code finds
# them at `.claude/agents/` in its working directory, which is the vault, so
# they need no more wiring than the skills did.
AGENTS_DIR = Path(".claude") / "agents"

SHIPPED = PACKAGE_DEFAULTS / "skills"
SHIPPED_AGENTS = PACKAGE_DEFAULTS / "agents"

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
        name=path.stem if path.name != "SKILL.md" else path.parent.name,
        path=path,
        triggers=triggers,
        managed=MANAGED in body,
    )


def _all_in(root: Path) -> list[Skill]:
    if not root.is_dir():
        return []
    found = (read(d / "SKILL.md") for d in sorted(root.iterdir()) if d.is_dir())
    return [s for s in found if s is not None]


def _all_files_in(root: Path) -> list[Skill]:
    """Agents, which are one file each rather than a folder."""
    if not root.is_dir():
        return []
    found = (read(f) for f in sorted(root.glob("*.md")))
    return [s for s in found if s is not None]


def shipped() -> list[Skill]:
    return _all_in(SHIPPED)


def installed(vault_root: Path) -> list[Skill]:
    return _all_in(Path(vault_root) / SKILLS_DIR)


def shipped_agents() -> list[Skill]:
    return _all_files_in(SHIPPED_AGENTS)


def installed_agents(vault_root: Path) -> list[Skill]:
    return _all_files_in(Path(vault_root) / AGENTS_DIR)


def triggers(vault_root: Path) -> dict[str, tuple[str, ...]]:
    """What to say to run each installed skill. For the router."""
    return {s.name: s.triggers for s in installed(vault_root) if s.triggers}


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    yours: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated)


def _install(source: Path, dest: Path, name: str, report: SyncReport) -> None:
    """Copy one shipped file into place, or explain why we did not.

    The whole overlay rule lives here, once, because skills and agents need
    exactly the same treatment and two copies of this logic would drift.
    """
    wanted = source.read_text()

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        report.added.append(name)
        return

    current = read(dest)
    if current and not current.managed:
        # The marker is gone, so someone has adopted this file. Overwriting it
        # would throw away their work with no warning.
        report.yours.append(name)
    elif dest.read_text() == wanted:
        report.unchanged.append(name)
    else:
        dest.write_text(wanted)
        report.updated.append(name)


def sync(vault_root: Path) -> SyncReport:
    """Bring the shipped skills into the vault without touching yours."""
    target = Path(vault_root) / SKILLS_DIR
    report = SyncReport()
    for skill in shipped():
        _install(skill.path, target / skill.name / "SKILL.md", skill.name, report)
    return report


def sync_agents(vault_root: Path) -> SyncReport:
    """Same overlay, for the sub-agents."""
    target = Path(vault_root) / AGENTS_DIR
    report = SyncReport()
    for agent in shipped_agents():
        _install(agent.path, target / f"{agent.name}.md", agent.name, report)
    return report
