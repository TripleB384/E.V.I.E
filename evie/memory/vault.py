"""Memory as a folder of markdown files.

Deliberately not a vector database. Three reasons:

  * You can read it. When E.V.I.E. gets something wrong about you, you open the
    file and fix the line, instead of re-embedding a corpus.
  * Agentic brains already have grep and file tools, so pointing them at the
    vault gives you retrieval for free, with no RAG layer to go stale.
  * It is an Obsidian vault. Your notes and her memory are the same files.

Layout:
    EVIE.md              who she is, what she knows about you, standing rules
    daily/YYYY-MM-DD.md  rolling log, one file a day
    classes/<course>/    syllabus, notes, deadlines
    business/            the startup
    people/  projects/
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

SUBDIRS = ("daily", "classes", "business", "people", "projects")

IDENTITY_TEMPLATE = """# E.V.I.E.

You are E.V.I.E., {owner}'s assistant. You help with college work, the
business, and whatever else comes up.

## How to talk
You are being spoken to out loud and answering out loud. Keep replies to a few
sentences unless asked for more. No bullet lists, no markdown headings, no code
blocks in speech -- say it the way a person would. If something needs detail,
give the short answer first and offer the rest.

Be direct. If something is a bad idea, say so once and then help anyway.

## About {owner}
<!-- Fill this in, or just tell her and let her write it here. -->

## Standing instructions
- Write anything worth remembering into the vault before the conversation ends.
- The daily log is for what happened; the topic folders are for what is true.
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


class Vault:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()

    # -- setup -----------------------------------------------------------

    def ensure(self, owner: str = "you") -> "Vault":
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (self.root / sub).mkdir(exist_ok=True)
        identity = self.root / "EVIE.md"
        if not identity.exists():
            identity.write_text(IDENTITY_TEMPLATE.format(owner=owner))
        return self

    @property
    def exists(self) -> bool:
        return self.root.is_dir()

    # -- identity --------------------------------------------------------

    def identity(self) -> str:
        path = self.root / "EVIE.md"
        return path.read_text().strip() if path.is_file() else ""

    # -- daily log -------------------------------------------------------

    def today(self) -> Path:
        path = self.root / "daily" / f"{_dt.date.today():%Y-%m-%d}.md"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {_dt.date.today():%A, %B %-d %Y}\n\n")
        return path

    def log(self, speaker: str, text: str) -> None:
        """Append one line to today's log. The transcript of record."""
        if not text.strip():
            return
        stamp = _dt.datetime.now().strftime("%H:%M")
        with self.today().open("a") as fh:
            fh.write(f"- **{stamp} {speaker}:** {text.strip()}\n")

    def log_turn(self, said: str, replied: str, brain: str) -> None:
        self.log("you", said)
        self.log(brain, replied)

    # -- notes -----------------------------------------------------------

    def note(self, folder: str, title: str) -> Path:
        path = self.root / folder / f"{_slug(title)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"# {title}\n\n")
        return path

    def recent_log(self, days: int = 3) -> str:
        """The last few days of log, for injecting into a brain with no file access."""
        out: list[str] = []
        for offset in range(days):
            day = _dt.date.today() - _dt.timedelta(days=offset)
            path = self.root / "daily" / f"{day:%Y-%m-%d}.md"
            if path.is_file():
                out.append(path.read_text().strip())
        return "\n\n".join(reversed(out))
