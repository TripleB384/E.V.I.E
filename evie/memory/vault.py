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


def _trim(text: str, cap: int) -> str:
    """Keep the head, drop from the end, and say so.

    Truncating from the front would lose today's date and the soonest
    deadlines -- the two things most likely to be asked about.
    """
    if len(text) <= cap:
        return text
    kept = text[:cap].rsplit("\n", 1)[0].rstrip()
    return kept + "\n\n(...trimmed. Read the vault directly for the rest.)"


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
        # Nested, so not in SUBDIRS. This is where Claude Code looks for
        # skills in any project, and `claude` runs with the vault as its
        # working directory -- so a file here needs no other wiring.
        self.skills_dir().mkdir(parents=True, exist_ok=True)
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

    # -- what she knows without opening anything -------------------------

    # Rides on every single request, so an uncapped vault would quietly
    # inflate the cost of saying hello. A year of deadlines is not context,
    # it is a bill.
    BRIEFING_CAP = 2400

    def deadlines_path(self) -> Path:
        return self.root / "classes" / "upcoming.md"

    def deadlines_age(self) -> _dt.timedelta | None:
        """How long since Canvas was last synced, or None if it never was.

        From the file's mtime rather than the "Synced from Canvas ..." line it
        contains: that line is written for a person to read, and parsing prose
        back out of a file we wrote as prose is a way to be wrong twice.
        """
        try:
            written = self.deadlines_path().stat().st_mtime
        except OSError:
            return None
        return _dt.datetime.now() - _dt.datetime.fromtimestamp(written)

    @staticmethod
    def _how_long(age: _dt.timedelta) -> str:
        hours = age.total_seconds() / 3600
        if hours < 1:
            return "in the last hour"
        if hours < 24:
            return f"{int(hours)} hours ago"
        days = int(hours // 24)
        return "yesterday" if days == 1 else f"{days} days ago"

    def briefing(self, *, days: int = 3, cap: int | None = None) -> str:
        """What is true right now, for a brain that cannot open a file.

        The vault was write-only until this existed: Canvas sync wrote
        `classes/upcoming.md` and nothing ever read it back, so asking "what's
        due this week" got "I'm not sure what's on your calendar yet" from a
        brain holding nothing but EVIE.md.

        Today's date leads, because a model has no clock and "this week"
        cannot be resolved without one.
        """
        cap = self.BRIEFING_CAP if cap is None else cap
        today = _dt.date.today()
        parts = [f"Today is {today:%A %-d %B %Y}."]

        if (deadlines := self._read(self.deadlines_path())):
            # Say when, always. A confident answer from a week-old file is
            # worse than a hedged one, and only she can know to hedge.
            age = self.deadlines_age()
            stamp = f" (last synced {self._how_long(age)})" if age else ""
            parts.append(f"## What is due{stamp}\n\n" + deadlines)
        if (log := self.recent_log(days)):
            parts.append(f"## The last {days} days\n\n" + log)

        return _trim("\n\n".join(parts), cap)

    def skills_dir(self) -> Path:
        from ..skills import SKILLS_DIR

        return self.root / SKILLS_DIR

    def skill_triggers(self) -> dict[str, tuple[str, ...]]:
        """What to say to run each installed skill, for the router.

        Imported lazily: `evie.skills` imports config, and config does not
        need to know the vault exists.
        """
        from ..skills import triggers

        return triggers(self.root) if self.exists else {}

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text().strip()
        except OSError:
            return ""


# -- keeping the vault somewhere other than one laptop ---------------------
#
# The storage worry is misplaced: markdown is tiny. A year of daily logs plus
# course notes runs to a few megabytes -- less than one phone photo. What is
# worth solving is the other half of the question: a single copy on a single
# machine is one spilled drink from gone, and unreachable from anywhere else.
#
# Git answers both, free: a private GitHub repo costs nothing, versions every
# change, and syncs to any machine. It also stays plain files, so Obsidian and
# any agentic brain keep working on it unchanged.

import subprocess


class GitError(Exception):
    pass


def check_remote(url: str) -> str | None:
    """Explain a malformed git remote, or return None if it looks fine.

    Git's own error for a mangled URL is unhelpful, and the two accepted
    shapes are easy to splice together -- pasting a browser URL after the
    scp-style `git@host:` prefix produces something that looks plausible and
    cannot work.
    """
    url = url.strip()
    if not url:
        return "empty remote"

    # The common splice: git@github.com:https://github.com/you/repo
    if url.startswith("git@") and "://" in url:
        tail = url.split("://", 1)[1]
        path = tail.split("/", 1)[1] if "/" in tail else "you/repo"
        return (
            "that is an SSH prefix with a web URL pasted after it. Use one form:\n"
            f"    https://github.com/{path}.git      (asks for a token)\n"
            f"    git@github.com:{path}.git          (needs an SSH key)"
        )

    if url.startswith(("https://", "http://", "ssh://", "git@")):
        # A browser URL for the repo page works; the page for a file does not.
        if "/tree/" in url or "/blob/" in url:
            return "that is a link to a page inside the repo, not the repo itself"
        return None

    # A local path is a valid remote and useful for testing -- but only accept
    # one that is written like a path. Path("my github repo").parent is ".",
    # which always exists, so a bare existence check waves through anything.
    if url.startswith(("/", "./", "../", "~")) or Path(url).exists():
        return None

    return (
        "that does not look like a git remote. Expected something like:\n"
        "    https://github.com/you/evie-vault.git\n"
        "    git@github.com:you/evie-vault.git"
    )


class VaultGit:
    """Version control for the vault, using whatever git is already installed."""

    def __init__(self, vault: "Vault") -> None:
        self.root = vault.root

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True
        )
        if check and result.returncode != 0:
            raise GitError((result.stderr or result.stdout).strip()[:500])
        return result

    @property
    def initialized(self) -> bool:
        return (self.root / ".git").is_dir()

    def remote(self) -> str | None:
        if not self.initialized:
            return None
        result = self._run("remote", "get-url", "origin", check=False)
        return result.stdout.strip() or None

    def setup(self, remote: str) -> None:
        """Turn the vault into a repo pointed at your own private remote."""
        if problem := check_remote(remote):
            raise GitError(problem)
        if not self.initialized:
            self._run("init")
            self._run("checkout", "-B", "main")
        # Conversation logs are personal; make it hard to publish them by
        # accident, and keep the noise out.
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".DS_Store\n.obsidian/workspace*\n")
        if self.remote():
            self._run("remote", "set-url", "origin", remote)
        else:
            self._run("remote", "add", "origin", remote)

    def sync(self, message: str | None = None) -> str:
        """Commit anything new and push. Returns a one-line summary."""
        if not self.initialized:
            raise GitError("vault is not a git repo yet -- run `evie memory setup <url>`")

        self._run("add", "-A")
        staged = self._run("diff", "--cached", "--name-only").stdout.split()

        if staged:
            import datetime as _d

            note = message or f"memory: {_d.datetime.now():%Y-%m-%d %H:%M}"
            self._run("-c", "user.email=evie@localhost", "-c", "user.name=E.V.I.E.",
                      "commit", "-m", note)

        if not self.remote():
            return f"committed {len(staged)} file(s); no remote set, nothing pushed"

        # Pull first: the same vault may have been written from another machine.
        self._run("pull", "--rebase", "origin", "main", check=False)
        push = self._run("push", "-u", "origin", "main", check=False)
        if push.returncode != 0:
            raise GitError((push.stderr or push.stdout).strip()[:500])
        return f"synced {len(staged)} changed file(s)" if staged else "already up to date"


def stats(vault: "Vault") -> dict:
    """What is actually in here, for anyone worried about disk."""
    files = [p for p in vault.root.rglob("*.md") if ".git" not in p.parts]
    total = sum(p.stat().st_size for p in files)
    days = sorted((vault.root / "daily").glob("*.md")) if vault.exists else []
    return {
        "files": len(files),
        "bytes": total,
        "days": len(days),
        "first": days[0].stem if days else None,
        "last": days[-1].stem if days else None,
    }
