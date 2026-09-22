"""Getting a credential from a person without it touching a command line.

Two keys have leaked during this project, both the same way, and neither was
carelessness: the documented way to set one put it in `argv`.

    echo 'export CANVAS_API_TOKEN=1773~...' >> ~/.zshrc

A secret on a command line is in shell history, is visible in `ps` while the
command runs, and sits in the scrollback ready to be copied into a chat
window along with whatever else was on screen. The second leak happened while
recovering from an unterminated quote in exactly that command.

So nothing here ever accepts a secret as an argument. It is read with
`getpass`, written once, and never echoed, logged, or put in an exception
message -- an error is the moment someone pastes their whole terminal at you.
"""

from __future__ import annotations

import os
from pathlib import Path

# One block, so re-running replaces a line instead of stacking exports that
# silently shadow each other.
MARK_START = "# >>> evie managed secrets >>>"
MARK_END = "# <<< evie managed secrets <<<"

# Things people type when they mean "I have not done this yet".
PLACEHOLDERS = {
    "your_token_here", "your_key_here", "paste_here", "xxx", "todo",
    "<token>", "<key>", "token", "key", "none", "null",
}


def shell_rc() -> Path | None:
    """The file to write, or None when guessing would be worse than asking.

    Writing to the wrong rc file is the quiet failure: it succeeds, reports
    success, and the variable is still unset in every terminal they open.
    """
    name = Path(os.environ.get("SHELL", "")).name
    if name == "zsh":
        return Path.home() / ".zshrc"
    if name in ("bash", "sh"):
        return Path.home() / ".bashrc"
    if name == "fish":
        return Path.home() / ".config" / "fish" / "config.fish"
    return None


def why_not(value: str) -> str | None:
    """Why this cannot be a credential, or None if it might be.

    Deliberately shape-only. A paste that silently "succeeds" with an empty
    string is worse than an error, because the failure surfaces later as an
    authentication problem nobody connects back to this.
    """
    if value is None or not value.strip():
        return "nothing was entered"
    if value != value.strip():
        return "it has whitespace around it — the paste picked up a newline or a space"
    if value.strip().lower() in PLACEHOLDERS:
        return "that is the placeholder, not a token"
    if len(value) < 12:
        return f"it is only {len(value)} characters, which is too short to be a key"
    if any(c in value for c in "'\"`$\\"):
        return "it contains a quote or a backslash, so the paste picked up part of a command"
    return None


def store(var: str, value: str, rc: Path) -> Path:
    """Write `export var=value` into the managed block, replacing any previous
    line for the same variable. Returns the file written."""
    if why_not(value):
        raise ValueError(why_not(value))

    line = (
        f"set -gx {var} {value}"
        if rc.name == "config.fish"
        else f"export {var}={value}"
    )
    existing = rc.read_text() if rc.is_file() else ""

    if MARK_START in existing and MARK_END in existing:
        before, _, rest = existing.partition(MARK_START)
        body, _, after = rest.partition(MARK_END)
        kept = [
            ln for ln in body.splitlines()
            if ln.strip() and not _sets(ln, var)
        ]
    else:
        before, after, kept = existing, "", []

    block = "\n".join([MARK_START, *kept, line, MARK_END])
    parts = [before.strip("\n"), block, after.strip("\n")]

    rc.parent.mkdir(parents=True, exist_ok=True)
    fresh = not rc.exists()
    rc.write_text("\n\n".join(p for p in parts if p) + "\n")
    if fresh:
        # Only on a file we created. Tightening the mode of an rc file
        # someone already has is not ours to decide.
        rc.chmod(0o600)
    return rc


def _sets(line: str, var: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith(f"export {var}=")
        or stripped.startswith(f"set -gx {var} ")
    )


def manual_instructions(var: str, rc: Path | None) -> str:
    """What to do when we will not write the file ourselves.

    Note the absence of a copy-pasteable `export KEY=value` line: printing one
    is how the value gets back onto a command line, which is the thing this
    module exists to prevent.
    """
    where = str(rc) if rc else "your shell's startup file"
    return (
        f"Add a line to {where} that exports {var}, then open a new terminal.\n"
        f"Type the value in rather than pasting a whole command, so it stays "
        f"out of your shell history."
    )
