"""The command centre's data file, built from what is actually true.

The guide this came from has you type today's numbers into `jarvis_data.js` by
hand every morning. E.V.I.E. already knows them: which brains are reachable,
how many requests each has spent today, what Canvas says is due, what was said
yesterday. So the page is shipped once and this regenerates its data.

The rule that shapes everything here is the one the dashboard would otherwise
break first: **never show a state we have not established.** `brains list` can
say a CLI is installed, because `shutil.which` found it; it cannot say the CLI
is logged in, because only a real call settles that. A green dot for `claude`
while its session had silently expired is exactly the lie that cost a voice
session -- so `Health.UNVERIFIED` renders as "unverified", a distinct state
with its own word, and `--probe` is what turns it into a fact.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .brains.base import Health

# Health -> the state name the page knows. Kept here rather than in the HTML
# so the page needs no knowledge of the brain layer.
_STATE = {
    Health.OK: "ready",
    Health.UNVERIFIED: "unverified",
    Health.EXHAUSTED: "exhausted",
    Health.UNAUTHENTICATED: "offline",
    Health.MISSING: "offline",
    Health.UNKNOWN: "unverified",
}

LOG_LINES = 7
_LOG_LINE = re.compile(r"^-\s+\*\*(\d{2}:\d{2})\s+([^:]+):\*\*\s*(.+)$")
_ROW = re.compile(r"^-\s+\[( |x)\]\s+\*\*(.+?)\*\*\s+—\s+(.+?)(?:\s+\((.+?)\))?"
                  r"(?:\s+·\s+([\d.]+)\s+pts)?(?:\s+·\s+\*\*missing\*\*)?\s*$")


@dataclass
class Deadlines:
    """What `classes/upcoming.md` says, read back rather than re-fetched."""

    total: int = 0
    late: int = 0
    done: int = 0
    missing: int = 0
    points: float = 0.0
    soonest: str = ""
    titles: list[str] = field(default_factory=list)
    synced: _dt.timedelta | None = None

    @property
    def open(self) -> int:
        return self.total - self.done


def read_deadlines(vault) -> Deadlines:
    """Parse the file `evie canvas sync` wrote.

    Reading our own markdown back is a little inelegant, and it is still the
    right call: the vault is the single source of truth every brain already
    answers from, so the dashboard showing anything else would mean two
    versions of "what is due" that could disagree.
    """
    out = Deadlines(synced=vault.deadlines_age() if vault else None)
    if not vault:
        return out
    try:
        text = vault.deadlines_path().read_text()
    except OSError:
        return out

    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        if not (m := _ROW.match(line.strip())):
            continue
        done = m.group(1) == "x"
        out.total += 1
        out.done += done
        if section.startswith("late") and not done:
            out.late += 1
        if "**missing**" in line and not done:
            out.missing += 1
        if m.group(5) and not done:
            # "At stake" means it can still be lost. Points already banked on a
            # submitted assignment are not at risk, and counting them makes the
            # headline figure bigger and wrong in the same stroke.
            out.points += float(m.group(5))
        if not done:
            out.titles.append(m.group(3).strip())
            if not out.soonest and not section.startswith("late"):
                out.soonest = f"{m.group(3).strip()} — {m.group(2).strip()}"
    return out


def read_log(vault, lines: int = LOG_LINES) -> list[str]:
    """The last few things said, newest last, as one line each."""
    if not vault:
        return []
    out: list[str] = []
    for raw in (vault.recent_log(2) or "").splitlines():
        if m := _LOG_LINE.match(raw.strip()):
            said = m.group(3).strip()
            out.append(f"{m.group(1)} {m.group(2)}: {said[:78]}"
                       + ("…" if len(said) > 78 else ""))
    return out[-lines:]


def _pct(part: float, whole: float) -> int:
    return int(round(100 * part / whole)) if whole else 0


async def build(registry, vault, settings, *, probe: bool = False) -> dict:
    """Everything the page renders. No field is invented; a missing source is
    reported as missing, which is why several values are "" rather than 0."""
    today = _dt.date.today()
    dl = read_deadlines(vault)

    connectors = await _connectors(registry, probe=probe)

    stats = []
    if dl.total:
        stats.append({"key": "handed in", "value": f"{dl.done}/{dl.total}",
                      "pct": _pct(dl.done, dl.total),
                      "note": f"{dl.open} still open"})
        stats.append({"key": "late", "value": str(dl.late),
                      "pct": _pct(dl.late, dl.total),
                      "note": f"{dl.missing} marked missing by Canvas" if dl.missing else ""})
        if dl.missing:
            stats.append({"key": "missing", "value": str(dl.missing),
                          "pct": _pct(dl.missing, dl.total),
                          "note": "Canvas has these as not handed in"})

    figures = []
    if dl.total:
        figures = [
            {"value": str(dl.open), "label": "still to do"},
            {"value": str(dl.late), "label": "overdue"},
        ]
        if dl.points:
            figures.append({"value": f"{dl.points:g}", "label": "points at stake"})
    figures.append({"value": str(sum(registry.used_today(n) for n in registry.names())),
                    "label": "requests today"})

    chip = ""
    if dl.synced is None and vault:
        chip = "Canvas has never been synced — run `evie canvas sync`."
    elif dl.synced is not None and dl.synced > _dt.timedelta(hours=24):
        chip = f"Canvas last synced {vault._how_long(dl.synced)} — these may be stale."
    elif dl.late:
        chip = f"{dl.late} overdue, {dl.missing} of them marked missing." if dl.missing \
            else f"{dl.late} overdue."

    owner = ""
    if vault and (ident := vault.identity()):
        if m := re.search(r"You are E\.V\.I\.E\., (.+?)'s assistant", ident):
            owner = m.group(1)

    return {
        "greeting": f"Good {_part_of_day()}{', ' + owner if owner else ''}."
                    f" {dl.open} things open." if dl.total
                    else f"Good {_part_of_day()}{', ' + owner if owner else ''}.",
        "generated": _dt.datetime.now().strftime("%a %-d %b, %-I:%M %p").lower(),
        "connectors": connectors,
        "stats": stats,
        "figures": figures,
        "priorities": dl.titles[:3],
        "headline": dl.soonest and f"Next up: {dl.soonest}" or "",
        "chip": chip,
        "closer": "WHAT SHOULD I HANDLE FIRST",
        "log": read_log(vault),
        "audio": "",
        "date": today.isoformat(),
    }


async def _connectors(registry, *, probe: bool = False) -> list[dict]:
    """One row per brain, saying only what we have established.

    Without `probe` this is `health()`, which is cheap and therefore vague: a
    CLI brain reports UNVERIFIED because `shutil.which` found the binary and
    nothing more. With `probe` it is `selftest.reach()` -- one real inference
    per brain -- so "unverified" becomes ready or offline for real. That is the
    difference between a dashboard you can read at a glance and one that once
    showed `claude` green for a whole voice session while it was logged out.
    """
    health = await registry.health()
    proven: dict[str, bool] = {}
    if probe:
        from .selftest import reach

        proven = {r.name: r.ok for r in await reach(registry) if not r.skipped}

    rows = []
    for name in registry.names():
        if registry.is_parked(name):
            rows.append({"name": name, "state": "parked", "label": "not set up",
                         "used": ""})
            continue
        status = health.get(name)
        state = _STATE.get(status.health if status else Health.UNKNOWN, "unverified")
        if name in proven:
            state, label = ("ready", "answered") if proven[name] else ("offline", "no answer")
        else:
            label = None
        rows.append({"name": name, "state": state, "label": label,
                     "used": registry.headroom(name)})
    return rows


def _part_of_day() -> str:
    hour = _dt.datetime.now().hour
    return "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"


def render_data(data: dict) -> str:
    """The `evie_data.js` the page loads. A plain assignment, nothing clever."""
    return (
        "// Written by `evie dashboard`. Every value comes from real state --\n"
        "// brain health, the Canvas file in your vault, the daily log.\n"
        "// Edit the vault, not this file; the next run overwrites it.\n"
        "window.EVIE_DATA = "
        + json.dumps(data, indent=2, ensure_ascii=False)
        + ";\n"
    )


def write(target: Path, data: dict, template: Path) -> Path:
    """Put the page and its data side by side. Returns the page."""
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    page = target / "dashboard.html"
    page.write_text(template.read_text())
    (target / "evie_data.js").write_text(render_data(data))
    return page
