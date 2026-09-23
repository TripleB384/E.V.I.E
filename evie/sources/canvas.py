"""Canvas LMS -- assignments and due dates, synced into the vault.

Deliberately not an MCP server. MCP tools only reach CLI brains, so every
"what's due Thursday" would boot a Claude Code session: 5-11s and real plan
allowance, for a question groq answers in 0.3s. Writing markdown into the
vault instead means the fast free brain can answer it, it survives a Canvas
outage, it works offline, and you can read it in Obsidian.

The exact field names inside a planner item could not be verified while this
was written -- Instructure's docs were unreachable -- and they genuinely vary
by `plannable_type`: an assignment, a quiz, a calendar event and a planner
note do not agree on where the title lives. So the parsing below tries the
plausible names in order and `describe_shape()` reports what a real response
actually contained. Guessing silently is how you end up with an empty vault
and no idea why.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

# Canvas defaults to 10 items a page and caps per_page at an unpublished
# number, so the Link header is the only reliable way to know there is more.
PER_PAGE = 100
MAX_PAGES = 20  # a stop, so a malformed Link header cannot loop forever
TIMEOUT = 30.0


class CanvasError(Exception):
    """Something went wrong talking to Canvas, said in a way you can act on."""


def missing_token_advice(var: str = "CANVAS_API_TOKEN") -> str:
    """Why there is no token, which is two different problems.

    A token written to a shell rc file does not reach a shell that was
    already running, so "I just saved it and it says it is not set" is the
    expected outcome of checking in the same terminal -- and needs the
    opposite advice from "you never set one".

    Prints no `export` line in either case. This message is read at exactly
    the moment someone is stuck and willing to paste anything, which is how
    two credentials have already leaked from this project.
    """
    from ..secrets import is_stored, shell_rc

    rc = shell_rc()
    if is_stored(var, rc):
        return (
            f"${var} is saved in {rc}, but this shell started before it was "
            f"written, so it has not been picked up.\n"
            f"  Run:  source {rc}\n"
            f"  ...or just open a new terminal."
        )
    return (
        f"no Canvas token yet. Run:\n"
        f"    evie canvas setup yourdistrict.instructure.com\n"
        f"  It will prompt you for one — nothing is echoed and nothing "
        f"reaches your shell history."
    )


def clean_base_url(raw: str) -> str:
    """Scheme and host, whatever someone pastes.

    What people actually have to hand is the URL in their browser bar, and
    that is `https://browardschools.instructure.com/?login_success=1` -- a
    redirect artefact, a trailing slash and a query string. Keeping any of
    them turns every API call into a 404 for a reason nobody would guess.
    """
    text = (raw or "").strip().strip("'\"")
    if not text:
        raise ValueError("no Canvas URL given")
    if "://" not in text:
        text = "https://" + text
    parts = urlparse(text)
    host = parts.netloc or parts.path.split("/")[0]
    plausible = host and " " not in host and (
        "." in host or ":" in host or host == "localhost"
    )
    if not plausible:
        raise ValueError(
            f"{raw!r} does not look like a Canvas host. It should be something "
            f"like yourdistrict.instructure.com"
        )
    scheme = parts.scheme if parts.scheme in ("http", "https") else "https"
    return f"{scheme}://{host}"


@dataclass
class Deadline:
    """One thing that is due, flattened out of whatever shape Canvas used."""

    course: str
    title: str
    due: _dt.datetime | None
    url: str = ""
    points: float | None = None
    kind: str = "assignment"
    submitted: bool = False
    missing: bool = False

    @property
    def overdue(self) -> bool:
        if self.due is None or self.submitted:
            return False
        return self.due < _dt.datetime.now(_dt.timezone.utc)

    def on(self) -> str:
        """The date itself, for anything written to a file.

        `when()` is computed at the moment it is called, so a file synced on
        Monday still claims "tomorrow" on Friday. Spoken aloud that is right;
        written down it rots. The vault gets this instead, and the briefing
        carries today's date so a model can work out "tomorrow" at the moment
        it is asked rather than the moment it was synced.
        """
        if self.due is None:
            return "no due date"
        local = self.due.astimezone()
        return f"{local:%a %-d %b}, {local:%-I:%M %p}".replace("AM", "am").replace(
            "PM", "pm"
        )

    def when(self) -> str:
        """How a person would say it, not an ISO timestamp."""
        if self.due is None:
            return "no due date"
        now = _dt.datetime.now(_dt.timezone.utc)
        days = (self.due.date() - now.date()).days
        clock = self.due.astimezone().strftime("%-I:%M %p").lower()
        if days == 0:
            return f"today {clock}"
        if days == 1:
            return f"tomorrow {clock}"
        if days == -1:
            return f"yesterday {clock}"
        if -7 < days < 0:
            return f"{-days} days ago"
        if 0 < days < 7:
            return f"{self.due.astimezone():%A} {clock}"
        return f"{self.due.astimezone():%a %-d %b} {clock}"


# -- reading whatever Canvas sent ------------------------------------------


def _first(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = data.get(name)
        if value not in (None, ""):
            return value
    return None


def _when(raw: Any) -> _dt.datetime | None:
    """Canvas sends ISO 8601 with a literal Z, which fromisoformat rejected
    before 3.11. Everything here is tz-aware or None -- never naive, because
    comparing the two raises and would take a sync down over a timezone."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        when = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=_dt.timezone.utc)


def _as_deadline(item: dict[str, Any], courses: dict[int, str]) -> Deadline | None:
    """Flatten one planner item, whatever kind it is.

    Returns None for the kinds that are not work: announcements and
    calendar events with no due date are noise in a deadline list.
    """
    kind = str(item.get("plannable_type") or "").strip() or "unknown"
    if kind in ("announcement", "calendar_event", "assessment_request"):
        return None

    plannable = item.get("plannable")
    plannable = plannable if isinstance(plannable, dict) else {}

    title = _first(plannable, "title", "name", "todo_date") or _first(
        item, "plannable_title", "title"
    )
    if not title:
        return None

    due = _when(_first(item, "plannable_date")) or _when(
        _first(plannable, "due_at", "todo_date", "start_at")
    )

    course_id = item.get("course_id")
    course = (
        _first(item, "context_name")
        or courses.get(course_id if isinstance(course_id, int) else -1)
        or "Unfiled"
    )

    submissions = item.get("submissions")
    submissions = submissions if isinstance(submissions, dict) else {}

    points = _first(plannable, "points_possible")
    return Deadline(
        course=str(course),
        title=str(title).strip(),
        due=due,
        url=str(_first(item, "html_url") or ""),
        points=float(points) if isinstance(points, (int, float)) else None,
        kind=kind.replace("_", " "),
        submitted=bool(submissions.get("submitted")),
        missing=bool(submissions.get("missing")),
    )


def describe_shape(items: Iterable[dict[str, Any]]) -> str:
    """What a real response actually contained.

    For when the sync returns nothing and the question is whether Canvas sent
    nothing or whether this module read it wrong.
    """
    items = list(items)
    if not items:
        return "Canvas returned no planner items at all."
    first = items[0]
    plannable = first.get("plannable")
    lines = [
        f"{len(items)} items. First item keys: {', '.join(sorted(first))}",
        f"  plannable_type: {first.get('plannable_type')!r}",
        f"  plannable_date: {first.get('plannable_date')!r}",
    ]
    if isinstance(plannable, dict):
        lines.append(f"  plannable keys: {', '.join(sorted(plannable))}")
    kinds = sorted({str(i.get("plannable_type")) for i in items})
    lines.append(f"  kinds present: {', '.join(kinds)}")
    return "\n".join(lines)


# -- the client ------------------------------------------------------------


class Canvas:
    def __init__(self, base_url: str, token: str) -> None:
        if not base_url:
            raise CanvasError(
                "no Canvas URL configured. Run:\n"
                "    evie canvas setup yourdistrict.instructure.com"
            )
        if not token:
            raise CanvasError(missing_token_advice())
        self.base_url = clean_base_url(base_url)
        self._token = token

    @classmethod
    def from_settings(cls, settings) -> "Canvas":
        canvas = getattr(settings, "canvas", None)
        env = getattr(canvas, "token_env", "CANVAS_API_TOKEN")
        return cls(getattr(canvas, "base_url", ""), os.environ.get(env, ""))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    async def _pages(self, path: str, **params: Any) -> list[dict[str, Any]]:
        """Walk Link rel="next" until Canvas stops offering one.

        The next URL is opaque by Canvas's own instruction -- it already
        carries every parameter -- so it is followed as given rather than
        rebuilt.
        """
        url = f"{self.base_url}/api/v1/{path.lstrip('/')}"
        query: dict[str, Any] | None = {"per_page": PER_PAGE, **params}
        out: list[dict[str, Any]] = []

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(TIMEOUT, connect=10.0), follow_redirects=True
            ) as client:
                for _ in range(MAX_PAGES):
                    resp = await client.get(url, params=query, headers=self._headers())
                    if resp.status_code != 200:
                        raise _explain(resp.status_code, resp.text, self.base_url)
                    body = resp.json()
                    if isinstance(body, dict):
                        body = [body]
                    out.extend(x for x in body if isinstance(x, dict))
                    nxt = _next_link(resp.headers.get("link", ""))
                    if not nxt:
                        break
                    url, query = nxt, None
        except httpx.TransportError as exc:
            raise CanvasError(
                f"could not reach {self.base_url} ({type(exc).__name__}: {exc})"
            ) from exc
        except ValueError as exc:
            raise CanvasError(
                f"{self.base_url} answered with something that is not JSON. "
                f"Is the URL right? A login page would look like this. ({exc})"
            ) from exc
        return out

    async def courses(self) -> dict[int, str]:
        rows = await self._pages("courses", enrollment_state="active")
        return {
            row["id"]: str(_first(row, "name", "course_code") or f"course {row['id']}")
            for row in rows
            if isinstance(row.get("id"), int)
        }

    async def planner(self, days_back: int = 7, days_ahead: int = 60) -> list[dict]:
        today = _dt.date.today()
        return await self._pages(
            "planner/items",
            start_date=(today - _dt.timedelta(days=days_back)).isoformat(),
            end_date=(today + _dt.timedelta(days=days_ahead)).isoformat(),
        )

    async def deadlines(
        self, days_back: int = 7, days_ahead: int = 60
    ) -> tuple[list[Deadline], list[dict]]:
        """Everything due, plus the raw items so a shape mismatch is visible."""
        courses = await self.courses()
        raw = await self.planner(days_back, days_ahead)
        found = [d for d in (_as_deadline(i, courses) for i in raw) if d]
        found.sort(key=lambda d: (d.due is None, d.due or _dt.datetime.max.replace(
            tzinfo=_dt.timezone.utc)))
        return found, raw

    async def whoami(self) -> str:
        rows = await self._pages("users/self")
        return str(_first(rows[0], "name", "short_name") or "?") if rows else "?"


_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="next"', re.IGNORECASE)


def _next_link(header: str) -> str | None:
    match = _NEXT.search(header or "")
    return match.group(1) if match else None


def _explain(status: int, body: str, base_url: str) -> CanvasError:
    snippet = (body or "").strip()[:300]
    if status in (401, 403):
        return CanvasError(
            "Canvas rejected the token. Generate a fresh one at "
            "Account -> Settings -> '+ New Access Token'. A token is shown "
            "once and cannot be read back, so a half-copied one looks exactly "
            f"like an expired one. Canvas said: {snippet}"
        )
    if status == 404:
        return CanvasError(
            f"{base_url} has no Canvas API there. Check the URL is your "
            f"district's Canvas host and nothing more -- no /login, no course "
            f"path. Canvas said: {snippet}"
        )
    return CanvasError(f"Canvas returned HTTP {status}: {snippet}")


async def refresh(settings, vault) -> str | None:
    """Re-sync if the vault copy has gone stale. Never fatal.

    Nothing re-synced Canvas until this existed, so she answered confidently
    from whatever was last pulled by hand -- and a deadline added this morning
    was invisible until someone remembered to run the command.

    Returns a line worth printing, or None when nothing happened. Canvas being
    unreachable is not a reason to refuse to answer: the vault copy still
    works, and `briefing()` already tells her how old it is so she can say so.
    """
    hours = float(getattr(getattr(settings, "canvas", None), "refresh_hours", 0) or 0)
    if hours <= 0:
        return None

    age = vault.deadlines_age()
    if age is not None and age < _dt.timedelta(hours=hours):
        return None

    try:
        client = Canvas.from_settings(settings)
        found, _ = await client.deadlines(
            settings.canvas.days_back, settings.canvas.days_ahead
        )
    except CanvasError as exc:
        # Deliberately soft. The most likely causes are no wifi and an expired
        # token, and neither should stop her answering from what she has.
        return f"Canvas is out of reach, so deadlines may be stale ({exc})"
    except Exception as exc:  # noqa: BLE001 - a sync must never take a turn down
        return f"Canvas sync failed ({type(exc).__name__}), using the saved copy"

    write(vault, found)
    return None if age is None else f"Refreshed {len(found)} deadlines from Canvas."


# -- writing it down -------------------------------------------------------

MARK_START = "<!-- evie:canvas -->"
MARK_END = "<!-- /evie:canvas -->"


def render(deadlines: list[Deadline], *, title: str) -> str:
    """Markdown a person would not mind reading, and a model can quote."""
    now = _dt.datetime.now().strftime("%a %-d %b, %-I:%M %p").lower()
    lines = [f"# {title}", "", f"*Synced from Canvas {now}.*", ""]

    if not deadlines:
        lines += ["Nothing due.", ""]
        return "\n".join(lines)

    overdue = [d for d in deadlines if d.overdue]
    ahead = [d for d in deadlines if not d.overdue]

    if overdue:
        lines += ["## Late", ""]
        lines += [_row(d) for d in overdue]
        lines += [""]
    if ahead:
        lines += ["## Coming up", ""]
        lines += [_row(d) for d in ahead]
        lines += [""]
    return "\n".join(lines)


def _row(d: Deadline) -> str:
    box = "x" if d.submitted else " "
    bits = [f"- [{box}] **{d.on()}** — {d.title}"]
    if d.course:
        bits.append(f" ({d.course})")
    if d.points:
        bits.append(f" · {d.points:g} pts")
    if d.missing and not d.submitted:
        bits.append(" · **missing**")
    return "".join(bits)


def write(vault, deadlines: list[Deadline]) -> list:
    """Write the roll-up and a file per course. Returns the paths written.

    Only the block between the markers is replaced, so notes you add to these
    files by hand survive a sync. The vault is somewhere Isaac writes too.
    """
    written = []
    root = vault.root / "classes"
    root.mkdir(parents=True, exist_ok=True)

    written.append(_replace_block(root / "upcoming.md",
                                  render(deadlines, title="Upcoming")))

    by_course: dict[str, list[Deadline]] = {}
    for d in deadlines:
        by_course.setdefault(d.course, []).append(d)

    for course, items in by_course.items():
        folder = root / _slug(course)
        folder.mkdir(parents=True, exist_ok=True)
        written.append(
            _replace_block(folder / "deadlines.md", render(items, title=course))
        )
    return written


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def _replace_block(path, body: str):
    """Swap the generated block, leave everything around it alone.

    The blank line after the closing marker is not cosmetic. An HTML block in
    CommonMark runs until a blank line, so a hand-written `## My notes`
    butted straight against `<!-- /evie:canvas -->` gets swallowed into the
    comment and never renders as a heading in Obsidian.
    """
    existing = path.read_text() if path.exists() else ""

    if MARK_START in existing and MARK_END in existing:
        before, _, rest = existing.partition(MARK_START)
        _, _, after = rest.partition(MARK_END)
    else:
        # No marker yet. Append rather than prepend: a file someone has
        # already written in has a title at the top, and shoving a generated
        # block above it buries their own heading mid-document.
        before, after = existing, ""

    parts = [
        before.strip("\n"),
        f"{MARK_START}\n{body.rstrip()}\n{MARK_END}",
        after.strip("\n"),
    ]
    path.write_text("\n\n".join(p for p in parts if p) + "\n")
    return path
