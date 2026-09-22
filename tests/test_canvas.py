"""Canvas, read defensively.

Instructure's API docs were unreachable from the machine this was written on,
and the shape genuinely varies by `plannable_type` -- an assignment, a quiz
and a planner note do not agree on where the title or the date lives. So the
parser tries the plausible names, and these tests pin every variant it claims
to handle rather than one guessed-at happy path.

If a real sync turns up a shape none of these match, `describe_shape` prints
what actually arrived and a case gets added here.
"""

import datetime as _dt

import httpx
import pytest

from evie.sources.canvas import (
    Canvas,
    CanvasError,
    Deadline,
    _as_deadline,
    _explain,
    _next_link,
    _when,
    describe_shape,
    render,
    write,
)

UTC = _dt.timezone.utc


def item(**over):
    """A planner item in the shape the controller documents."""
    base = {
        "plannable_type": "assignment",
        "plannable_date": "2026-09-25T23:59:00Z",
        "course_id": 101,
        "context_name": "AICE ENG LANG AS-Sigler",
        "html_url": "https://x.instructure.com/courses/101/assignments/9",
        "submissions": {"submitted": False, "missing": False, "graded": False},
        "plannable": {
            "id": 9,
            "title": "Directed Writing draft",
            "due_at": "2026-09-25T23:59:00Z",
            "points_possible": 25,
        },
    }
    base.update(over)
    return base


class TestReadingWhateverCanvasSent:
    def test_an_assignment(self):
        d = _as_deadline(item(), {})
        assert d.title == "Directed Writing draft"
        assert d.course == "AICE ENG LANG AS-Sigler"
        assert d.points == 25
        assert d.due == _dt.datetime(2026, 9, 25, 23, 59, tzinfo=UTC)

    def test_a_quiz_named_with_name_not_title(self):
        """Not every plannable uses `title`, and guessing one would drop the
        whole kind silently."""
        d = _as_deadline(
            item(plannable_type="quiz",
                 plannable={"name": "Unit 3 quiz", "due_at": "2026-09-26T16:00:00Z"}),
            {},
        )
        assert d.title == "Unit 3 quiz"
        assert d.kind == "quiz"

    def test_a_planner_note_dated_with_todo_date(self):
        d = _as_deadline(
            {"plannable_type": "planner_note", "plannable_date": None,
             "plannable": {"title": "Start the essay", "todo_date": "2026-09-24T12:00:00Z"}},
            {},
        )
        assert d.title == "Start the essay"
        assert d.due == _dt.datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    def test_the_top_level_date_wins_over_the_nested_one(self):
        """`plannable_date` is Canvas's own normalised field; the nested one
        varies by kind."""
        d = _as_deadline(
            item(plannable_date="2026-10-01T10:00:00Z",
                 plannable={"title": "x", "due_at": "2026-09-25T23:59:00Z"}),
            {},
        )
        assert d.due.day == 1

    def test_the_course_falls_back_to_the_course_list(self):
        d = _as_deadline(item(context_name=None, course_id=101), {101: "AI301"})
        assert d.course == "AI301"

    def test_an_unknown_course_is_unfiled_not_a_crash(self):
        assert _as_deadline(item(context_name=None, course_id=999), {}).course == "Unfiled"

    @pytest.mark.parametrize("kind", ["announcement", "calendar_event",
                                      "assessment_request"])
    def test_noise_is_dropped(self, kind):
        """An announcement is not a deadline. Listing it as one makes the
        file useless for the question it exists to answer."""
        assert _as_deadline(item(plannable_type=kind), {}) is None

    def test_an_item_with_no_title_anywhere_is_dropped(self):
        assert _as_deadline(item(plannable={}, plannable_date=None), {}) is None

    def test_submission_state_is_carried(self):
        d = _as_deadline(item(submissions={"submitted": True, "missing": False}), {})
        assert d.submitted and not d.missing
        d = _as_deadline(item(submissions={"submitted": False, "missing": True}), {})
        assert d.missing

    def test_submissions_false_instead_of_an_object(self):
        """Canvas sends `false` here when nothing is submittable, not {}."""
        d = _as_deadline(item(submissions=False), {})
        assert d is not None and not d.submitted


class TestDates:
    def test_a_trailing_z_parses(self):
        """fromisoformat rejected a literal Z before 3.11, and this project
        supports 3.11 up."""
        assert _when("2026-09-25T23:59:00Z") == _dt.datetime(2026, 9, 25, 23, 59, tzinfo=UTC)

    def test_an_offset_parses(self):
        assert _when("2026-09-25T19:59:00-04:00").astimezone(UTC).hour == 23

    def test_a_naive_stamp_is_made_aware(self):
        """Comparing naive and aware raises, and would take down a whole sync
        over one oddly-formatted date."""
        assert _when("2026-09-25T23:59:00").tzinfo is not None

    @pytest.mark.parametrize("junk", ["", None, "never", 12345, "2026-13-45"])
    def test_junk_is_none_not_an_exception(self, junk):
        assert _when(junk) is None

    def test_overdue_never_compares_naive_to_aware(self):
        past = Deadline("c", "t", _dt.datetime(2020, 1, 1, tzinfo=UTC))
        assert past.overdue is True

    def test_something_submitted_is_not_overdue(self):
        past = Deadline("c", "t", _dt.datetime(2020, 1, 1, tzinfo=UTC), submitted=True)
        assert past.overdue is False

    def test_no_due_date_is_not_overdue(self):
        assert Deadline("c", "t", None).overdue is False

    def test_it_says_when_the_way_a_person_would(self):
        now = _dt.datetime.now(UTC)
        assert "today" in Deadline("c", "t", now + _dt.timedelta(hours=2)).when()
        assert "tomorrow" in Deadline("c", "t", now + _dt.timedelta(days=1)).when()
        assert Deadline("c", "t", None).when() == "no due date"


class TestPagination:
    def test_it_finds_the_next_link(self):
        header = (
            '<https://x/api/v1/planner/items?page=1>; rel="current",'
            '<https://x/api/v1/planner/items?page=2>; rel="next",'
            '<https://x/api/v1/planner/items?page=9>; rel="last"'
        )
        assert _next_link(header) == "https://x/api/v1/planner/items?page=2"

    @pytest.mark.parametrize("header", [
        "", None,
        '<https://x/api/v1/x?page=1>; rel="current"',   # last page: no next
        '<https://x/api/v1/x?page=1>; rel="prev"',
    ])
    def test_no_next_means_stop(self, header):
        assert _next_link(header) is None

    async def test_it_follows_pages_to_the_end(self, monkeypatch):
        pages = [
            ([{"id": 1}], '<https://x/api/v1/courses?page=2>; rel="next"'),
            ([{"id": 2}], ""),
        ]
        seen = []

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params=None, headers=None):
                seen.append(url)
                body, link = pages[len(seen) - 1]
                return httpx.Response(
                    200, json=body, headers={"link": link},
                    request=httpx.Request("GET", url),
                )

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
        rows = await Canvas("https://x.instructure.com", "t")._pages("courses")
        assert [r["id"] for r in rows] == [1, 2]
        assert seen[1] == "https://x/api/v1/courses?page=2", "next must be followed as given"

    async def test_a_transport_failure_is_a_canvas_error(self, monkeypatch):
        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **kw):
                raise httpx.ConnectError("no route")

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
        with pytest.raises(CanvasError, match="could not reach"):
            await Canvas("https://x.instructure.com", "t")._pages("courses")

    async def test_html_instead_of_json_says_so(self, monkeypatch):
        """Pointing at a login page is the likeliest URL mistake, and a raw
        JSONDecodeError explains none of it."""
        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params=None, headers=None):
                return httpx.Response(200, text="<html>Log in</html>",
                                      request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
        with pytest.raises(CanvasError, match="not JSON"):
            await Canvas("https://x.instructure.com", "t")._pages("courses")


class TestErrors:
    def test_a_rejected_token_explains_the_once_only_problem(self):
        exc = _explain(401, '{"errors":[{"message":"Invalid access token."}]}', "https://x")
        assert "New Access Token" in str(exc)
        assert "shown" in str(exc) and "once" in str(exc)

    def test_a_404_blames_the_url_not_the_token(self):
        assert "URL" in str(_explain(404, "", "https://x"))

    def test_anything_else_keeps_the_status(self):
        assert "503" in str(_explain(503, "down for maintenance", "https://x"))

    def test_no_url_configured_names_the_file_and_the_key(self):
        with pytest.raises(CanvasError, match="config.yaml"):
            Canvas("", "token")

    def test_no_token_names_the_env_var_and_where_to_get_one(self):
        with pytest.raises(CanvasError) as exc:
            Canvas("https://x.instructure.com", "")
        assert "CANVAS_API_TOKEN" in str(exc.value)
        assert "not in this repo" in str(exc.value)

    def test_a_bare_host_gets_a_scheme(self):
        assert Canvas("broward.instructure.com", "t").base_url.startswith("https://")

    def test_a_trailing_slash_does_not_double_up(self):
        assert Canvas("https://x.instructure.com/", "t").base_url.endswith("com")


class TestWritingItDown:
    def _vault(self, tmp_path):
        from evie.memory import Vault

        return Vault(tmp_path / "v").ensure("test")

    def _some(self):
        soon = _dt.datetime.now(UTC) + _dt.timedelta(days=2)
        past = _dt.datetime.now(UTC) - _dt.timedelta(days=3)
        return [
            Deadline("AICE ENG LANG", "Directed Writing", soon, points=25),
            Deadline("AI301", "Lab 4", past, missing=True),
        ]

    def test_it_writes_a_rollup_and_one_file_per_course(self, tmp_path):
        vault = self._vault(tmp_path)
        paths = write(vault, self._some())
        assert (vault.root / "classes" / "upcoming.md").exists()
        assert (vault.root / "classes" / "ai301" / "deadlines.md").exists()
        assert len(paths) == 3

    def test_late_work_is_separated_from_upcoming(self, tmp_path):
        body = render(self._some(), title="Upcoming")
        assert "## Late" in body and "## Coming up" in body
        assert body.index("## Late") < body.index("## Coming up"), "late first"

    def test_missing_work_is_marked(self, tmp_path):
        assert "**missing**" in render(self._some(), title="x")

    def test_submitted_work_is_ticked(self):
        done = [Deadline("c", "Essay", None, submitted=True)]
        assert "- [x]" in render(done, title="x")

    def test_hand_written_notes_survive_a_resync(self, tmp_path):
        """The vault is somewhere Isaac writes too. A sync that flattened his
        own notes would make it a place he stops trusting."""
        vault = self._vault(tmp_path)
        write(vault, self._some())
        path = vault.root / "classes" / "upcoming.md"
        path.write_text(path.read_text() + "\n## My own notes\n\nAsk Sigler about this.\n")

        write(vault, self._some())
        after = path.read_text()
        assert "Ask Sigler about this." in after
        assert after.count("## My own notes") == 1

    def test_resyncing_does_not_stack_duplicate_blocks(self, tmp_path):
        vault = self._vault(tmp_path)
        for _ in range(3):
            write(vault, self._some())
        body = (vault.root / "classes" / "upcoming.md").read_text()
        assert body.count("<!-- evie:canvas -->") == 1

    def test_nothing_due_says_so(self):
        assert "Nothing due." in render([], title="Upcoming")

    def test_a_course_name_with_slashes_cannot_escape_the_folder(self, tmp_path):
        """A course called "AP STAT 1/2" must not write to a sibling
        directory, or worse."""
        vault = self._vault(tmp_path)
        write(vault, [Deadline("../../etc/AP STAT 1/2", "x", None)])
        written = list((vault.root / "classes").rglob("deadlines.md"))
        assert written, "it still wrote something"
        for path in written:
            assert vault.root in path.parents


class TestShapeReport:
    """For when a sync returns nothing and the question is whether Canvas sent
    nothing or whether this module read it wrong."""

    def test_it_names_the_keys_that_actually_arrived(self):
        report = describe_shape([item()])
        assert "plannable_type" in report
        assert "points_possible" in report, "the nested keys are the useful part"

    def test_it_lists_every_kind_present(self):
        report = describe_shape([item(), item(plannable_type="quiz")])
        assert "assignment" in report and "quiz" in report

    def test_an_empty_response_says_empty(self):
        assert "no planner items" in describe_shape([])


class TestTheSeamRenders:
    """An HTML block in CommonMark runs until a blank line.

    A hand-written `## My notes` butted straight against the closing marker
    gets swallowed into the comment and never renders as a heading in
    Obsidian -- which is the whole point of writing markdown rather than JSON.
    """

    def _vault(self, tmp_path):
        from evie.memory import Vault

        return Vault(tmp_path / "v").ensure("test")

    def test_a_blank_line_follows_the_closing_marker(self, tmp_path):
        vault = self._vault(tmp_path)
        write(vault, [Deadline("AI301", "Lab 4", None)])
        path = vault.root / "classes" / "upcoming.md"
        path.write_text(path.read_text().rstrip() + "\n## My notes\n\nAsk Sigler.\n")

        write(vault, [Deadline("AI301", "Lab 4", None)])
        lines = path.read_text().splitlines()
        close = lines.index("<!-- /evie:canvas -->")
        assert lines[close + 1] == "", "the heading after it would not render"
        assert "## My notes" in lines

    def test_text_above_the_block_is_kept_too(self, tmp_path):
        vault = self._vault(tmp_path)
        path = vault.root / "classes"
        path.mkdir(parents=True, exist_ok=True)
        (path / "upcoming.md").write_text("# My own title\n\nSome preamble.\n")

        write(vault, [Deadline("AI301", "Lab 4", None)])
        body = (path / "upcoming.md").read_text()
        assert body.startswith("# My own title")
        assert "Some preamble." in body
        assert "<!-- evie:canvas -->" in body

    def test_it_never_grows_blank_lines_on_repeated_syncs(self, tmp_path):
        vault = self._vault(tmp_path)
        path = vault.root / "classes" / "upcoming.md"
        for _ in range(4):
            write(vault, [Deadline("AI301", "Lab 4", None)])
        assert "\n\n\n" not in path.read_text()
