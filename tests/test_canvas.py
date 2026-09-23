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

    def test_no_url_configured_names_the_command_to_fix_it(self):
        with pytest.raises(CanvasError, match="evie canvas setup"):
            Canvas("", "token")

    def test_no_token_points_at_setup_not_an_export_line(self, monkeypatch):
        """This error is read at the moment someone is stuck and willing to
        paste anything, so it must not hand them an `export KEY=value` to
        run. That is how two credentials have already leaked here."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr("evie.secrets.is_stored", lambda var, rc: False)
        with pytest.raises(CanvasError) as exc:
            Canvas("https://x.instructure.com", "")
        assert "evie canvas setup" in str(exc.value)
        assert "export " not in str(exc.value)

    def test_a_token_saved_but_not_loaded_says_source_not_setup(self, monkeypatch):
        """Exactly what happened on the Mac: saved to ~/.zshrc, then checked
        in the same terminal, which started before the file was written.
        Telling someone to set it up again would be wrong advice."""
        monkeypatch.setattr("evie.secrets.is_stored", lambda var, rc: True)
        monkeypatch.setattr("evie.secrets.shell_rc",
                            lambda: __import__("pathlib").Path("/home/x/.zshrc"))
        with pytest.raises(CanvasError) as exc:
            Canvas("https://x.instructure.com", "")
        message = str(exc.value)
        assert "source /home/x/.zshrc" in message
        assert "new terminal" in message
        assert "evie canvas setup" not in message, "it is already set up"
        assert "export " not in message

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


class TestTheUrlSomeoneActuallyHas:
    """What people have to hand is their browser bar, not a tidy hostname.

    Isaac pasted `https://browardschools.instructure.com/?login_success=1` --
    a redirect artefact, a trailing slash and a query string. Keeping any of
    them turns every API call into a 404 for a reason nobody would guess.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize("raw", [
        "https://browardschools.instructure.com/?login_success=1",
        "https://browardschools.instructure.com/",
        "https://browardschools.instructure.com/courses/101/assignments",
        "browardschools.instructure.com",
        "  https://browardschools.instructure.com  ",
        "'https://browardschools.instructure.com'",
        "https://browardschools.instructure.com#grades",
    ])
    def test_everything_reduces_to_scheme_and_host(self, raw):
        from evie.sources.canvas import clean_base_url

        assert clean_base_url(raw) == "https://browardschools.instructure.com"

    def test_a_local_canvas_keeps_its_port_and_scheme(self):
        """The dot check was rejecting localhost, which breaks anyone running
        Canvas locally -- and the integration test server."""
        from evie.sources.canvas import clean_base_url

        assert clean_base_url("http://localhost:8799") == "http://localhost:8799"

    @_pytest.mark.parametrize("bad", ["", "   ", "my school", "canvas", None])
    def test_nonsense_is_refused_with_an_example(self, bad):
        from evie.sources.canvas import clean_base_url

        with pytest.raises(ValueError, match="instructure.com|no Canvas URL"):
            clean_base_url(bad)

    def test_the_client_normalises_on_the_way_in(self):
        assert Canvas("browardschools.instructure.com/?login_success=1", "t").base_url == (
            "https://browardschools.instructure.com"
        )

    def test_the_missing_url_error_names_the_command_not_a_yaml_block(self):
        """A YAML snippet in an error message is something people paste into a
        shell. That is exactly how this bug was reported."""
        with pytest.raises(CanvasError) as exc:
            Canvas("", "token")
        assert "evie canvas setup" in str(exc.value)
        assert "base_url:" not in str(exc.value)


class TestNoCanvasErrorTeachesTheUnsafePattern:
    """The check that would have caught the one that survived.

    `export KEY=...` was removed from the base_url branch of the constructor
    and left in the token branch six lines below it, so the message shown at
    the exact moment someone is stuck was still the one instruction
    guaranteed to put a secret on a command line.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize("make", [
        lambda: Canvas("", "token"),
        lambda: Canvas("https://x.instructure.com", ""),
    ])
    def test_the_constructor_errors_are_clean(self, make, monkeypatch):
        monkeypatch.setattr("evie.secrets.is_stored", lambda var, rc: False)
        with pytest.raises(CanvasError) as exc:
            make()
        assert "export " not in str(exc.value)

    @_pytest.mark.parametrize("status,body", [
        (401, '{"errors":[{"message":"Invalid access token."}]}'),
        (403, ""),
        (404, ""),
        (500, "boom"),
    ])
    def test_the_http_errors_are_clean_too(self, status, body):
        from evie.sources.canvas import _explain

        assert "export " not in str(_explain(status, body, "https://x"))


class TestWrittenDatesDoNotRot:
    """`when()` is computed when it is called, so a file synced on Monday
    still claims "tomorrow" on Friday. Spoken aloud that is right; written
    down it is a lie that gets worse every day."""

    def test_the_file_carries_an_absolute_date(self):
        due = _dt.datetime.now(UTC) + _dt.timedelta(days=1)
        body = render([Deadline("AI301", "Lab 4", due)], title="Upcoming")
        assert "tomorrow" not in body
        assert f"{due.astimezone():%-d %b}" in body

    def test_speech_still_gets_the_relative_form(self):
        """Absolute dates read aloud are worse, not better: "Thursday" beats
        "the twenty-fifth of September"."""
        soon = Deadline("c", "t", _dt.datetime.now(UTC) + _dt.timedelta(days=1))
        assert soon.when().startswith("tomorrow")

    def test_a_file_read_a_week_later_is_still_true(self, tmp_path):
        from evie.memory import Vault

        due = _dt.datetime.now(UTC) + _dt.timedelta(days=2)
        vault = Vault(tmp_path / "v").ensure("test")
        write(vault, [Deadline("AI301", "Lab 4", due)])
        body = (vault.root / "classes" / "upcoming.md").read_text()
        # Nothing in the row depends on when it is read.
        for rots in ("today", "tomorrow", "yesterday", "days ago"):
            assert rots not in body.split("Synced from Canvas")[-1].split("\n", 1)[1]

    def test_no_due_date_still_says_so(self):
        assert Deadline("c", "t", None).on() == "no due date"


class TestStaleness:
    """Nothing re-synced Canvas, so she answered confidently from whatever
    was last pulled by hand. A deadline posted this morning stayed invisible
    until someone remembered the command."""

    import pytest as _pytest

    def _vault(self, tmp_path, age_hours=None):
        from evie.memory import Vault

        vault = Vault(tmp_path / "v").ensure("test")
        if age_hours is not None:
            write(vault, [Deadline("AI301", "old thing", None)])
            import os

            old = _dt.datetime.now().timestamp() - age_hours * 3600
            os.utime(vault.deadlines_path(), (old, old))
        return vault

    def _settings(self, tmp_path, **canvas):
        from evie.config import CanvasSettings, Settings

        return Settings(
            vault=tmp_path / "v",
            canvas=CanvasSettings(base_url="https://x.instructure.com", **canvas),
        )

    def test_a_fresh_copy_is_left_alone(self, tmp_path, monkeypatch):
        """A sync per question would be a network round trip on every
        "what's due", which is the cost the vault exists to avoid."""
        from evie.sources.canvas import refresh

        called = []
        monkeypatch.setattr(Canvas, "deadlines", lambda *a, **k: called.append(1))
        vault = self._vault(tmp_path, age_hours=1)
        import asyncio

        assert asyncio.run(refresh(self._settings(tmp_path), vault)) is None
        assert not called, "it should not have touched the network"

    async def test_a_stale_copy_is_refreshed(self, tmp_path, monkeypatch):
        from evie.sources.canvas import refresh

        async def fake(self, *a, **k):
            return [Deadline("AI301", "brand new thing", None)], []

        monkeypatch.setattr(Canvas, "deadlines", fake)
        monkeypatch.setenv("CANVAS_API_TOKEN", "1773~" + "a" * 40)
        vault = self._vault(tmp_path, age_hours=99)

        await refresh(self._settings(tmp_path), vault)
        assert "brand new thing" in vault.deadlines_path().read_text()

    async def test_canvas_being_down_is_not_fatal(self, tmp_path, monkeypatch):
        """No wifi and an expired token are the likely causes, and neither is
        a reason to refuse to answer from what she already has."""
        from evie.sources.canvas import refresh

        async def boom(self, *a, **k):
            raise CanvasError("cannot reach it")

        monkeypatch.setattr(Canvas, "deadlines", boom)
        monkeypatch.setenv("CANVAS_API_TOKEN", "1773~" + "a" * 40)
        vault = self._vault(tmp_path, age_hours=99)

        note = await refresh(self._settings(tmp_path), vault)
        assert note and "stale" in note
        assert "old thing" in vault.deadlines_path().read_text(), "kept what it had"

    async def test_an_unexpected_error_is_not_fatal_either(self, tmp_path, monkeypatch):
        from evie.sources.canvas import refresh

        async def boom(self, *a, **k):
            raise RuntimeError("something nobody predicted")

        monkeypatch.setattr(Canvas, "deadlines", boom)
        monkeypatch.setenv("CANVAS_API_TOKEN", "1773~" + "a" * 40)
        note = await refresh(self._settings(tmp_path), self._vault(tmp_path, 99))
        assert note and "saved copy" in note

    async def test_zero_hours_turns_it_off(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(Canvas, "deadlines", lambda *a, **k: called.append(1))
        from evie.sources.canvas import refresh

        vault = self._vault(tmp_path, age_hours=99)
        assert await refresh(self._settings(tmp_path, refresh_hours=0), vault) is None
        assert not called

    async def test_no_canvas_configured_does_nothing(self, tmp_path):
        from evie.config import Settings
        from evie.sources.canvas import refresh

        # No base_url: Canvas.from_settings would raise, and a first run with
        # no Canvas at all must not print an error about it.
        note = await refresh(Settings(vault=tmp_path / "v"), self._vault(tmp_path, 99))
        assert note is None or "reach" in note
