"""The command centre, and the one thing it must never do.

A dashboard is read at a glance, from across a room, and believed. That makes
it the worst possible place to overstate what we know. `brains list` can say a
CLI is *installed*, because `shutil.which` found the binary; it cannot say the
CLI is *logged in*, because only a real call settles that. A green dot for
`claude` while its session had silently expired is exactly the failure that
cost a voice session -- so most of these tests are about a state being
reported as unproven rather than good.
"""

import datetime as _dt

import pytest

from evie import dashboard
from evie.brains import BrainRegistry, Context, Health
from evie.config import PACKAGE_DEFAULTS, Settings
from evie.memory import Vault
from tests.test_brains import FakeBrain

TEMPLATE = PACKAGE_DEFAULTS / "dashboard" / "dashboard.html"

UPCOMING = """# Upcoming

*Synced from Canvas wed 23 sep, 5:12 am.*

## Late

- [ ] **Tue 16 Sep, 11:59 pm** — Solution Planner (AICE GLBL PERSP) · 200 pts · **missing**
- [ ] **Wed 17 Sep, 11:59 pm** — Linux+ Domain 1 (C3 Cyber 3) · 300 pts

## Coming up

- [x] **Mon 22 Sep, 11:59 pm** — Genre Research Presentation (AICE MEDIASTUDIES) · 100 pts
- [ ] **Wed 23 Sep, 3:00 pm** — Genre Research Voiceover (AICE MEDIASTUDIES) · 80 pts
- [ ] **Fri 25 Sep, 11:59 pm** — Script and Visual Support (AICE GLBL PERSP) · 400 pts
"""


@pytest.fixture
def vault(tmp_path):
    v = Vault(tmp_path / "v").ensure("Isaac")
    v.deadlines_path().parent.mkdir(parents=True, exist_ok=True)
    v.deadlines_path().write_text(UPCOMING)
    return v


class Unverified(FakeBrain):
    """A CLI brain: found on PATH, never actually called."""

    async def health(self):
        from evie.brains import BrainStatus

        return BrainStatus(Health.UNVERIFIED, "installed")


def registry(**over):
    reg = BrainRegistry(
        {"claude": Unverified("claude", agentic=True),
         "groq": FakeBrain("groq"),
         "openrouter": FakeBrain("openrouter")},
        default="groq",
        **over,
    )
    reg.parked["openrouter"] = "$OPENROUTER_API_KEY is not set"
    return reg


class TestReadingWhatCanvasWrote:
    def test_it_counts_what_is_there(self, vault):
        dl = dashboard.read_deadlines(vault)
        assert (dl.total, dl.done, dl.open) == (5, 1, 4)

    def test_late_is_the_late_section_only(self, vault):
        assert dashboard.read_deadlines(vault).late == 2

    def test_missing_is_canvas_saying_so_not_us_inferring_it(self, vault):
        """Late and missing are different claims: Canvas marks the second."""
        assert dashboard.read_deadlines(vault).missing == 1

    def test_points_add_up_and_skip_the_one_handed_in(self, vault):
        assert dashboard.read_deadlines(vault).points == 980

    def test_a_submitted_item_is_not_a_priority(self, vault):
        titles = dashboard.read_deadlines(vault).titles
        assert "Genre Research Presentation" not in titles

    def test_no_vault_is_empty_not_an_error(self):
        assert dashboard.read_deadlines(None).total == 0

    def test_a_vault_with_no_sync_is_empty_not_an_error(self, tmp_path):
        assert dashboard.read_deadlines(Vault(tmp_path / "bare")).total == 0


class TestItNeverClaimsAStateItHasNotEstablished:
    async def test_an_uncalled_cli_brain_is_unverified_not_ready(self, vault):
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        claude = next(c for c in data["connectors"] if c["name"] == "claude")
        assert claude["state"] == "unverified"
        assert claude["state"] != "ready"

    async def test_a_parked_brain_says_it_is_not_set_up(self, vault):
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        row = next(c for c in data["connectors"] if c["name"] == "openrouter")
        assert row["state"] == "parked"
        assert row["used"] == "", "a brain with no key has spent nothing"

    async def test_probing_turns_unverified_into_a_fact(self, vault):
        """`--probe` is the difference between a dashboard you can read at a
        glance and one that once showed a logged-out brain green all session."""
        reg = registry()
        before = await dashboard.build(reg, vault, Settings(vault=vault.root))
        after = await dashboard.build(reg, vault, Settings(vault=vault.root), probe=True)
        state = lambda d, n: next(c for c in d["connectors"] if c["name"] == n)["state"]
        assert state(before, "claude") == "unverified"
        assert state(after, "claude") == "ready", "it answered, so it is proven"

    async def test_a_brain_that_fails_the_probe_reads_offline(self, vault):
        from evie.brains import BrainUnavailable

        reg = registry()
        reg.get("claude").fail = BrainUnavailable("Not logged in")
        data = await dashboard.build(reg, vault, Settings(vault=vault.root), probe=True)
        row = next(c for c in data["connectors"] if c["name"] == "claude")
        assert row["state"] == "offline"


class TestNoNumberIsInvented:
    async def test_every_figure_traces_to_the_vault(self, vault):
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        dl = dashboard.read_deadlines(vault)
        figures = {f["label"]: f["value"] for f in data["figures"]}
        assert figures["still to do"] == str(dl.open)
        assert figures["overdue"] == str(dl.late)
        assert figures["points at stake"] == f"{dl.points:g}"

    async def test_every_bar_length_is_a_real_proportion(self, vault):
        """A bar pinned at a constant encodes nothing, which is worse than no
        bar -- it reads as information and is not."""
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        dl = dashboard.read_deadlines(vault)
        by_key = {s["key"]: s for s in data["stats"]}
        assert by_key["handed in"]["pct"] == round(100 * dl.done / dl.total)
        assert by_key["late"]["pct"] == round(100 * dl.late / dl.total)
        assert all(0 <= s["pct"] <= 100 for s in data["stats"])

    async def test_no_canvas_sync_says_so_rather_than_showing_zeros(self, tmp_path):
        bare = Vault(tmp_path / "bare").ensure("Isaac")
        data = await dashboard.build(registry(), bare, Settings(vault=bare.root))
        assert data["stats"] == [], "no data is not the same as zero"
        assert "never been synced" in data["chip"]

    async def test_a_stale_sync_is_flagged(self, vault, monkeypatch):
        monkeypatch.setattr(Vault, "deadlines_age",
                            lambda self: _dt.timedelta(days=4))
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        assert "stale" in data["chip"]

    async def test_the_priorities_are_the_soonest_open_ones(self, vault):
        data = await dashboard.build(registry(), vault, Settings(vault=vault.root))
        assert len(data["priorities"]) <= 3
        assert "Genre Research Presentation" not in data["priorities"]

    async def test_it_survives_a_vault_it_cannot_read(self, tmp_path):
        data = await dashboard.build(registry(), None, Settings(vault=tmp_path))
        assert data["connectors"], "the brains are still knowable"
        assert data["stats"] == []


class TestTheLogPanel:
    def test_it_reads_the_daily_log_back(self, vault):
        vault.log("you", "What's due this week?")
        vault.log("groq", "The voiceover at three.")
        lines = dashboard.read_log(vault)
        assert any("you: What's due this week?" in l for l in lines)
        assert any("groq:" in l for l in lines)

    def test_a_long_line_is_cut_not_wrapped_forever(self, vault):
        vault.log("groq", "x" * 400)
        assert all(len(l) < 120 for l in dashboard.read_log(vault))

    def test_no_log_is_empty_not_an_error(self, tmp_path):
        assert dashboard.read_log(Vault(tmp_path / "bare")) == []


class TestThePageItself:
    def test_it_is_self_contained(self):
        """One file, no libraries. The only external reference is the font,
        and there is a fallback stack so it works with the network off."""
        html = TEMPLATE.read_text()
        import re

        srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
        assert srcs == ["evie_data.js"], f"unexpected scripts: {srcs}"
        hrefs = re.findall(r'<link[^>]+href="([^"]+)"', html)
        assert all("fonts.g" in h for h in hrefs), f"unexpected stylesheets: {hrefs}"

    def test_the_data_file_is_the_only_way_in(self):
        html = TEMPLATE.read_text()
        assert "window.EVIE_DATA" in html
        assert "fetch(" not in html, "the page must not go looking for data itself"

    def test_it_knows_every_state_the_builder_can_emit(self):
        html = TEMPLATE.read_text()
        for state in {"ready", "unverified", "exhausted", "offline", "parked"}:
            assert f'{state}:' in html or f'"{state}"' in html, state

    def test_status_is_never_carried_by_colour_alone(self):
        """A green and an amber dot are the same dot to a red/green colour
        blind reader, and "unverified" is not a shade of green anyway -- it is
        a different claim and needs its own word."""
        html = TEMPLATE.read_text()
        assert '<span class="st">' in html
        assert "unverified" in html and "not set up" in html

    def test_the_written_page_and_data_land_together(self, tmp_path, vault):
        data = {"greeting": "hi", "connectors": [], "stats": [], "figures": [],
                "priorities": [], "chip": "", "log": []}
        page = dashboard.write(tmp_path / "out", data, TEMPLATE)
        assert page.name == "dashboard.html"
        assert (page.parent / "evie_data.js").is_file()

    def test_the_data_file_is_a_plain_assignment(self, tmp_path):
        js = dashboard.render_data({"greeting": "hi <there>", "log": ["a"]})
        assert js.strip().endswith(";")
        assert "window.EVIE_DATA = " in js
        import json

        body = js.split("window.EVIE_DATA = ", 1)[1].rsplit(";", 1)[0]
        assert json.loads(body)["greeting"] == "hi <there>"
