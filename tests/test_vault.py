"""Memory is a folder of markdown files you can open and fix."""

import datetime as dt

from evie.memory import Vault


class TestLayout:
    def test_ensure_creates_the_structure(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Sam")
        for sub in ("daily", "classes", "business", "people", "projects"):
            assert (vault.root / sub).is_dir()
        assert "Sam" in (vault.root / "EVIE.md").read_text()

    def test_ensure_does_not_clobber_an_existing_identity(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Sam")
        (vault.root / "EVIE.md").write_text("my own words")
        vault.ensure("Sam")
        assert vault.identity() == "my own words"

    def test_missing_vault_reports_absent(self, tmp_path):
        assert not Vault(tmp_path / "nothing").exists


class TestLogging:
    def test_log_appends_to_today(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        vault.log("you", "the project is a compiler")
        assert "compiler" in vault.today().read_text()

    def test_log_turn_records_both_sides(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        vault.log_turn("when is it due", "Thursday", "claude")
        text = vault.today().read_text()
        assert "when is it due" in text and "Thursday" in text and "claude" in text

    def test_blank_lines_are_not_logged(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        before = vault.today().read_text()
        vault.log("you", "   ")
        assert vault.today().read_text() == before

    def test_todays_file_is_dated(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        assert vault.today().name == f"{dt.date.today():%Y-%m-%d}.md"

    def test_recent_log_reads_back(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        vault.log("you", "remember the investor call")
        assert "investor call" in vault.recent_log()


class TestNotes:
    def test_note_slugifies_the_title(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        path = vault.note("classes", "CS 320: Compilers!")
        assert path.name == "cs-320-compilers.md"
        assert path.read_text().startswith("# CS 320: Compilers!")

    def test_note_is_idempotent(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure()
        vault.note("business", "pricing").write_text("# pricing\n\nten dollars")
        assert "ten dollars" in vault.note("business", "pricing").read_text()
