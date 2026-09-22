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


class TestGitBackup:
    """A vault on one laptop is one spilled drink from gone.

    Git gives free versioned backup to a private repo and keeps everything
    plain files, so Obsidian and agentic brains work on it unchanged.
    """

    def _vault_and_remote(self, tmp_path):
        import subprocess

        from evie.memory import Vault, VaultGit

        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        vault = Vault(tmp_path / "vault").ensure("test")
        return vault, VaultGit(vault), remote

    def test_sync_before_setup_says_so(self, tmp_path):
        from evie.memory import GitError, Vault, VaultGit

        git = VaultGit(Vault(tmp_path / "v").ensure())
        try:
            git.sync()
            raise AssertionError("should have refused")
        except GitError as exc:
            assert "not a git repo" in str(exc)

    def test_setup_then_sync_pushes_the_vault(self, tmp_path):
        import subprocess

        vault, git, remote = self._vault_and_remote(tmp_path)
        vault.log("you", "the compiler project is due Thursday")

        git.setup(str(remote))
        assert git.initialized and git.remote() == str(remote)
        assert "synced" in git.sync()

        listed = subprocess.run(
            ["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert "EVIE.md" in listed
        assert any(f.startswith("daily/") for f in listed)

    def test_a_second_sync_with_no_changes_is_a_no_op(self, tmp_path):
        vault, git, remote = self._vault_and_remote(tmp_path)
        git.setup(str(remote))
        git.sync()
        assert "up to date" in git.sync()

    def test_setup_is_repeatable(self, tmp_path):
        vault, git, remote = self._vault_and_remote(tmp_path)
        git.setup(str(remote))
        git.setup(str(remote))  # must not fail on an existing remote
        assert git.remote() == str(remote)


class TestStats:
    def test_it_counts_what_is_there(self, tmp_path):
        from evie.memory import Vault, stats

        vault = Vault(tmp_path / "v").ensure()
        vault.log("you", "something worth keeping")
        vault.note("classes", "CS 320")

        info = stats(vault)
        assert info["files"] >= 3        # EVIE.md, a daily log, a class note
        assert info["days"] == 1
        assert info["bytes"] > 0

    def test_an_empty_vault_reports_cleanly(self, tmp_path):
        from evie.memory import Vault, stats

        info = stats(Vault(tmp_path / "v").ensure())
        assert info["days"] == 0 and info["first"] is None


class TestRemoteValidation:
    """Git's error for a malformed remote explains nothing, and the two valid
    shapes splice together into something that looks plausible."""

    def test_the_spliced_url_is_caught_and_explained(self):
        from evie.memory import check_remote

        problem = check_remote("git@github.com:https://github.com/TripleB384/evie-vault")
        assert problem is not None
        # Must show both correct forms with the user's own path filled in.
        assert "https://github.com/TripleB384/evie-vault.git" in problem
        assert "git@github.com:TripleB384/evie-vault.git" in problem

    def test_both_real_forms_pass(self):
        from evie.memory import check_remote

        assert check_remote("https://github.com/you/evie-vault.git") is None
        assert check_remote("git@github.com:you/evie-vault.git") is None
        assert check_remote("ssh://git@github.com/you/evie-vault.git") is None

    def test_a_link_to_a_page_inside_the_repo_is_rejected(self):
        from evie.memory import check_remote

        assert check_remote("https://github.com/you/repo/tree/main/notes") is not None

    def test_nonsense_is_rejected_with_examples(self):
        from evie.memory import check_remote

        problem = check_remote("my github repo")
        assert problem and "evie-vault.git" in problem

    def test_setup_refuses_a_bad_remote(self, tmp_path):
        from evie.memory import GitError, Vault, VaultGit

        git = VaultGit(Vault(tmp_path / "v").ensure())
        try:
            git.setup("git@github.com:https://github.com/you/repo")
            raise AssertionError("should have refused")
        except GitError as exc:
            assert "SSH prefix" in str(exc)


class TestBriefing:
    """What she knows without opening a file.

    The vault was write-only until this existed: Canvas sync wrote sixteen
    real deadlines into classes/upcoming.md and "what's due this week" still
    got "I'm not sure what's on your calendar yet", because nothing ever read
    the file back.
    """

    import datetime as _dt

    def _stocked(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Isaac")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").write_text(
            "# Upcoming\n\n## Coming up\n\n"
            "- [ ] **Thu 25 Sep, 11:59 pm** — Directed Writing draft (AICE ENG LANG)\n"
        )
        vault.log("you", "the compiler project is the big one")
        return vault

    def test_it_leads_with_todays_date(self, tmp_path):
        """A model has no clock, and "this week" cannot be resolved without
        one. Every deadline answer is wrong otherwise."""
        block = self._stocked(tmp_path).briefing()
        assert block.startswith("Today is ")
        assert f"{self._dt.date.today():%Y}" in block

    def test_it_carries_the_deadlines(self, tmp_path):
        block = self._stocked(tmp_path).briefing()
        assert "Directed Writing draft" in block
        assert "Thu 25 Sep" in block

    def test_it_carries_the_recent_log(self, tmp_path):
        assert "compiler project" in self._stocked(tmp_path).briefing()

    def test_hand_written_notes_come_along(self, tmp_path):
        """Notes added outside the sync markers are context too."""
        vault = self._stocked(tmp_path)
        path = vault.root / "classes" / "upcoming.md"
        path.write_text(path.read_text() + "\n## Mine\n\nSigler moved the rubric.\n")
        assert "Sigler moved the rubric" in vault.briefing()

    def test_an_empty_vault_is_just_the_date(self, tmp_path):
        block = Vault(tmp_path / "empty").briefing()
        assert block.startswith("Today is")
        assert "What is due" not in block

    def test_no_canvas_sync_yet_is_not_an_error(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Isaac")
        block = vault.briefing()
        assert "Today is" in block and "What is due" not in block

    def test_it_stays_under_the_cap(self, tmp_path):
        """This rides on every request. An uncapped vault would quietly
        inflate the cost of saying hello."""
        vault = Vault(tmp_path / "v").ensure("Isaac")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").write_text(
            "\n".join(f"- [ ] item number {i}" for i in range(4000))
        )
        block = vault.briefing()
        assert len(block) <= vault.BRIEFING_CAP + 80, len(block)
        assert "trimmed" in block, "it should say it was cut, not end mid-thought"

    def test_trimming_keeps_the_head(self, tmp_path):
        """Cutting from the front would lose today's date and the soonest
        deadlines — the two things most likely to be asked about."""
        vault = Vault(tmp_path / "v").ensure("Isaac")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").write_text(
            "- [ ] DUE FIRST\n" + "\n".join(f"- [ ] filler {i}" for i in range(4000))
        )
        block = vault.briefing()
        assert "Today is" in block
        assert "DUE FIRST" in block

    def test_an_unreadable_file_does_not_take_the_turn_down(self, tmp_path):
        vault = Vault(tmp_path / "v").ensure("Isaac")
        classes = vault.root / "classes"
        classes.mkdir(parents=True, exist_ok=True)
        (classes / "upcoming.md").mkdir()   # a directory where a file should be
        assert "Today is" in vault.briefing()
