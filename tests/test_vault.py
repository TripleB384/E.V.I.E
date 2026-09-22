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
