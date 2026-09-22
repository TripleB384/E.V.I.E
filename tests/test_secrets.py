"""Getting a credential from a person without it touching a command line.

Two keys leaked during this project, both the same way, and neither was
carelessness -- the documented way to set one put it in `argv`:

    echo 'export CANVAS_API_TOKEN=1773~...' >> ~/.zshrc

That is in shell history, visible in `ps`, and sitting in the scrollback to
be copied into a chat window. The second leak happened while recovering from
an unterminated quote in that exact command.

So these tests are mostly about what must *never* happen: no token-shaped
argument, no echo, no value in an error.
"""

from pathlib import Path

import pytest

from evie.secrets import (
    MARK_END,
    MARK_START,
    manual_instructions,
    shell_rc,
    store,
    why_not,
)

REAL = "1773~" + "aB3xQ9zK7m" * 6   # the right shape, not a real token


class TestRefusingABadPaste:
    """A paste that silently succeeds is worse than an error: it surfaces
    later as an auth failure nobody connects back to this."""

    @pytest.mark.parametrize("value,hint", [
        ("", "nothing"),
        ("   ", "nothing"),
        ("your_token_here", "placeholder"),
        ("<token>", "placeholder"),
        ("short", "too short"),
        (f" {REAL} ", "whitespace"),
        (f"{REAL}\t", "whitespace"),
    ])
    def test_it_says_what_is_wrong(self, value, hint):
        reason = why_not(value)
        assert reason and hint in reason

    def test_a_pasted_command_fragment_is_caught(self):
        """The `quote>` recovery paste looks like this, and writing it would
        produce a file that breaks every new shell."""
        reason = why_not("export CANVAS_API_TOKEN='1773~abcdefghijklmnop")
        assert reason and "quote" in reason

    def test_a_plausible_token_passes(self):
        assert why_not(REAL) is None

    @pytest.mark.parametrize("value", [
        f" {REAL} ",                                   # whitespace around a real one
        f"export CANVAS_API_TOKEN='{REAL}",            # the quote> recovery paste
        "1773~abc",                                    # too short, but token-shaped
    ])
    def test_the_reason_never_quotes_the_value_back(self, value):
        """An error message is exactly the moment someone pastes their whole
        terminal at you, so it must not carry the secret into the paste."""
        reason = why_not(value)
        assert reason, "these should all be refused"
        assert REAL not in reason
        assert value.strip() not in reason
        assert "1773~" not in reason


class TestWritingItDown:
    def test_it_writes_an_export_inside_a_marked_block(self, tmp_path):
        rc = tmp_path / ".zshrc"
        store("CANVAS_API_TOKEN", REAL, rc)
        body = rc.read_text()
        assert f"export CANVAS_API_TOKEN={REAL}" in body
        assert MARK_START in body and MARK_END in body

    def test_rerunning_replaces_rather_than_stacks(self, tmp_path):
        """Two exports of the same variable is a bug that hides itself: the
        last one wins and the first looks like it worked."""
        rc = tmp_path / ".zshrc"
        store("CANVAS_API_TOKEN", REAL, rc)
        store("CANVAS_API_TOKEN", "9999~" + "z" * 40, rc)
        body = rc.read_text()
        assert body.count("export CANVAS_API_TOKEN=") == 1
        assert REAL not in body, "the old value must be gone"
        assert body.count(MARK_START) == 1

    def test_a_second_variable_joins_the_same_block(self, tmp_path):
        rc = tmp_path / ".zshrc"
        store("CANVAS_API_TOKEN", REAL, rc)
        store("GROQ_API_KEY", "gsk_" + "a" * 40, rc)
        body = rc.read_text()
        assert body.count(MARK_START) == 1
        assert "CANVAS_API_TOKEN" in body and "GROQ_API_KEY" in body

    def test_it_never_touches_what_was_already_in_the_file(self, tmp_path):
        rc = tmp_path / ".zshrc"
        rc.write_text("# my shell\nalias gs='git status'\nexport PATH=$PATH:/opt\n")
        store("CANVAS_API_TOKEN", REAL, rc)
        body = rc.read_text()
        assert "alias gs='git status'" in body
        assert "export PATH=$PATH:/opt" in body

    def test_it_appends_rather_than_prepends(self, tmp_path):
        """An rc file is read top to bottom and people put their PATH setup
        first for a reason."""
        rc = tmp_path / ".zshrc"
        rc.write_text("export PATH=/usr/local/bin:$PATH\n")
        store("CANVAS_API_TOKEN", REAL, rc)
        body = rc.read_text()
        assert body.index("PATH") < body.index(MARK_START)

    def test_fish_gets_fish_syntax(self, tmp_path):
        rc = tmp_path / "config.fish"
        store("CANVAS_API_TOKEN", REAL, rc)
        assert f"set -gx CANVAS_API_TOKEN {REAL}" in rc.read_text()
        assert "export" not in rc.read_text()

    def test_a_file_we_create_is_not_world_readable(self, tmp_path):
        rc = tmp_path / ".zshrc"
        store("CANVAS_API_TOKEN", REAL, rc)
        assert rc.stat().st_mode & 0o077 == 0

    def test_a_file_that_already_existed_keeps_its_mode(self, tmp_path):
        """Tightening the permissions of someone's existing rc file is not
        ours to decide, and would be a surprising side effect."""
        rc = tmp_path / ".zshrc"
        rc.write_text("# mine\n")
        rc.chmod(0o644)
        store("CANVAS_API_TOKEN", REAL, rc)
        assert rc.stat().st_mode & 0o777 == 0o644

    def test_it_refuses_a_bad_value_rather_than_writing_it(self, tmp_path):
        rc = tmp_path / ".zshrc"
        with pytest.raises(ValueError):
            store("CANVAS_API_TOKEN", "your_token_here", rc)
        assert not rc.exists(), "nothing should have been written"


class TestPickingTheRightFile:
    """Writing to the wrong rc file is the quiet failure: it succeeds, reports
    success, and the variable is still unset in every terminal they open."""

    @pytest.mark.parametrize("shell,expected", [
        ("/bin/zsh", ".zshrc"),
        ("/usr/bin/zsh", ".zshrc"),
        ("/bin/bash", ".bashrc"),
        ("/bin/sh", ".bashrc"),
    ])
    def test_it_knows_the_common_shells(self, monkeypatch, shell, expected):
        monkeypatch.setenv("SHELL", shell)
        assert shell_rc().name == expected

    def test_fish_goes_to_its_own_config(self, monkeypatch):
        monkeypatch.setenv("SHELL", "/opt/homebrew/bin/fish")
        assert shell_rc() == Path.home() / ".config" / "fish" / "config.fish"

    @pytest.mark.parametrize("shell", ["", "/bin/nushell", "/usr/bin/elvish"])
    def test_an_unknown_shell_is_none_rather_than_a_guess(self, monkeypatch, shell):
        monkeypatch.setenv("SHELL", shell)
        assert shell_rc() is None


class TestTheInstructionsDoNotReintroduceTheProblem:
    def test_they_contain_no_pasteable_export_line(self):
        """Printing `export KEY=value` is how the value gets back onto a
        command line, which is the entire thing this module prevents."""
        text = manual_instructions("CANVAS_API_TOKEN", Path("/home/x/.zshrc"))
        assert "export CANVAS_API_TOKEN=" not in text
        assert "echo" not in text
        assert "CANVAS_API_TOKEN" in text, "it still has to say which variable"

    def test_they_name_the_file_when_we_know_it(self):
        assert "/home/x/.zshrc" in manual_instructions("K", Path("/home/x/.zshrc"))

    def test_they_stay_useful_when_we_do_not(self):
        text = manual_instructions("K", None)
        assert "startup file" in text


class TestTheCliOffersNoWayToPassASecret:
    """The convenience someone adds later, that quietly undoes all of this."""

    def test_canvas_setup_has_no_token_option(self):
        from evie.cli import canvas_setup

        names = {opt.name for opt in canvas_setup.params}
        assert not (names & {"token", "key", "secret", "password"}), (
            f"a secret passed as an argument lands in shell history: {names}"
        )

    def test_no_command_anywhere_takes_a_token_argument(self):
        from evie.cli import main

        offenders = []
        for name, cmd in main.commands.items():
            for sub in ([cmd] if not hasattr(cmd, "commands")
                        else list(cmd.commands.values())):
                for opt in sub.params:
                    if opt.name in ("token", "api_key", "secret", "password"):
                        offenders.append(f"{name} {sub.name} --{opt.name}")
        assert not offenders, "secrets must never come from argv: " + ", ".join(offenders)
