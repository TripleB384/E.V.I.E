"""Config is the thing users edit, so its errors have to be legible.

A typo in brains.yaml should say what is wrong with which brain, not fail
somewhere deep in a subprocess call an hour later.
"""

import pytest
import yaml

from evie.brains import CliBrain, EchoBrain, OpenAICompatBrain
from evie.config import ConfigError, Settings, build_brain, load_registry


def write(tmp_path, data):
    path = tmp_path / "brains.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestBuildBrain:
    def test_builds_a_cli_brain(self):
        brain = build_brain("claude", {"kind": "cli", "command": ["claude", "-p", "{prompt}"]})
        assert isinstance(brain, CliBrain) and brain.agentic

    def test_builds_an_http_brain(self):
        brain = build_brain("groq", {"kind": "openai", "base_url": "http://x/v1", "model": "m"})
        assert isinstance(brain, OpenAICompatBrain) and not brain.agentic

    def test_builds_an_echo_brain(self):
        assert isinstance(build_brain("echo", {"kind": "echo"}), EchoBrain)

    def test_cli_needs_a_prompt_slot(self):
        # Without {prompt} the command would run with no input and hang.
        with pytest.raises(ConfigError, match="prompt"):
            build_brain("bad", {"kind": "cli", "command": ["claude", "-p"]})

    def test_cli_needs_a_command(self):
        with pytest.raises(ConfigError, match="command"):
            build_brain("bad", {"kind": "cli"})

    def test_http_needs_base_url_and_model(self):
        with pytest.raises(ConfigError, match="base_url"):
            build_brain("bad", {"kind": "openai", "model": "m"})
        with pytest.raises(ConfigError, match="model"):
            build_brain("bad", {"kind": "openai", "base_url": "http://x"})

    def test_unknown_kind_names_the_options(self):
        with pytest.raises(ConfigError, match="cli, openai or echo"):
            build_brain("bad", {"kind": "telepathy"})


class TestLoadRegistry:
    def test_loads_and_wires_aliases(self, tmp_path):
        path = write(tmp_path, {
            "default": "a",
            "quick": "b",
            "fallback": ["a", "b"],
            "brains": {
                "a": {"kind": "echo", "aliases": ["first"]},
                "b": {"kind": "echo", "daily_limit": 500},
            },
        })
        reg = load_registry(path)
        assert reg.active == "a"
        assert reg.quick == "b"
        assert reg.fallback == ["a", "b"]
        assert reg.resolve("first") == "a"
        assert reg.headroom("b") == "0/500"

    def test_disabled_brains_are_skipped(self, tmp_path):
        path = write(tmp_path, {
            "default": "a",
            "brains": {"a": {"kind": "echo"}, "b": {"kind": "echo", "enabled": False}},
        })
        assert load_registry(path).names() == ["a"]

    def test_fallback_entries_that_are_disabled_are_dropped(self, tmp_path):
        path = write(tmp_path, {
            "default": "a",
            "fallback": ["a", "gone"],
            "brains": {"a": {"kind": "echo"}},
        })
        assert load_registry(path).fallback == ["a"]

    def test_default_pointing_at_a_disabled_brain_is_an_error(self, tmp_path):
        path = write(tmp_path, {
            "default": "b",
            "brains": {"a": {"kind": "echo"}, "b": {"kind": "echo", "enabled": False}},
        })
        with pytest.raises(ConfigError, match="default"):
            load_registry(path)

    def test_empty_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="no brains"):
            load_registry(write(tmp_path, {"default": "a"}))

    def test_every_brain_disabled_is_an_error(self, tmp_path):
        path = write(tmp_path, {"brains": {"a": {"kind": "echo", "enabled": False}}})
        with pytest.raises(ConfigError, match="disabled"):
            load_registry(path)


class TestShippedDefaults:
    def test_defaults_ship_everything_init_copies(self):
        """`evie init` copies these out of the package. A .gitignore pattern
        once excluded config.yaml from the repo, which broke a fresh clone
        without breaking anything locally."""
        from evie.config import PACKAGE_DEFAULTS

        for name in ("brains.yaml", "config.yaml", "EVIE.md"):
            assert (PACKAGE_DEFAULTS / name).is_file(), f"missing default: {name}"

    def test_echo_works_out_of_the_box(self):
        """The zero-credential smoke test has to work with zero setup.

        It shipped disabled, so the first command in the setup instructions
        -- the one that proves your install before any account is involved --
        failed with "no brain called 'echo'".
        """
        from evie.config import PACKAGE_DEFAULTS

        reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
        assert "echo" in reg.names()
        assert reg.resolve("echo") == "echo"
        # ...but it must never be a fallback: silently answering "You said: x"
        # instead of a real reply would look like a working assistant.
        assert "echo" not in reg.fallback
        assert reg.active != "echo"

    def test_an_unknown_brain_names_the_alternatives(self):
        from evie.brains.registry import UnknownBrain
        from evie.config import PACKAGE_DEFAULTS

        reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
        with pytest.raises(UnknownBrain) as exc:
            reg.resolve("clod")
        message = str(exc.value)
        assert "clod" in message and "claude" in message
        assert not message.startswith("\'"), "KeyError repr quoting leaked through"

    def test_the_bundled_brains_yaml_actually_loads(self):
        """The file every new user starts from must not be broken."""
        from evie.config import PACKAGE_DEFAULTS

        reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
        assert reg.active == "claude"
        assert reg.resolve("google") == "gemini_cli"
        assert reg.get("claude").agentic, "a CLI brain must be able to use tools"
        assert not reg.get("groq").agentic, "an HTTP brain has no hands"
        # Everything the fallback chain names must actually be loadable, or
        # the chain silently gets shorter than it looks.
        assert all(n in reg.names() for n in reg.fallback)

    def test_disabled_brains_contribute_no_aliases(self):
        """Otherwise "switch to local" resolves to a brain that isn't there."""
        from evie.config import PACKAGE_DEFAULTS

        reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
        for alias in ("local", "offline", "copilot", "openrouter"):
            if alias not in reg:
                continue
            assert reg.resolve(alias) in reg.names()

    def test_the_bundled_config_yaml_loads(self):
        from evie.config import PACKAGE_DEFAULTS

        settings = Settings.load(PACKAGE_DEFAULTS / "config.yaml")
        assert settings.hotkey == "alt_r"
        assert settings.voice.tts == "kokoro"
        assert settings.ears.model == "small.en"

    def test_settings_fall_back_to_defaults_when_nothing_is_configured(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("EVIE_HOME", str(tmp_path / "nope"))
        monkeypatch.setattr("evie.config.PACKAGE_DEFAULTS", tmp_path / "nope")
        assert Settings.load().hotkey == "alt_r"

    def test_a_named_config_that_is_missing_says_so(self, tmp_path):
        # Silently ignoring a path the user typed would hide their typo.
        with pytest.raises(ConfigError, match="not found"):
            Settings.load(tmp_path / "typo.yaml")


class TestRepoHygiene:
    """Checks on what the repository ships, not on what the code does.

    A stale root brains.yaml shadowed the package defaults for every clone
    while every other test passed -- because the defaults were correct the
    whole time; they just were not what got loaded. Catching that means
    looking at what git tracks.
    """

    def _tracked(self) -> set[str]:
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        )
        return set(out.stdout.split())

    @pytest.mark.parametrize("path", ["brains.yaml", "config.yaml"])
    def test_root_config_is_not_shipped(self, path):
        tracked = self._tracked()
        assert path not in tracked, (
            f"{path} is tracked at the repo root. Config resolves ./{path} before "
            f"the package defaults, so shipping it freezes every clone's config at "
            f"whatever this file said. Run: git rm --cached {path}"
        )

    # Shapes of the credentials this project plausibly touches. Matching the
    # prefix plus a run of key characters keeps `api_key_env: GROQ_API_KEY`
    # and prose like "sk-ant-..." from tripping it.
    SECRET_SHAPES = (
        r"gsk_[A-Za-z0-9]{20,}",        # Groq
        r"sk-ant-[A-Za-z0-9_-]{20,}",   # Anthropic
        r"sk-[A-Za-z0-9]{32,}",         # OpenAI
        r"ghp_[A-Za-z0-9]{20,}",        # GitHub
        r"AIza[A-Za-z0-9_-]{30,}",      # Google
        r"xi-api-key:\s*[A-Za-z0-9]{20,}",  # ElevenLabs
    )

    def test_no_tracked_file_contains_a_credential(self):
        """This repo is public and its users paste API keys into shells.

        Keys live in the environment and nothing writes them to disk, but that
        is a property of today's code, not a guarantee about tomorrow's. Check
        what is actually tracked.
        """
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        pattern = re.compile("|".join(self.SECRET_SHAPES))
        offenders = []
        for rel in sorted(self._tracked()):
            path = root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if match := pattern.search(text):
                # Report the location and the shape, never the value.
                offenders.append(f"{rel} (matched {match.re.pattern[:12]}…)")
        assert not offenders, "credential-shaped strings in tracked files: " + "; ".join(
            offenders
        )

    def test_the_scanner_would_catch_a_real_key(self):
        """A scanner nobody has seen fail is not a scanner."""
        import re

        pattern = re.compile("|".join(self.SECRET_SHAPES))
        assert pattern.search("GROQ_API_KEY=gsk_" + "a1B2c3D4e5F6g7H8i9J0k1L2")
        assert pattern.search("token: ghp_" + "0123456789abcdefghijABCD")
        # ...and would not fire on the config that names variables.
        assert not pattern.search("api_key_env: GROQ_API_KEY")
        assert not pattern.search("export GROQ_API_KEY=your_key_here")

    def test_env_files_are_ignored(self):
        from pathlib import Path

        rules = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
        for rule in (".env", "*.env"):
            assert rule in rules, f"{rule} must be gitignored in a public repo"

    def test_the_defaults_that_should_ship_do(self):
        tracked = self._tracked()
        for name in ("brains.yaml", "config.yaml", "EVIE.md"):
            assert f"evie/defaults/{name}" in tracked
