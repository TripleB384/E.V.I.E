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
