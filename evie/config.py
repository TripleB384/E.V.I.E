"""Loading brains.yaml and config.yaml into live objects.

Config resolution, nearest wins:

    ./brains.yaml                 (project, for development)
    ~/.evie/brains.yaml           (yours)
    <package>/defaults/brains.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .brains import (
    Brain,
    BrainRegistry,
    CliBrain,
    CliBrainSpec,
    EchoBrain,
    HttpBrainSpec,
    OpenAICompatBrain,
)

PACKAGE_DEFAULTS = Path(__file__).parent / "defaults"


def user_dir() -> Path:
    """Where your own config lives.

    Read on every call rather than captured at import. A module-level constant
    froze $EVIE_HOME to whatever it was when the package was first imported,
    which is invisible in normal use -- the environment is set before launch --
    and makes the whole layer untestable in-process.
    """
    return Path(os.environ.get("EVIE_HOME", Path.home() / ".evie"))


class ConfigError(Exception):
    pass


def _search_path(filename: str) -> list[Path]:
    return [Path.cwd() / filename, user_dir() / filename, PACKAGE_DEFAULTS / filename]


def config_layers(filename: str) -> list[Path]:
    """Every config file that applies, least specific first.

    Config used to be winner-takes-all: the nearest file won and the rest were
    ignored. `evie init` copies the shipped defaults into ~/.evie, which froze
    a snapshot the day you ran it -- new brains and changed defaults could
    never reach you, silently. Setting GEMINI_API_KEY did nothing, because no
    brain existed to read it, and nothing was wrong enough to report.

    Layering instead: package defaults underneath, your edits on top.
    """
    found = [p for p in reversed(_search_path(filename)) if p.is_file()]
    if not found:
        raise ConfigError(
            f"no {filename} found. Looked in: "
            + ", ".join(str(p) for p in _search_path(filename))
        )
    return found


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Overlay `over` onto `base`, one level into nested mappings.

    Deep enough for brains (per-brain keys merge, so overriding a model does
    not delete its aliases) and shallow enough to stay predictable. A list
    replaces rather than appends: someone who writes a `fallback` order means
    that order, not that order plus ours.
    """
    result = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **{
                k: ({**result[key][k], **v}
                    if isinstance(v, dict) and isinstance(result[key].get(k), dict)
                    else v)
                for k, v in value.items()
            }}
        else:
            result[key] = value
    return result


def find_config(filename: str) -> Path:
    for candidate in _search_path(filename):
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"no {filename} found. Looked in: "
        + ", ".join(str(p) for p in _search_path(filename))
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


# -- settings ------------------------------------------------------------


@dataclass
class VoiceSettings:
    tts: str = "kokoro"
    kokoro_voice: str = "af_heart"
    speed: float = 1.0
    elevenlabs_voice_id: str = ""


@dataclass
class EarSettings:
    model: str = "small.en"
    compute_type: str = "int8"
    language: str = "en"


@dataclass
class Settings:
    hotkey: str = "alt_r"
    wake_word: str = ""  # phase 5; empty means push-to-talk only
    vault: Path = field(default_factory=lambda: Path.home() / "EVIE" / "vault")
    identity_file: str = "EVIE.md"
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    ears: EarSettings = field(default_factory=EarSettings)
    transcript_turns: int = 12

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        if path is None:
            try:
                raw: dict[str, Any] = {}
                for layer in config_layers("config.yaml"):
                    raw = _merge(raw, _load_yaml(layer))
            except ConfigError:
                return cls()  # no config anywhere is fine; defaults are sane
        elif not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        else:
            raw = _load_yaml(path)
        return cls(
            hotkey=raw.get("hotkey", cls.hotkey),
            wake_word=raw.get("wake_word", ""),
            vault=Path(os.path.expanduser(raw.get("vault", "~/EVIE/vault"))),
            identity_file=raw.get("identity_file", cls.identity_file),
            transcript_turns=int(raw.get("transcript_turns", cls.transcript_turns)),
            voice=VoiceSettings(**(raw.get("voice") or {})),
            ears=EarSettings(**(raw.get("ears") or {})),
        )


# -- brains --------------------------------------------------------------


def build_brain(name: str, spec: dict[str, Any]) -> Brain:
    kind = spec.get("kind", "openai")

    if kind == "cli":
        command = spec.get("command")
        if not isinstance(command, list) or not command:
            raise ConfigError(f"brain {name!r}: 'command' must be a non-empty list")
        if not any("{prompt}" in str(a) for a in command):
            raise ConfigError(f"brain {name!r}: 'command' must contain a {{prompt}} slot")
        return CliBrain(
            CliBrainSpec(
                name=name,
                command=[str(a) for a in command],
                agentic=bool(spec.get("agentic", True)),
                parser=spec.get("parser", "plain"),
                resume_flag=spec.get("resume_flag"),
                cwd=spec.get("cwd"),
                env=spec.get("env"),
                timeout=float(spec.get("timeout", 300.0)),
                aliases=tuple(spec.get("aliases") or ()),
            )
        )

    if kind == "openai":
        for required in ("base_url", "model"):
            if not spec.get(required):
                raise ConfigError(f"brain {name!r}: missing '{required}'")
        return OpenAICompatBrain(
            HttpBrainSpec(
                name=name,
                base_url=str(spec["base_url"]),
                model=str(spec["model"]),
                api_key_env=spec.get("api_key_env"),
                agentic=bool(spec.get("agentic", False)),
                temperature=float(spec.get("temperature", 0.7)),
                max_tokens=spec.get("max_tokens"),
                timeout=float(spec.get("timeout", 120.0)),
                aliases=tuple(spec.get("aliases") or ()),
            )
        )

    if kind == "echo":
        return EchoBrain(name=name)

    raise ConfigError(f"brain {name!r}: unknown kind {kind!r} (use cli, openai or echo)")


def load_registry(path: Path | None = None) -> BrainRegistry:
    if path is not None:
        raw, sources = _load_yaml(path), [path]
    else:
        sources = config_layers("brains.yaml")
        raw = {}
        for layer in sources:
            raw = _merge(raw, _load_yaml(layer))

    specs = raw.get("brains") or {}
    if not specs:
        raise ConfigError(f"{sources[-1]} defines no brains")

    brains: dict[str, Brain] = {}
    aliases: dict[str, str] = {}
    limits: dict[str, int] = {}

    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"brain {name!r} must be a mapping")
        if spec.get("enabled") is False:
            continue
        brains[name] = build_brain(name, spec)
        for alias in spec.get("aliases") or ():
            aliases[str(alias)] = name
        if limit := spec.get("daily_limit"):
            limits[name] = int(limit)

    if not brains:
        raise ConfigError(f"{sources[-1]}: every brain is disabled")

    default = raw.get("default") or next(iter(brains))
    if default not in brains:
        raise ConfigError(f"default brain {default!r} is not defined or is disabled")

    registry = BrainRegistry(
        brains,
        default=default,
        quick=raw.get("quick"),
        tiers=raw.get("tiers") or {},
        fallback=raw.get("fallback") or [],
        aliases=aliases,
        daily_limits=limits,
    )
    registry.sources = sources
    return registry


def load_identity(settings: Settings) -> str:
    """Read EVIE.md -- who she is and what she already knows about you."""
    for candidate in (
        settings.vault / settings.identity_file,
        user_dir() / settings.identity_file,
        Path.cwd() / settings.identity_file,
        PACKAGE_DEFAULTS / "EVIE.md",
    ):
        if candidate.is_file():
            return candidate.read_text().strip()
    return ""
