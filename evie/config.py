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
USER_DIR = Path(os.environ.get("EVIE_HOME", Path.home() / ".evie"))


class ConfigError(Exception):
    pass


def _search_path(filename: str) -> list[Path]:
    return [Path.cwd() / filename, USER_DIR / filename, PACKAGE_DEFAULTS / filename]


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
                path = find_config("config.yaml")
            except ConfigError:
                return cls()  # no config anywhere is fine; defaults are sane
        elif not path.is_file():
            raise ConfigError(f"config file not found: {path}")
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
    path = path or find_config("brains.yaml")
    raw = _load_yaml(path)

    specs = raw.get("brains") or {}
    if not specs:
        raise ConfigError(f"{path} defines no brains")

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
        raise ConfigError(f"{path}: every brain is disabled")

    default = raw.get("default") or next(iter(brains))
    if default not in brains:
        raise ConfigError(f"default brain {default!r} is not defined or is disabled")

    return BrainRegistry(
        brains,
        default=default,
        quick=raw.get("quick"),
        fallback=raw.get("fallback") or [],
        aliases=aliases,
        daily_limits=limits,
    )


def load_identity(settings: Settings) -> str:
    """Read EVIE.md -- who she is and what she already knows about you."""
    for candidate in (
        settings.vault / settings.identity_file,
        USER_DIR / settings.identity_file,
        Path.cwd() / settings.identity_file,
        PACKAGE_DEFAULTS / "EVIE.md",
    ):
        if candidate.is_file():
            return candidate.read_text().strip()
    return ""
