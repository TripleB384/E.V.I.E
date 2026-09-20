"""The text-to-speech seam.

Same idea as the brain seam: everything downstream depends on this protocol,
so swapping Kokoro for ElevenLabs is a config change. Engines return float32
mono samples rather than playing audio themselves, which keeps playback,
queueing and barge-in in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # numpy is a voice extra, not a core dependency
    import numpy as np


class TTSError(Exception):
    pass


@runtime_checkable
class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synth(self, text: str) -> "np.ndarray":
        """Render text to float32 mono samples in [-1, 1]."""
        ...

    def close(self) -> None:
        ...


def load_engine(name: str, settings) -> TTSEngine:
    """Build the configured engine, falling back rather than failing silently."""
    from .base import TTSError  # noqa: F401  (re-export for callers)

    if name == "kokoro":
        from .kokoro import KokoroEngine

        return KokoroEngine(voice=settings.kokoro_voice, speed=settings.speed)
    if name == "say":
        from .say import SayEngine

        return SayEngine(speed=settings.speed)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsEngine

        return ElevenLabsEngine(voice_id=settings.elevenlabs_voice_id)
    if name == "null":
        from .null import NullEngine

        return NullEngine()
    raise TTSError(f"unknown TTS engine {name!r} (kokoro, say, elevenlabs, null)")
