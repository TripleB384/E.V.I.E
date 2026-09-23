"""Kokoro-82M, running locally.

Free, unlimited, offline, and roughly ten times faster than realtime on Apple
Silicon -- which is why it is the default rather than a hosted voice. The model
is about 350MB and downloads once.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import TTSError

MODEL_DIR = Path(os.environ.get("EVIE_HOME", Path.home() / ".evie")) / "models" / "kokoro"
MODEL_FILE = MODEL_DIR / "kokoro-v1.0.onnx"
VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"

# Kept in one place so `evie doctor` can print them without importing onnx.
DOWNLOADS = {
    MODEL_FILE: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    VOICES_FILE: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def missing_files() -> list[Path]:
    return [p for p in DOWNLOADS if not p.is_file()]


class KokoroEngine:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, voice: str = "af_heart", speed: float = 1.0) -> None:
        if missing := missing_files():
            raise TTSError(
                "Kokoro model files are missing: "
                + ", ".join(p.name for p in missing)
                + ". Run `evie doctor --fix` to download them."
            )
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise TTSError(
                "kokoro-onnx is not installed. Run: uv pip install -e '.[voice]'"
            ) from exc

        self.voice = voice
        self.speed = speed
        self._kokoro = Kokoro(str(MODEL_FILE), str(VOICES_FILE))

    def synth(self, text: str):
        import numpy as np

        if not text.strip():
            return np.zeros(0, dtype="float32")
        samples, rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        self.sample_rate = rate
        return np.asarray(samples, dtype="float32")

    def warm(self) -> None:
        """Pay the first-inference cost before she claims to be ready.

        ONNX Runtime defers graph optimisation and memory allocation to the
        first `create()`, not to loading the model. Measured on an M-series
        Mac: the first thing she ever said took 3.97s from first token to
        first audio, and every reply after it took 0.6-1.0s. Nothing was
        wrong -- the cost is real and unavoidable, it was just being paid at
        the single worst moment, in the middle of answering.
        """
        try:
            self.synth("ready")
        except Exception:  # noqa: BLE001 - a warm-up must never stop startup
            pass

    def close(self) -> None:
        self._kokoro = None
