"""macOS `say`. Robotic, but it is always there and it never fails to install.

Worth keeping as the fallback: if Kokoro's model download is broken or ONNX
misbehaves after an OS update, E.V.I.E. still talks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import TTSError


class SayEngine:
    name = "say"
    sample_rate = 22050

    def __init__(self, voice: str = "Samantha", speed: float = 1.0) -> None:
        if shutil.which("say") is None:
            raise TTSError("`say` not found -- this engine is macOS only")
        self.voice = voice
        self.words_per_minute = str(int(175 * speed))

    def synth(self, text: str):
        import numpy as np
        import soundfile as sf

        if not text.strip():
            return np.zeros(0, dtype="float32")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            subprocess.run(
                [
                    "say", "-v", self.voice, "-r", self.words_per_minute,
                    "--data-format=LEF32@22050", "-o", str(path), text,
                ],
                check=True,
                capture_output=True,
            )
            samples, rate = sf.read(path, dtype="float32", always_2d=False)

        self.sample_rate = rate
        return samples if samples.ndim == 1 else samples.mean(axis=1)

    def close(self) -> None:
        pass
