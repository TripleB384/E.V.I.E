"""A voice that makes no sound. Lets the full loop run in CI and containers."""

from __future__ import annotations


class NullEngine:
    name = "null"
    sample_rate = 24000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synth(self, text: str):
        import numpy as np

        self.spoken.append(text)
        return np.zeros(0, dtype="float32")

    def close(self) -> None:
        pass
