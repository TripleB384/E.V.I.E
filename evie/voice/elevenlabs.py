"""ElevenLabs, for when you want the showpiece voice.

Not the default on purpose: the free tier is about ten minutes of audio a
month, which real daily use exhausts in a day. Good for a demo, bad for a
daily driver -- hence Kokoro by default and this behind a config flag.
"""

from __future__ import annotations

import io
import os

import httpx

from .base import TTSError

API = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsEngine:
    name = "elevenlabs"
    sample_rate = 22050

    def __init__(self, voice_id: str, model: str = "eleven_turbo_v2_5") -> None:
        self.key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.key:
            raise TTSError("$ELEVENLABS_API_KEY is not set")
        if not voice_id:
            raise TTSError("voice.elevenlabs_voice_id is not set in config.yaml")
        self.voice_id = voice_id
        self.model = model

    def synth(self, text: str):
        import numpy as np
        import soundfile as sf

        if not text.strip():
            return np.zeros(0, dtype="float32")

        resp = httpx.post(
            f"{API}/{self.voice_id}",
            headers={"xi-api-key": self.key, "Accept": "audio/mpeg"},
            json={"text": text, "model_id": self.model},
            timeout=60.0,
        )
        if resp.status_code == 401:
            raise TTSError("ElevenLabs rejected the API key")
        if resp.status_code == 429:
            raise TTSError("ElevenLabs quota exhausted -- switch to kokoro")
        resp.raise_for_status()

        samples, rate = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
        self.sample_rate = rate
        return samples if samples.ndim == 1 else samples.mean(axis=1)

    def close(self) -> None:
        pass
