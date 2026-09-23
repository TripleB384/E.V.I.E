"""Speech to text, via faster-whisper.

`small.en` at int8 is the sweet spot on Apple Silicon: about a second for a
normal utterance, and noticeably better than `base` at names and technical
words -- which matters when half of what you say is a course code or a
product name. The model is around 500MB and downloads once.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("EVIE_HOME", Path.home() / ".evie")) / "models" / "whisper"


class STTError(Exception):
    pass


class Ears:
    def __init__(
        self,
        model: str = "small.en",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise STTError(
                "faster-whisper is not installed. Run: uv pip install -e '.[voice]'"
            ) from exc

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.language = language
        try:
            self._model = WhisperModel(
                model,
                device="cpu",  # CTranslate2 has no Metal backend; int8 CPU is fast enough
                compute_type=compute_type,
                download_root=str(MODEL_DIR),
            )
        except Exception as exc:
            # The first run downloads ~500MB from Hugging Face. A bare
            # "403 Forbidden" here tells the user nothing about what failed
            # or what to do next.
            raise STTError(
                f"could not load the {model!r} speech model: {exc}\n"
                f"  The model downloads from Hugging Face on first run "
                f"(~500MB) into {MODEL_DIR}.\n"
                f"  Check your network, or set ears.model to a model you "
                f"already have cached."
            ) from exc

    def warm(self) -> None:
        """Same first-inference cost as Kokoro, paid up front.

        Faint noise rather than silence: `transcribe` returns early on
        anything below the audible threshold, so zeros would never reach the
        model and would warm nothing.
        """
        import numpy as np

        noise = (np.random.default_rng(0).standard_normal(6400) * 0.01).astype("float32")
        try:
            self.transcribe(noise)
        except Exception:  # noqa: BLE001 - a warm-up must never stop startup
            pass

    def transcribe(self, samples, sample_rate: int = 16000) -> str:
        """Transcribe float32 mono samples. Returns '' for silence."""
        import numpy as np

        audio = np.asarray(samples, dtype="float32")
        if audio.size == 0 or float(np.abs(audio).max()) < 1e-4:
            return ""
        if sample_rate != 16000:
            raise STTError(f"expected 16kHz audio, got {sample_rate}Hz")

        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,            # greedy: the latency win is worth more than the accuracy
            vad_filter=True,        # drop the silence around a push-to-talk press
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
