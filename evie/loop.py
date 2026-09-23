"""The voice loop: hold a key, talk, hear an answer.

The ordering here is the whole game. Synthesis starts on the first complete
sentence rather than the last, so E.V.I.E. begins speaking while the model is
still writing. That one decision is the difference between a three-second
response and a twelve-second one, and no amount of faster hardware buys it
back if you get it wrong.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from rich.console import Console

from .assistant import Assistant
from .audio import PushToTalk, Speaker
from .ears import Ears
from .text import for_speech, speakable
from .voice import load_engine


@dataclass
class Timings:
    """Per-stage latency, so a slow turn points at its own cause."""

    stt: float = 0.0
    first_token: float = 0.0
    # Time to the first chunk of the *answer*, as opposed to the first chunk of
    # anything. They differ when she speaks before the brain does -- announcing
    # a skill, or naming a fallback -- and without this the announcement hides
    # exactly what it was added to cover: a sweep reported `first token 1.7s`
    # while the brain took 38 seconds.
    brain: float = 0.0
    first_audio: float = 0.0
    total: float = 0.0
    stages: list[str] = field(default_factory=list)

    def render(self) -> str:
        # Only when something was said first; otherwise it repeats first token
        # on every ordinary turn.
        spoke_first = self.brain - self.first_token > 0.05
        return (
            f"stt {self.stt:.2f}s · first token {self.first_token:.2f}s · "
            + (f"brain {self.brain:.2f}s · " if spoke_first else "")
            + f"first audio {self.first_audio:.2f}s · total {self.total:.2f}s"
        )


class VoiceLoop:
    def __init__(self, assistant: Assistant, *, debug: bool = False) -> None:
        self.assistant = assistant
        self.debug = debug
        self.console = Console()
        settings = assistant.settings
        try:
            self.ears = Ears(
                model=settings.ears.model,
                compute_type=settings.ears.compute_type,
                language=settings.ears.language,
            )
        except Exception as exc:
            raise RuntimeError(f"speech-to-text failed to start: {exc}") from exc
        try:
            self.tts = load_engine(settings.voice.tts, settings.voice)
        except Exception as exc:
            raise RuntimeError(
                f"text-to-speech ({settings.voice.tts}) failed to start: {exc}\n"
                f"  Try `evie say \"test\" --engine say` to fall back to the "
                f"system voice."
            ) from exc

    async def run(self) -> None:
        settings = self.assistant.settings
        key = settings.hotkey

        self._warm_up()

        with Speaker(self.tts.sample_rate) as speaker, PushToTalk(key) as mic:
            self.console.print(
                f"[bold green]E.V.I.E. ready[/] — hold [bold]{key}[/] to talk, "
                f"running on [bold]{self.assistant.registry.active}[/]. Ctrl-C to quit."
            )
            loop = asyncio.get_running_loop()

            while True:
                # The mic read goes to a thread so the event loop stays free, and
                # returns empty-handed every quarter second so Ctrl-C can land.
                audio = await loop.run_in_executor(None, mic.next_utterance)
                if audio is None:
                    continue
                speaker.stop()  # pressing the key while she talks cuts her off

                turn = time.perf_counter()
                said = await loop.run_in_executor(None, self.ears.transcribe, audio)
                timings = Timings(stt=time.perf_counter() - turn)

                if not said:
                    continue
                self.console.print(f"[dim]you:[/] {said}")

                quit_after = await self._answer(said, speaker, timings, turn)
                timings.total = time.perf_counter() - turn
                if self.debug:
                    self.console.print(f"[dim]{timings.render()}[/]")
                if quit_after:
                    speaker.wait(timeout=10)
                    return

    def _warm_up(self) -> None:
        """Run one throwaway inference through each model before saying ready.

        Loading a model is not the same as warming it: both Whisper and Kokoro
        defer real work to the first call. Without this the first answer of
        every session was ~3.5s slower than every answer after it, which reads
        as "she is slow" rather than "the model is starting".

        Nothing here is allowed to fail the session -- an engine with no
        `warm` is simply skipped.
        """
        for part in (self.ears, self.tts):
            if callable(warm := getattr(part, "warm", None)):
                warm()

    async def _answer(self, said, speaker, timings, turn_start) -> bool:
        """Stream one answer to the speakers. Returns True if she should quit."""
        first_token_at = first_audio_at = brain_at = None
        should_quit = False
        printed: list[str] = []

        async def text_stream():
            nonlocal first_token_at, brain_at, should_quit
            async for kind, chunk in self.assistant.respond(said):
                if kind == "meta":
                    should_quit = should_quit or chunk == "quit"
                    if chunk == "stop":
                        speaker.stop()
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if kind == "notice":
                    self.console.print(f"[yellow]{chunk}[/]")
                else:
                    if brain_at is None:
                        brain_at = time.perf_counter()
                    printed.append(chunk)
                yield chunk

        loop = asyncio.get_running_loop()
        async for phrase in speakable(text_stream()):
            spoken = for_speech(phrase)
            if not spoken:
                continue
            samples = await loop.run_in_executor(None, self.tts.synth, spoken)
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            speaker.say(samples, self.tts.sample_rate)

        if printed:
            self.console.print(f"[bold cyan]evie:[/] {''.join(printed).strip()}")
        timings.first_token = (first_token_at or turn_start) - turn_start
        timings.brain = (brain_at or first_token_at or turn_start) - turn_start
        timings.first_audio = (first_audio_at or turn_start) - turn_start
        return should_quit
