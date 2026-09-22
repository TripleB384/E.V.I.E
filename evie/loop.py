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
    first_audio: float = 0.0
    total: float = 0.0
    stages: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (
            f"stt {self.stt:.2f}s · first token {self.first_token:.2f}s · "
            f"first audio {self.first_audio:.2f}s · total {self.total:.2f}s"
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

    async def _answer(self, said, speaker, timings, turn_start) -> bool:
        """Stream one answer to the speakers. Returns True if she should quit."""
        first_token_at = first_audio_at = None
        should_quit = False
        printed: list[str] = []

        async def text_stream():
            nonlocal first_token_at, should_quit
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
        timings.first_audio = (first_audio_at or turn_start) - turn_start
        return should_quit
