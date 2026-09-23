"""Push-to-talk plumbing that does not need a microphone.

`PushToTalk` imports sounddevice and pynput inside `__enter__`, so the queue
and lifecycle logic is testable anywhere -- which is where the shutdown bug
lived.
"""

import threading
import time

import pytest

from evie.audio.capture import PushToTalk, accessibility_trusted


class TestInterruptibleReads:
    """A mic read that never returns stops the whole process from exiting.

    `next` on a plain blocking `queue.get()` ran in an executor thread, and
    Python joins those threads at interpreter exit. Ctrl-C needed two presses
    and still printed a KeyboardInterrupt traceback out of
    threading._shutdown.
    """

    def test_a_read_with_nothing_waiting_returns(self):
        started = time.monotonic()
        assert PushToTalk().next_utterance(timeout=0.05) is None
        assert time.monotonic() - started < 1.0, "must not block indefinitely"

    def test_a_read_returns_queued_audio(self):
        mic = PushToTalk()
        mic._utterances.put("audio")
        assert mic.next_utterance(timeout=0.05) == "audio"

    def test_the_generator_stops_once_closed(self):
        mic = PushToTalk()
        mic._utterances.put("one")

        collected = []

        def drain():
            for item in mic.utterances():
                collected.append(item)

        worker = threading.Thread(target=drain, daemon=True)
        worker.start()
        time.sleep(0.1)
        mic._closed.set()
        worker.join(timeout=3)

        assert not worker.is_alive(), "the capture loop must end when closed"
        assert collected == ["one"]


class TestAccessibilityProbe:
    def test_it_answers_or_declines_to_guess(self):
        # None where the question doesn't apply (non-macOS, or pyobjc absent);
        # a bool where it does. Never a crash, since it runs during startup.
        assert accessibility_trusted() in (True, False, None)

    def test_non_macos_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        assert accessibility_trusted() is None


class TestWarmUp:
    """Loading a model is not warming it. Both Whisper and Kokoro defer real
    work to the first call, so the first answer of every session was ~3.5s
    slower than every answer after it — which reads as "she is slow" rather
    than "the model is starting"."""

    def test_the_loop_warms_every_part_before_saying_ready(self):
        from evie.loop import VoiceLoop

        warmed = []

        class Part:
            def __init__(self, name):
                self.name = name

            def warm(self):
                warmed.append(self.name)

        loop = VoiceLoop.__new__(VoiceLoop)
        loop.ears, loop.tts = Part("ears"), Part("tts")
        loop._warm_up()
        assert warmed == ["ears", "tts"]

    def test_a_part_with_no_warm_is_skipped(self):
        """`say` and `null` have nothing to warm, and must not crash startup."""
        from evie.loop import VoiceLoop

        loop = VoiceLoop.__new__(VoiceLoop)
        loop.ears = loop.tts = object()
        loop._warm_up()   # must not raise

    def test_kokoro_warm_survives_a_broken_engine(self, monkeypatch):
        """A warm-up is an optimisation. Failing one must never stop a
        session that would otherwise work."""
        from evie.voice.kokoro import KokoroEngine

        engine = KokoroEngine.__new__(KokoroEngine)
        monkeypatch.setattr(
            KokoroEngine, "synth",
            lambda self, text: (_ for _ in ()).throw(RuntimeError("onnx said no")),
        )
        engine.warm()   # must not raise

    def test_whisper_warms_with_audible_noise_not_silence(self, monkeypatch):
        """`transcribe` returns early on anything below the audible
        threshold, so zeros would never reach the model and warm nothing.

        numpy is a voice extra, and the whole point of the core install is
        that it does not need one -- so this skips rather than fails where it
        is absent, which is exactly what CI runs.
        """
        np = pytest.importorskip("numpy")

        from evie.ears.stt import Ears

        ears = Ears.__new__(Ears)
        seen = {}

        def capture(self, samples, sample_rate=16000):
            seen["peak"] = float(np.abs(np.asarray(samples)).max())
            return ""

        monkeypatch.setattr(Ears, "transcribe", capture)
        ears.warm()
        assert seen["peak"] > 1e-4, "silence would not reach the model"


class TestTheDebugLineStillShowsTheWait:
    """Adding a spoken acknowledgement before a long job nearly destroyed the
    thing that measured it.

    `first_token` is set on the first chunk of *any* kind, so with "Running
    the weekly deadline sweep" in front of it, a run where the brain took 38
    seconds reported `first token 1.70s`. The number that mattered vanished
    behind the fix for the problem it described.
    """

    def test_an_ordinary_turn_reads_as_it_always_did(self):
        from evie.loop import Timings

        line = Timings(
            stt=1.63, first_token=2.80, brain=2.80, first_audio=4.97, total=55.11
        ).render()
        assert "brain" not in line, "it repeats first token, so it is noise"
        assert line.startswith("stt 1.63s · first token 2.80s · first audio")

    def test_speaking_first_surfaces_the_real_latency(self):
        from evie.loop import Timings

        line = Timings(
            stt=1.63, first_token=1.70, brain=38.20, first_audio=2.52, total=57.85
        ).render()
        assert "brain 38.20s" in line
        assert "first token 1.70s" in line, "both, or you cannot see the gap"

    def test_a_turn_that_produced_nothing_does_not_invent_a_number(self):
        from evie.loop import Timings

        assert "brain" not in Timings().render()
