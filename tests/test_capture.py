"""Push-to-talk plumbing that does not need a microphone.

`PushToTalk` imports sounddevice and pynput inside `__enter__`, so the queue
and lifecycle logic is testable anywhere -- which is where the shutdown bug
lived.
"""

import threading
import time

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
