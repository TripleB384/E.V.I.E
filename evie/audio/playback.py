"""Audio out, with the ability to shut up mid-sentence.

Speech is queued rather than played inline so synthesis of the next phrase
overlaps playback of the current one -- without that, you hear a gap at every
sentence boundary. `stop()` drops the queue so E.V.I.E. goes quiet the instant
you interrupt her, which matters more than it sounds: an assistant you cannot
cut off is exhausting to talk to.
"""

from __future__ import annotations

import queue
import threading


class Speaker:
    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()
        self.unavailable: str | None = None

    def __enter__(self) -> "Speaker":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            # No output device, or PortAudio missing. Drain the queue instead of
            # dying: a silent assistant is bad, a hung one is worse.
            self.unavailable = str(exc)
            sd = None

        while True:
            item = self._queue.get()
            if item is None:
                return
            samples, rate = item
            if sd is None:
                self._mark_idle()
                continue
            if self._stop.is_set() or samples is None or len(samples) == 0:
                self._mark_idle()
                continue
            try:
                sd.play(samples, rate)
                while sd.get_stream().active:
                    if self._stop.is_set():
                        sd.stop()
                        break
                    sd.sleep(20)
            except Exception:
                pass  # a dead output device must not take the assistant down
            self._mark_idle()

    def _mark_idle(self) -> None:
        if self._queue.empty():
            self._idle.set()

    def say(self, samples, rate: int | None = None) -> None:
        if samples is None or len(samples) == 0:
            return
        self._stop.clear()
        self._idle.clear()
        self._queue.put((samples, rate or self.sample_rate))

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut the current phrase."""
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._idle.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)
