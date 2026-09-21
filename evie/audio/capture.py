"""Push-to-talk capture.

Hold a key, talk, release. No wake word, no voice-activity guessing, no false
triggers -- which is why it ships first. A wake word is a phase-5 nicety on
top of this, not a replacement for it.

macOS will not deliver global key events to a process that has not been
granted Input Monitoring and Accessibility. Those permissions attach to your
*terminal application*, not to Python, and that catches everyone at least
once. `evie doctor` checks for it explicitly.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Iterator

SAMPLE_RATE = 16000  # what Whisper wants; resampling later is wasted work
BLOCK = 1024
POLL = 0.25  # how often a blocked mic read wakes up so Ctrl-C can land


class AudioError(Exception):
    pass


def accessibility_trusted() -> bool | None:
    """Is this process allowed to watch global key events?

    macOS gates that behind Accessibility, and pynput's only signal is a line
    on stderr -- which scrolls past while the app cheerfully reports itself
    ready and then ignores every key press. AXIsProcessTrusted answers it
    directly. Returns None where the question doesn't apply or can't be asked.
    """
    if sys.platform != "darwin":
        return None
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return None
    try:
        return bool(AXIsProcessTrusted())
    except Exception:
        return None


def request_accessibility() -> bool | None:
    """Ask macOS to show its Accessibility prompt for this app.

    This is the reliable way to get a terminal into the list: the system adds
    the entry itself. Hunting for the "+" button in System Settings is the
    unreliable way, and on some versions there isn't one to find.
    """
    if sys.platform != "darwin":
        return None
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions

        try:
            from ApplicationServices import kAXTrustedCheckOptionPrompt as prompt_key
        except ImportError:
            prompt_key = "AXTrustedCheckOptionPrompt"
        return bool(AXIsProcessTrustedWithOptions({prompt_key: True}))
    except Exception:
        return None


def _resolve_key(name: str):
    from pynput import keyboard

    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise AudioError(
        f"unknown hotkey {name!r}. Try one of: alt_r, alt_l, cmd_r, ctrl_r, f13"
    )


class PushToTalk:
    """Records while the hotkey is held; yields one array per utterance."""

    def __init__(self, hotkey: str = "alt_r", sample_rate: int = SAMPLE_RATE) -> None:
        self.hotkey_name = hotkey
        self.sample_rate = sample_rate
        self._utterances: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._held = threading.Event()
        self._press_count = 0
        self._listener = None
        self._stream = None
        self._frames: list = []

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "PushToTalk":
        try:
            import sounddevice as sd
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise AudioError(
                "audio extras are not installed. Run: uv pip install -e '.[voice]'"
            ) from exc

        key = _resolve_key(self.hotkey_name)

        def on_press(k):
            if k == key and not self._held.is_set():
                self._press_count += 1
                self._frames = []
                self._held.set()

        def on_release(k):
            if k == key and self._held.is_set():
                self._held.clear()
                self._flush()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=BLOCK,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            self._listener.stop()
            raise AudioError(
                f"could not open the microphone: {exc}. "
                "On macOS, grant your terminal Microphone access in "
                "System Settings > Privacy & Security."
            ) from exc
        return self

    def __exit__(self, *exc) -> None:
        self._closed.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._listener is not None:
            self._listener.stop()

    # -- capture ---------------------------------------------------------

    def _on_audio(self, indata, _frames, _time, status) -> None:
        if self._held.is_set():
            self._frames.append(indata.copy())

    def _flush(self) -> None:
        import numpy as np

        if not self._frames:
            return
        audio = np.concatenate(self._frames, axis=0).reshape(-1)
        self._frames = []
        # Under ~0.3s is almost always a mis-press, not speech.
        if audio.size >= int(0.3 * self.sample_rate):
            self._utterances.put(audio)

    def next_utterance(self, timeout: float = POLL):
        """Wait briefly for an utterance. Returns None if none arrived.

        A plain blocking `get()` here was enough to wedge shutdown: it runs in
        an executor thread, and at interpreter exit Python joins those threads,
        so a read that never returns means the process never exits. Ctrl-C took
        two presses and left a KeyboardInterrupt traceback from
        threading._shutdown. Polling costs nothing and makes the thread
        reliably finishable.
        """
        try:
            return self._utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    def utterances(self) -> Iterator:
        """Yield completed utterances until the listener is closed."""
        while not self._closed.is_set():
            if (audio := self.next_utterance()) is not None:
                yield audio

    # -- barge-in --------------------------------------------------------

    def press_count(self) -> int:
        """Monotonic press tally. Compare across a turn to detect an interrupt."""
        return self._press_count
