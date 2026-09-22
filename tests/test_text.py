"""The sentence chunker decides whether E.V.I.E. feels fast or slow."""

import pytest

from evie.text import for_speech, iter_sentences, speakable, split_sentences


async def _stream(*chunks):
    for c in chunks:
        yield c


class TestSplitSentences:
    def test_splits_on_terminators(self):
        done, rest = split_sentences("The lexer is finished. Now the parser.")
        # Both are complete -- holding the last one back would delay audio
        # for no reason, since a terminator already ended it.
        assert done == ["The lexer is finished.", "Now the parser."]
        assert rest == ""

    def test_holds_incomplete_tail(self):
        done, rest = split_sentences("This one is complete enough. And this is not")
        assert done == ["This one is complete enough."]
        assert rest.strip() == "And this is not"

    def test_does_not_split_on_abbreviations(self):
        done, _ = split_sentences("Your meeting with Dr. Chen is at 3pm today, confirmed.")
        assert done == ["Your meeting with Dr. Chen is at 3pm today, confirmed."]

    def test_does_not_split_decimals(self):
        done, _ = split_sentences("Revenue came in at 12.5 thousand this month, up nine.")
        assert len(done) == 1

    def test_keeps_short_fragments_buffered(self):
        # "Yes." alone would sound clipped; wait for more.
        done, rest = split_sentences("Yes.")
        assert done == []
        assert rest == "Yes."


class TestSpeakable:
    async def test_yields_first_sentence_before_stream_ends(self):
        out = []
        async for phrase in speakable(
            _stream("The compiler project ", "is due Thursday. ", "You have the lexer done.")
        ):
            out.append(phrase)
        assert out[0] == "The compiler project is due Thursday."
        assert len(out) == 2

    async def test_flushes_unpunctuated_runs(self):
        long_run = "word " * 100  # no punctuation at all
        out = [p async for p in speakable(_stream(long_run))]
        assert len(out) > 1, "a model that never punctuates must not stall the voice"

    async def test_emits_trailing_remainder(self):
        out = [p async for p in speakable(_stream("no terminator here at all"))]
        assert out == ["no terminator here at all"]

    async def test_empty_stream_is_silent(self):
        assert [p async for p in speakable(_stream())] == []


class TestForSpeech:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Run `pytest` now", "Run pytest now"),
            ("## Heading\nbody", "Heading\nbody"),
            ("- one\n- two", "one\ntwo"),
            ("**bold** and _soft_", "bold and soft"),
            ("See [the docs](https://x.com/y)", "See the docs"),
            ("Go to https://example.com now", "Go to a link now"),
        ],
    )
    def test_strips_what_hears_badly(self, raw, expected):
        assert for_speech(raw) == expected

    def test_omits_code_blocks(self):
        spoken = for_speech("Here:\n```python\nprint(1)\n```\nThat's it.")
        assert "print" not in spoken
        assert "code omitted" in spoken

    def test_iter_sentences_covers_whole_text(self):
        text = "First sentence here. Second sentence here. Trailing bit"
        assert "".join(iter_sentences(text)).replace(" ", "") == text.replace(" ", "")


class TestSingleSentenceAnswers:
    """Spoken answers frequently run to one sentence. Waiting for a full stop
    then means waiting for the entire reply.

    Measured on a real turn before this was handled: first token 5.41s, first
    audio 13.35s -- eight seconds of silence while a 192-character answer with
    no interior period accumulated.
    """

    MOON = (
        "The Moon is about 238,900 miles (384,400 km) from Earth on average — though it "
        "varies a bit since the orbit isn't a perfect circle, from around 225,000 miles "
        "at closest to 252,000 at farthest."
    )

    async def _chunks(self, text, size=8):
        async def stream():
            for i in range(0, len(text), size):
                yield text[i : i + size]

        return [p async for p in speakable(stream())]

    async def test_a_one_sentence_answer_still_starts_early(self):
        chunks = await self._chunks(self.MOON)
        assert len(chunks) > 1, "must not buffer a whole single-sentence reply"
        assert len(chunks[0]) < len(self.MOON) / 2, "first phrase should be well short"

    async def test_the_first_phrase_is_a_natural_pause(self):
        chunks = await self._chunks(self.MOON)
        assert chunks[0].endswith("on average"), chunks[0]

    async def test_nothing_is_dropped(self):
        chunks = await self._chunks(self.MOON)
        rejoined = " ".join(chunks).replace(" ", "")
        original = self.MOON.replace(" ", "").replace("—", "")
        assert rejoined.replace("—", "") == original

    async def test_the_first_phrase_beats_the_sentence_threshold(self):
        """A comma early on should release audio sooner than a distant period."""
        text = "Yes, absolutely, and the reason is that the compiler runs in two passes."
        chunks = await self._chunks(text)
        assert chunks[0].startswith("Yes, absolutely,"), chunks[0]


class TestSpokenPunctuation:
    def test_a_dash_becomes_a_pause_not_a_word(self):
        assert for_speech("on average — though it varies") == "on average, though it varies"

    def test_a_phrase_never_opens_on_punctuation(self):
        # Clause splitting can leave a dash or comma leading the next chunk.
        assert for_speech("— though it varies") == "though it varies"
        assert for_speech(", and then") == "and then"
