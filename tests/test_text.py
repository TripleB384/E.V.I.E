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
