"""Turning a token stream into speakable chunks.

This is the whole trick behind E.V.I.E. feeling fast. A model streams tokens
one at a time, but a TTS engine needs a phrase. Waiting for the full answer
before speaking costs several seconds of dead air; flushing on every token
produces stuttering nonsense. So we buffer and flush on sentence boundaries,
with a soft cap so a model that writes long unpunctuated runs still gets heard.

Speech also wants different text than a screen does: an abbreviation is fine to
read and awkward to hear, and nobody wants a code fence read aloud.
"""

from __future__ import annotations

import re
from typing import AsyncIterator, Iterator

# A sentence end, but not a decimal point, an abbreviation, or an ellipsis
# mid-thought. Requires the following character to be whitespace or the end.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s)|(?<=[.!?])$|\n\n")
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.",
    "fig.", "approx.", "no.", "inc.", "jr.", "sr.", "a.m.", "p.m.", "u.s.",
}
_MIN_CHUNK = 12     # below this, keep buffering; a lone "Yes." sounds clipped
_MAX_CHUNK = 320    # above this, cut at a word boundary rather than stall


def _ends_on_abbreviation(text: str) -> bool:
    tail = text.rstrip().lower().split()
    return bool(tail) and tail[-1] in _ABBREVIATIONS


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off complete sentences, returning them and the unflushed remainder."""
    out: list[str] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        candidate = buffer[cursor:end]
        if _ends_on_abbreviation(candidate):
            continue
        if len(candidate.strip()) < _MIN_CHUNK:
            continue
        out.append(candidate.strip())
        cursor = end
    return out, buffer[cursor:]


def cut_long(buffer: str) -> tuple[list[str], str]:
    """Break an over-long unpunctuated buffer at word boundaries.

    Some models write for a paragraph without a full stop. Without this, the
    sentence splitter never fires and E.V.I.E. stays silent until the answer
    ends -- the exact stall the whole streaming design exists to avoid.
    """
    out: list[str] = []
    while len(buffer) > _MAX_CHUNK:
        cut = buffer.rfind(" ", 0, _MAX_CHUNK)
        if cut <= 0:
            cut = _MAX_CHUNK  # one enormous token; cut mid-word rather than stall
        if piece := buffer[:cut].strip():
            out.append(piece)
        buffer = buffer[cut:].lstrip()
    return out, buffer


async def speakable(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup a token stream into phrases a TTS engine can speak naturally."""
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        sentences, buffer = split_sentences(buffer)
        for sentence in sentences:
            yield sentence
        forced, buffer = cut_long(buffer)
        for phrase in forced:
            yield phrase
    if buffer.strip():
        yield buffer.strip()


# -- making text sound like speech --------------------------------------

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,2}|_{1,2})(\S.*?\S|\S)\1")


def for_speech(text: str) -> str:
    """Strip what reads fine but hears badly."""
    text = _CODE_FENCE.sub(" (code omitted) ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def iter_sentences(text: str) -> Iterator[str]:
    """Synchronous counterpart, for speaking text you already have in full."""
    sentences, remainder = split_sentences(text)
    yield from sentences
    if remainder.strip():
        yield remainder.strip()
