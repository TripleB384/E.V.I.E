"""What to do with something you just said.

Two jobs, in order:

  1. Intercept. "Switch to Gemini" is a command to E.V.I.E., not a question for
     a model. Handling it here means a brain swap is instant and costs nothing
     -- no tokens, no quota, no round trip.
  2. Route. A question goes to the fast cheap brain; a task that needs to touch
     files or run commands goes to an agentic one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .brains import BrainRegistry
from .brains.registry import UnknownBrain


class Action(Enum):
    ANSWER = "answer"      # send it to a brain
    REPLY = "reply"        # E.V.I.E. handles it herself, speak `text`
    STOP = "stop"          # stop talking
    QUIT = "quit"


@dataclass
class Decision:
    action: Action
    text: str = ""            # the prompt to send, or the reply to speak
    brain: str | None = None  # force a specific brain for this turn
    tier: str = "normal"      # how hard we judged it, for --debug


# Leading "evie, ..." is address, not content.
#
# Whisper does not hear a name, it hears phonemes, and "Evie" comes back as
# EV, E.V., Eevee, Evee, Eve, Ivy, Avi and more. Matching only the spelling
# meant a command was forwarded to a model, which then earnestly explained it
# could not change models -- a baffling answer to a question that should never
# have reached it.
#
# Anchored to the very start and requiring a separator, so "the eve of the
# election" cannot be mistaken for being spoken to.
# "even" and bare "iv" are deliberately absent: both are ordinary words, and
# "even though I tried" was being heard as being spoken to.
_NAME = r"(?:e+v+(?:ie|ee|y|e)?|e\.?\s?v\.?|ivy|avi[ae]?|eevie)"
_ADDRESS = re.compile(
    rf"^\s*(?:hey\s+|ok(?:ay)?\s+|yo\s+)?{_NAME}[\s,.:!?-]+",
    re.IGNORECASE,
)

# Leading filler. Speech does not start where the command starts: "and then
# tell me which brain you're using", "if you go back to auto routing" and
# "so what brain are you on" all carry the real request in the middle, and
# every start-anchored pattern here missed all three.
#
# Bounded on purpose — it swallows conjunctions, politeness and a short lead-in
# verb, not arbitrary text, so "explain why you should switch back to Claude"
# stays a question for a model rather than becoming a command.
_FILLER = (
    r"^(?:\s*(?:and|so|but|then|also|now|ok(?:ay)?|well|um+|uh+|"
    r"please|actually|wait|hey|yeah|yes|no|hmm+|"
    r"i\s+(?:said|meant|want(?:ed)?\s+to\s+know|wonder(?:ed)?)|"
    r"my\s+bad|sorry|just|quick(?:ly)?|maybe|can\s+you|could\s+you|"
    r"would\s+you|will\s+you|do\s+you\s+know|tell\s+me|remind\s+me|"
    r"let'?s|if(?:\s+you)?)\b[\s,.:;-]*){0,4}"
)
# "no" is filler, because "no, switch to Claude" is a correction and means
# switch. "No, don't switch to Claude" is the opposite in the same words, so
# any pattern that *acts* on what it matches has to refuse the negated form:
# the filler eats the "no" quite happily and leaves a command behind.
_NOT_NEGATED = r"(?!(?:do\s*not|don'?t|never|no\s+need|rather\s+not|instead\s+of)\b)"

# Speech comes out inflected. "Switched to Gemini", "switching to Gemini" and
# "let's use Gemini" all mean the same thing, and matching only bare stems sent
# every one of them to a model.
#
# These take _FILLER for the same reason the question patterns do, and went a
# release without it: the two patterns that actually *perform* a switch kept a
# narrow politeness prefix while every pattern that only reports state was
# widened. So "Now switch to Claude" — the ordinary way to say it — reached a
# model, which answered "I'm staying right here on the current model" and
# settled a routing question it has no authority over. Any leading word at all
# broke the command, not just an unusual one.
_SWITCH = re.compile(
    _FILLER + _NOT_NEGATED +
    r"(?:switch|swap|flip|jump|mov)(?:e|ed|es|ing)?\s+"
    r"(?:over\s+|back\s+)?(?:to|two|too)\s+(?:the\s+)?(.+?)"
    r"(?:\s+brain|\s+model|\s+please|\s+instead)?[.!?]*$",
    re.IGNORECASE,
)
_USE = re.compile(
    _FILLER + _NOT_NEGATED +
    r"(?:chang(?:e|ed|es|ing)|us(?:e|ed|es|ing)|run(?:ning)?\s+on|"
    r"go(?:ing)?\s+(?:back\s+)?to)\s+"
    r"(?:over\s+)?(?:to\s+)?(?:the\s+)?(.+?)"
    r"(?:\s+brain|\s+model|\s+instead|\s+please)?[.!?]*$",
    re.IGNORECASE,
)

# Asking to be told who is answering, in some form. Two earlier versions were
# too strict: the first demanded the literal phrase "running on|using|on", so
# "what brain did you use?" missed; the second demanded that the question
# start the utterance, so "...and then tell me which brain you're using" and
# "my bad I meant which brain you're using" both missed. Speech rarely starts
# where the question does.
#
# "brain", "ai" and "llm" are taken as self-referential on their own, because
# in this assistant they mean nothing else. "model" is not -- "what is the
# best model for our pricing" is an ordinary question -- so it is only a
# swap query when the sentence also says *you* are on or using it.
_SELF_USE = (
    r"(?:you'?re|you\s+are|are\s+you|do\s+you|did\s+you|you)\s+"
    r"(?:currently\s+|still\s+)?(?:on|us(?:e|ed|ing)|runn?ing|power\w*)\b"
)
_WHICH = re.compile(
    _FILLER + r"(?:"
    # "which brain", "what other brain", "what AI"
    r"(?:who|what|which)\s+(?:\w+\s+){0,1}?(?:brains?|ai|llm)\b"
    # "...which model you're using", "what brain did you use to answer that"
    r"|(?:who|what|which)\b[^.?!]{0,50}?\b(?:brains?|models?|ai|llm)\b"
    r"[^.?!]{0,30}?\b" + _SELF_USE +
    r"|(?:who|what)\s+(?:are\s+you\s+)?(?:running\s+on|using)\b"
    r"|are\s+you\s+(?:still\s+)?(?:on|using)\b"
    r")",
    re.IGNORECASE,
)
_LIST = re.compile(
    _FILLER + r"(?:list|what are)\s+(?:your\s+)?brains?\b", re.IGNORECASE
)
# "Go back" on its own means the default brain. Anchored to the end of the
# utterance, because "switch back to Claude" is a switch, not a reset --
# unanchored, it swallowed the brain name and quietly reset instead.
_RESET = re.compile(
    _FILLER + r"(?:go\s+back|switch\s+back|reset)"
    r"(?:\s+to\s+(?:the\s+)?(?:default|normal|usual|start|beginning))?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
# Hand routing back to the tiers after pinning a brain by hand.
#
# "if you go back to auto routing" reached a model, which confirmed a switch
# that never happened -- so the lead-in has to be allowed, not just the bare
# word.
_BACK_TO = (
    r"(?:(?:go(?:ing)?|switch(?:ing)?|revert(?:ing)?|reset(?:ting)?|"
    r"fall(?:ing)?|put\s+(?:it|us))\s+)?(?:back\s+)?to\s+"
)
_AUTO = re.compile(
    _FILLER + r"(?:" + _BACK_TO + r")?"
    r"(?:auto(?:matic(?:ally)?)?\b(?:\s+(?:routing|route|mode|"
    r"select\w*|pick\w*|choos\w*))?"
    r"|you\s+(?:choose|pick|decide)"
    r"|(?:choose|pick|decide)\s+(?:for\s+)?yourself"
    r"|stop\s+using\s+\w+|unpin)\b",
    re.IGNORECASE,
)
_STOP = re.compile(r"^(?:stop|quiet|shut\s+up|never\s*mind|cancel)[.!?]*$", re.IGNORECASE)
_QUIT = re.compile(r"^(?:quit|exit|goodbye|good\s*night|shut\s*down)[.!?]*$", re.IGNORECASE)

# --- how hard is this? ----------------------------------------------------
#
# Four tiers, decided before any model is called, because asking a model how
# hard a question is costs as much as answering it. The signals are crude on
# purpose: a wrong guess falls one tier, it does not fail.

# Work that needs reasoning, structure, or sustained output. A stronger brain
# earns its latency here; for "what time is it" it does not.
_HARD_WORK = re.compile(
    r"\b(analy[sz]e|compare|contrast|evaluate|critique|review|assess|"
    r"design|architect|plan|strategy|strategi[sz]e|outline|"
    r"debug|diagnose|troubleshoot|derive|prove|optimi[sz]e|refactor|"
    r"essay|paper|thesis|report|proposal|pitch|memo|"
    r"draft|compose|rewrite|word(?:ing|smith)|"
    r"write\s+(?:me\s+)?(?:a|an|the|some)\b|"
    r"^why\b|why (?:does|do|is|are|did|would|should)|how (?:would|should|could) i|"
    r"walk me through|step by step|pros and cons|trade-?offs?|"
    r"help me (?:think|decide|figure)|what(?:'s| is) the best way)\b",
    re.IGNORECASE,
)
# Length alone is a decent proxy once a request stops being a lookup.
_HARD_WORDS = 28

# Verbs that mean "do something", not "tell me something". These need a brain
# with hands -- file access, shell, MCP tools.
_TASK_VERBS = re.compile(
    r"\b(open|read|edit|create|save|delete|rename|move|copy|"
    r"organi[sz]e|refactor|fix|run|install|commit|push|search\s+my|look\s+at\s+my|"
    r"check\s+my|update\s+my|add\s+to\s+my|remind\s+me|schedule|"
    r"summari[sz]e\s+my|(?:write|save|export)\s+(?:it\s+|that\s+|them\s+)?"
    r"(?:to|into|in)\s+(?:a\s+)?(?:file|note|the\s+vault))\b",
    re.IGNORECASE,
)
# Things that are plainly conversational, even if long.
_QUICK_STARTS = re.compile(
    r"^(what(?:'s| is| are)?|who|when|where|how (?:do|does|did|is|are|\w+)|"
    r"is|are|can|could|should|does|do|did|tell me about|explain|define)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hey|hi|hello|yo|sup|morning|good\s+(?:morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|cheers|nice|cool|ok|okay|got\s+it|never\s*mind)"
    r"[\s,!.?]*$",
    re.IGNORECASE,
)
_QUICK_MAX_WORDS = 18


def strip_address(text: str) -> str:
    return _ADDRESS.sub("", text).strip()


def wants_agentic(text: str) -> bool:
    """True when answering this plausibly means touching something real."""
    return bool(_TASK_VERBS.search(text))


def is_quick(text: str) -> bool:
    if _GREETING.match(text):
        return True
    words = text.split()
    return (
        len(words) <= _QUICK_MAX_WORDS
        and bool(_QUICK_STARTS.match(text))
        and not wants_agentic(text)
    )


def complexity(text: str) -> str:
    """Classify a request as agentic, hard, simple or normal.

    Checked in that order, because the categories overlap: "summarize my
    lecture notes" reads as hard work but is really a file task, and
    "compare these two" is short but not a lookup.
    """
    if wants_agentic(text):
        return "agentic"
    if _HARD_WORK.search(text) or len(text.split()) >= _HARD_WORDS:
        return "hard"
    if is_quick(text):
        return "simple"
    return "normal"


def route(said: str, registry: BrainRegistry) -> Decision:
    """Turn a transcribed utterance into something to do."""
    text = strip_address(said)
    if not text:
        return Decision(Action.REPLY, "I didn't catch that.")

    if _QUIT.match(text):
        return Decision(Action.QUIT)
    if _STOP.match(text):
        return Decision(Action.STOP)

    if _LIST.match(text):
        ready = ", ".join(registry.usable()) or "nothing"
        return Decision(
            Action.REPLY, f"I can run on {ready}. Right now, {registry.active}."
        )

    if _WHICH.match(text):
        held = " — you picked that one" if registry.pinned else ""
        return Decision(Action.REPLY, f"I'm running on {registry.active}{held}.")

    if _RESET.match(text):
        return Decision(Action.REPLY, registry.reset().spoken())

    if _AUTO.match(text):
        registry.unpin()
        return Decision(Action.REPLY, "Choosing for myself again.")

    for pattern in (_SWITCH, _USE):
        if match := pattern.match(text):
            wanted = match.group(1).strip()
            try:
                result = registry.use(wanted)  # pins: a choice by hand sticks
            except UnknownBrain:
                # "use the smallest font" is not a brain swap -- fall through
                # and let a model answer it.
                break
            spoken = result.spoken()
            if registry.is_parked(wanted := result.brain):
                spoken += f" Heads up, {registry.parked[wanted]}."
            return Decision(Action.REPLY, spoken)

    tier = complexity(text)
    return Decision(Action.ANSWER, text, registry.for_tier(tier), tier)
