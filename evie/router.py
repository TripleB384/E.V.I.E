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
_ADDRESS = re.compile(r"^\s*(hey\s+|ok\s+)?(evie|eevee|evy|ivy)[\s,.:!-]+", re.IGNORECASE)

_SWITCH = re.compile(
    r"^(?:please\s+)?(?:switch|change|swap|flip)\s+(?:over\s+)?to\s+(?:the\s+)?(.+?)"
    r"(?:\s+brain|\s+model|\s+please)?[.!?]*$",
    re.IGNORECASE,
)
_USE = re.compile(
    r"^(?:please\s+)?(?:use|run\s+on|go\s+to)\s+(?:the\s+)?(.+?)"
    r"(?:\s+brain|\s+model|\s+instead|\s+please)?[.!?]*$",
    re.IGNORECASE,
)
_WHICH = re.compile(
    r"^(?:who|what|which)\s+(?:brain|model|ai)?\s*(?:are\s+you\s+)?"
    r"(?:running\s+on|using|on)\b.*$|^which\s+brain\b.*$",
    re.IGNORECASE,
)
_LIST = re.compile(r"^(?:list|what are)\s+(?:your\s+)?brains?\b.*$", re.IGNORECASE)
_RESET = re.compile(r"^(?:go\s+back|switch\s+back|reset)\b.*$", re.IGNORECASE)
# Hand routing back to the tiers after pinning a brain by hand.
_AUTO = re.compile(
    r"^(?:auto|automatic|you\s+(?:choose|pick|decide)|"
    r"(?:choose|pick|decide)\s+(?:for\s+)?yourself|"
    r"stop\s+using\s+\w+|unpin)\b.*$",
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
