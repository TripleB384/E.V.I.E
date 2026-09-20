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
_STOP = re.compile(r"^(?:stop|quiet|shut\s+up|never\s*mind|cancel)[.!?]*$", re.IGNORECASE)
_QUIT = re.compile(r"^(?:quit|exit|goodbye|good\s*night|shut\s*down)[.!?]*$", re.IGNORECASE)

# Verbs that mean "do something", not "tell me something". These need a brain
# with hands -- file access, shell, MCP tools.
_TASK_VERBS = re.compile(
    r"\b(open|read|write|edit|create|make|build|save|delete|rename|move|copy|"
    r"organi[sz]e|refactor|fix|run|install|commit|push|search\s+my|look\s+at\s+my|"
    r"check\s+my|update\s+my|add\s+to\s+my|remind\s+me|schedule|draft|summari[sz]e\s+my)\b",
    re.IGNORECASE,
)
# Things that are plainly conversational, even if long.
_QUICK_STARTS = re.compile(
    r"^(what(?:'s| is| are)?|who|when|where|why|how (?:do|does|much|many|long)|"
    r"is|are|can|could|should|does|do|did|tell me about|explain|define)\b",
    re.IGNORECASE,
)
_QUICK_MAX_WORDS = 18


def strip_address(text: str) -> str:
    return _ADDRESS.sub("", text).strip()


def wants_agentic(text: str) -> bool:
    """True when answering this plausibly means touching something real."""
    return bool(_TASK_VERBS.search(text))


def is_quick(text: str) -> bool:
    words = text.split()
    return (
        len(words) <= _QUICK_MAX_WORDS
        and bool(_QUICK_STARTS.match(text))
        and not wants_agentic(text)
    )


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
        names = ", ".join(registry.names())
        return Decision(Action.REPLY, f"I can run on {names}. Right now, {registry.active}.")

    if _WHICH.match(text):
        return Decision(Action.REPLY, f"I'm running on {registry.active}.")

    if _RESET.match(text):
        return Decision(Action.REPLY, registry.reset().spoken())

    for pattern in (_SWITCH, _USE):
        if match := pattern.match(text):
            wanted = match.group(1).strip()
            try:
                return Decision(Action.REPLY, registry.use(wanted).spoken())
            except UnknownBrain:
                # "use the smallest font" is not a brain swap -- fall through
                # and let a model answer it.
                break

    brain = None
    if registry.quick and is_quick(text):
        brain = registry.quick
    elif wants_agentic(text) and not registry.active_brain.agentic:
        brain = next(
            (n for n in registry.names() if registry.get(n).agentic),
            None,
        )

    return Decision(Action.ANSWER, text, brain)
