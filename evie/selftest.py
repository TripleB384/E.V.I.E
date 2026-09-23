"""Does any of this actually work?

Every other check in this project asks a cheaper question. `brains list` pings
`/models` or runs `shutil.which`; neither runs an inference, and `claude` once
showed green in that table while real calls were already failing. The fallback
chain is covered end to end in the test suite, but only against fakes raising
a synthetic `BrainExhausted` -- and every real failure of that chain has been a
*classification* bug rather than a chain bug: Google answering 400 for a bad
key, a model-404 marked fatal. Both stranded a whole turn. Both passed the
suite.

So this module makes real calls. Four checks, in increasing cost:

  switch    no keys, no network, no model calls -- and asserts that last part
  classify  provoke a real rejection from each provider, check we read it right
  chain     make the head of the chain fail, confirm the next brain answers
  reach     one real inference per brain

Formatting lives in cli.py. Everything here returns data.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import AsyncIterator, Callable, Iterable, Iterator

from .brains import (
    BrainError,
    BrainExhausted,
    BrainRefused,
    BrainRegistry,
    Context,
    OpenAICompatBrain,
)

# Short enough to cost almost nothing, explicit enough that a brain answering
# something else tells you the plumbing is wrong rather than the model.
PROBE = "Reply with exactly: OK"

# A key that is the right shape and definitely not valid anywhere. Some
# providers route differently for a malformed key than for a rejected one, so
# "" or "x" would test a different code path than the one that bites you.
BAD_KEY = "sk-evie-selftest-invalid-0000000000000000000000000000"
BAD_KEY_VAR = "EVIE_SELFTEST_BAD_KEY"

# `claude -p` boots a whole Claude Code session before it answers; 5-11s is
# normal and not a failure.
REACH_TIMEOUT = 90.0


@dataclass
class Result:
    phase: str          # reach | classify | chain | switch
    name: str
    ok: bool
    detail: str
    skipped: bool = False   # not configured, or not testable -- not a failure
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return not self.ok and not self.skipped


def failures(results: Iterable[Result]) -> list[Result]:
    return [r for r in results if r.failed]


# -- helpers ---------------------------------------------------------------


class _Faulty:
    """Stands in for a brain and fails on command.

    A free tier cannot be exhausted on demand without spending it, so the
    chain is tested by making its head fail the way a spent quota does. This
    proves the chain *reacts*; `classify` is what proves a real provider error
    is read correctly in the first place.
    """

    def __init__(self, real, error: BrainError) -> None:
        self._real = real
        self._error = error
        self.name = real.name
        self.agentic = real.agentic

    def describe(self) -> str:
        return f"{self.name} (forced to fail)"

    def missing(self) -> str | None:
        return None

    async def stream(self, prompt: str, ctx: Context) -> AsyncIterator[str]:
        raise self._error
        yield ""  # unreachable; makes this an async generator

    async def health(self):
        return await self._real.health()


@contextmanager
def _substituted(registry: BrainRegistry, name: str, impl) -> Iterator[None]:
    """Swap one brain out and put it back, whatever happens."""
    original = registry._brains[name]
    registry._brains[name] = impl
    try:
        yield
    finally:
        registry._brains[name] = original


@contextmanager
def _env(var: str, value: str) -> Iterator[None]:
    before = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = before


def _transport_failure(exc: BaseException) -> bool:
    """True when we never reached the provider at all.

    A blocked egress proxy, no wifi or a local server that is not running all
    surface as `BrainUnavailable`, which is *correct* behaviour -- but reading
    that as "this provider classifies its errors properly" would be a free
    pass for a test that never ran.
    """
    cause = getattr(exc, "__cause__", None)
    return cause is not None and type(cause).__module__.startswith("httpx")


def _blamed(notices: Iterable[str]) -> set[str]:
    """Which brains a run of notices actually reported as having failed.

    `_explain()` leads with the failing brain's name ("groq is rate limited
    (...). Switching to gemini_api."), so the first word is the one being
    blamed. Matching the name anywhere in the notice instead would count
    "Switching to b" as "b failed", which is the opposite of what it says.
    """
    return {n.split()[0] for n in notices if n.split()}


def _usable_chain(registry: BrainRegistry, skip: Iterable[str] = ()) -> list[str]:
    skip = set(skip)
    return [
        n for n in registry._chain(registry.active)
        if not registry.is_parked(n) and n not in skip
    ]


# -- 1. can every brain actually answer? -----------------------------------


async def reach(
    registry: BrainRegistry,
    *,
    only: str | None = None,
    skip: Iterable[str] = (),
) -> list[Result]:
    """One real inference per brain.

    Calls each brain *directly*, never through `registry.stream()`. The chain
    exists to hide a broken brain, which is exactly what must not happen when
    the question being asked is "is this brain broken".
    """
    skip = {registry.resolve(n) for n in skip}
    names = [registry.resolve(only)] if only else registry.names()
    results: list[Result] = []

    for name in names:
        if name in skip:
            results.append(Result("reach", name, True, "skipped by request", skipped=True))
            continue
        if registry.is_parked(name):
            results.append(
                Result("reach", name, True, registry.parked[name], skipped=True)
            )
            continue

        impl = registry.get(name)
        ctx = Context(system="You are a test probe. Answer in one word.")
        started = time.monotonic()
        try:
            said = await asyncio.wait_for(_collect(impl, ctx), timeout=REACH_TIMEOUT)
        except asyncio.TimeoutError:
            results.append(
                Result("reach", name, False, f"no answer within {REACH_TIMEOUT:.0f}s",
                       seconds=time.monotonic() - started)
            )
        except Exception as exc:
            results.append(
                Result("reach", name, False, f"{type(exc).__name__}: {exc}",
                       seconds=time.monotonic() - started)
            )
        else:
            elapsed = time.monotonic() - started
            if said.strip():
                results.append(
                    Result("reach", name, True, said.strip()[:60], seconds=elapsed)
                )
            else:
                # No error and no text. A model id that silently returns
                # nothing, or a stream shape we do not parse.
                results.append(
                    Result("reach", name, False, "connected but returned no text",
                           seconds=elapsed)
                )
    return results


async def _collect(impl, ctx: Context) -> str:
    parts: list[str] = []
    async for chunk in impl.stream(PROBE, ctx):
        parts.append(chunk)
        if sum(len(p) for p in parts) > 200:
            break  # enough to prove it talks; do not pay for a whole answer
    return "".join(parts)


# -- 2. is a real provider rejection read as retryable? --------------------


async def classify(
    registry: BrainRegistry, *, only: str | None = None, skip: Iterable[str] = ()
) -> list[Result]:
    """Hand each provider a junk key and check how we read the refusal.

    This is the regression that has bitten twice. A rejected credential must
    raise something `retryable_elsewhere`, so the chain moves on. Classified
    as `BrainRefused` instead, one bad key ends the turn and strands the user
    even though four working brains sit behind it.

    Rejected requests are not billed, so this costs nothing.
    """
    skip = {registry.resolve(n) for n in skip}
    names = [registry.resolve(only)] if only else registry.names()
    results: list[Result] = []

    for name in names:
        impl = registry.get(name)
        if name in skip:
            results.append(Result("classify", name, True, "skipped by request", skipped=True))
            continue
        if not isinstance(impl, OpenAICompatBrain) or not impl.spec.api_key_env:
            # A CLI brain authenticates through a login we are not going to
            # break on purpose, and a keyless endpoint has no credential to
            # reject.
            results.append(
                Result("classify", name, True, "no API key to invalidate", skipped=True)
            )
            continue

        # `type(impl)`, not the base class: a subclass stays a subclass, which
        # is what lets this be tested without a network.
        probe = type(impl)(replace(impl.spec, api_key_env=BAD_KEY_VAR))
        started = time.monotonic()
        try:
            with _env(BAD_KEY_VAR, BAD_KEY):
                said = await asyncio.wait_for(
                    _collect(probe, Context()), timeout=REACH_TIMEOUT
                )
        except asyncio.TimeoutError:
            results.append(
                Result("classify", name, True, "provider did not answer in time",
                       skipped=True, seconds=time.monotonic() - started)
            )
        except BrainError as exc:
            elapsed = time.monotonic() - started
            if _transport_failure(exc):
                results.append(
                    Result("classify", name, True,
                           "provider unreachable from here — not tested",
                           skipped=True, seconds=elapsed)
                )
            elif exc.retryable_elsewhere:
                results.append(
                    Result("classify", name, True,
                           f"rejected as {type(exc).__name__} — chain moves on",
                           seconds=elapsed)
                )
            else:
                results.append(
                    Result("classify", name, False,
                           f"rejected as {type(exc).__name__} — FATAL, would strand "
                           f"the turn instead of falling back: {exc}",
                           seconds=elapsed)
                )
        except Exception as exc:
            results.append(
                Result("classify", name, False,
                       f"raised {type(exc).__name__}, which the chain cannot "
                       f"handle at all: {exc}",
                       seconds=time.monotonic() - started)
            )
        else:
            # No error raised. Two very different situations, and the first
            # version of this reported both as "the invalid key was accepted"
            # -- which accused `github` of something it had probably not done.
            elapsed = time.monotonic() - started
            if said.strip():
                results.append(
                    Result("classify", name, False,
                           f"a deliberately invalid key was accepted — it "
                           f"answered: {said.strip()[:60]}",
                           seconds=elapsed)
                )
            else:
                results.append(
                    Result("classify", name, True,
                           "answered with no content and no error — nothing to "
                           "classify; the chain treats an empty answer as a "
                           "failure and moves on",
                           skipped=True, seconds=elapsed)
                )
    return results


# -- 3. does the chain move on? --------------------------------------------


async def chain(
    registry: BrainRegistry, *, deep: bool = False, skip: Iterable[str] = ()
) -> list[Result]:
    """Make a brain fail and confirm the next one picks the answer up."""
    usable = _usable_chain(registry, {registry.resolve(n) for n in skip})
    if len(usable) < 2:
        return [
            Result("chain", "fallback", True,
                   f"needs two usable brains, found {len(usable)} — "
                   f"configure another key to test this",
                   skipped=True)
        ]

    positions = range(len(usable) - 1) if deep else [0]
    results = [await _exhaust_at(registry, usable, i) for i in positions]
    results.append(await _refusal_stops_the_chain(registry, usable))
    return results


async def _exhaust_at(
    registry: BrainRegistry, usable: list[str], index: int
) -> Result:
    """Force one brain to fail; confirm someone further down answers.

    Deliberately not "the *next* brain answers". On the first real run this
    reported a failure when the chain had worked perfectly: groq was forced
    to fail, gemini_api was genuinely 403'd, and openrouter answered. Two
    brains down and still an answer is the design succeeding, not failing.
    What matters is that some brain after the victim picked it up and that
    you heard about every one that did not.
    """
    victim = usable[index]
    candidates = usable[index + 1:]
    label = f"{victim} gives out"
    started = time.monotonic()

    error = BrainExhausted("simulated: out of quota")
    with _restored(registry), _substituted(registry, victim, _Faulty(registry.get(victim), error)):
        registry.use(victim, pin=False)
        notices: list[str] = []
        text: list[str] = []
        try:
            async for kind, chunk in registry.stream(PROBE, Context()):
                (text if kind == "text" else notices).append(chunk)
        except Exception as exc:
            return Result("chain", label, False,
                          f"nobody picked it up — the whole turn died: {exc}",
                          seconds=time.monotonic() - started)

        elapsed = time.monotonic() - started
        landed = registry.active
        if not text:
            return Result("chain", label, False, "fell through but produced no answer",
                          seconds=elapsed)
        if landed not in candidates:
            return Result("chain", label, False,
                          f"answered, but landed on {landed}, which is not after "
                          f"{victim} in the chain",
                          seconds=elapsed)
        blamed = _blamed(notices)
        if victim not in blamed:
            return Result("chain", label, False,
                          "switched silently — you would never hear that it happened",
                          seconds=elapsed)

        # Every brain it stepped over should have been blamed too, or a second
        # failure passes unmentioned behind the first and you never learn that
        # a brain has been dead for a week.
        walked = usable[index:candidates.index(landed) + index + 2]
        unannounced = [n for n in walked[:-1] if n not in blamed]
        if unannounced:
            return Result("chain", label, False,
                          f"answered, but you were never told about "
                          f"{', '.join(unannounced)} failing",
                          seconds=elapsed)
        return Result("chain", label, True,
                      f"{' → '.join(walked)} · heard: {notices[0].strip()[:50]}",
                      seconds=elapsed)


async def _refusal_stops_the_chain(
    registry: BrainRegistry, usable: list[str]
) -> Result:
    """The inverse, and free: a genuinely bad request must NOT walk the chain.

    Reaching for the next brain after a malformed request would spend every
    quota you own re-asking a question that fails identically everywhere. No
    provider is contacted here -- the head raises before a request goes out.
    """
    victim = usable[0]
    error = BrainRefused("simulated: malformed request")
    with _restored(registry), _substituted(registry, victim, _Faulty(registry.get(victim), error)):
        registry.use(victim, pin=False)
        try:
            async for _ in registry.stream(PROBE, Context()):
                pass
        except BrainRefused:
            return Result("chain", "refusal stops the chain", True,
                          "a bad request failed once instead of on every brain")
        except Exception as exc:
            return Result("chain", "refusal stops the chain", False,
                          f"raised {type(exc).__name__} instead of BrainRefused: {exc}")
        return Result("chain", "refusal stops the chain", False,
                      "the chain retried a request that fails everywhere")


@contextmanager
def _restored(registry: BrainRegistry) -> Iterator[None]:
    """Put routing state back. A successful fallback moves `active` on purpose."""
    active, pinned = registry.active, registry.pinned
    try:
        yield
    finally:
        registry._active, registry._pinned = active, pinned


# -- 4. does switching still cost nothing? ---------------------------------

# Every line here was said out loud and mishandled at least once. They are
# kept verbatim, mangling included, because tidied-up phrasings are the ones
# that already worked.
SWITCH_CASES: list[tuple[str, str | None]] = [
    ("switch to Gemini", "gemini_cli"),
    ("EV switch to open route.", "openrouter"),
    ("Eevee switched to Grok.", "groq"),
    ("Evie, switch to claude", "claude"),
    ("switch back to claude", "claude"),
    ("go back to groq", "groq"),
    ("No switch to Claude", "claude"),        # any lead-in word broke this
    ("What are you using?", None),
    ("which brain are you using", None),      # answered, nothing switched
    ("what brain are you on", None),
    ("if you go back to auto routing.", None),
    ("list your brains", None),
]


def switching(build: Callable[[], BrainRegistry]) -> list[Result]:
    """Run the real voice commands through the real router.

    Needs no keys and no network, so this runs on a machine with nothing
    configured at all. A fresh registry per case, because a switch pins.

    The assertion that matters is the last one: the request counters must not
    move. A swap that quietly costs a model call is the bug that produced
    "Switched to OpenRouter" while nothing had switched.
    """
    from .router import Action, route

    results: list[Result] = []
    for said, expected in SWITCH_CASES:
        registry = build()
        before = {n: registry.used_today(n) for n in registry.names()}
        try:
            decision = route(said, registry)
        except Exception as exc:
            results.append(
                Result("switch", said, False, f"{type(exc).__name__}: {exc}")
            )
            continue

        spent = [n for n in registry.names() if registry.used_today(n) != before[n]]
        if decision.action is not Action.REPLY:
            results.append(
                Result("switch", said, False,
                       f"reached a model instead of being handled locally "
                       f"({decision.action.value})")
            )
        elif spent:
            results.append(
                Result("switch", said, False,
                       f"handled locally but still called {', '.join(spent)}")
            )
        elif expected and registry.active != expected:
            results.append(
                Result("switch", said, False,
                       f"landed on {registry.active}, expected {expected}")
            )
        else:
            results.append(
                Result("switch", said, True, decision.text.strip()[:70])
            )
    return results
