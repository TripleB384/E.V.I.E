"""The checks that make real calls, checked here without making any.

`evie brains test` is the thing that finally asks "does this brain answer",
so its own logic has to be right before its verdicts mean anything. A
self-test that reports green because it silently fell back to a working brain
is worse than no self-test.
"""

import httpx
import pytest

from evie import selftest
from evie.brains import (
    BrainExhausted,
    BrainRefused,
    BrainRegistry,
    BrainUnavailable,
    Context,
    EchoBrain,
    HttpBrainSpec,
    OpenAICompatBrain,
)
from tests.test_brains import FakeBrain


def registry(**brains):
    opts = {k: v for k, v in brains.items() if k in ("default", "fallback", "aliases")}
    impls = {k: v for k, v in brains.items() if k not in opts}
    opts.setdefault("default", next(iter(impls)))
    return BrainRegistry(impls, **opts)


def http_brain(name, error=None, key_env="EVIE_TEST_KEY"):
    """An HTTP brain that fails however the test says, with no network.

    `classify` rebuilds the brain via `type(impl)(spec)` to swap the key env
    var, so the behaviour has to live on the class, not the instance.
    """

    class Probe(OpenAICompatBrain):
        async def stream(self, prompt, ctx):
            if error is not None:
                raise error
            yield "accepted the junk key"

    return Probe(HttpBrainSpec(name, "https://example.invalid/v1", "m", api_key_env=key_env))


def only(results, name):
    for r in results:
        if r.name == name:
            return r
    # StopIteration out of an async test surfaces as an unrelated RuntimeError,
    # which hides which row was actually missing.
    raise AssertionError(f"no {name!r} row in {[r.name for r in results]}")


# -- reach -----------------------------------------------------------------


class TestReach:
    async def test_a_brain_that_answers_passes(self):
        results = await selftest.reach(registry(echo=EchoBrain()))
        assert only(results, "echo").ok
        assert "You said" in only(results, "echo").detail

    async def test_a_brain_that_raises_fails_with_the_reason(self):
        reg = registry(bad=FakeBrain("bad", fail=BrainUnavailable("key rejected")))
        result = only(await selftest.reach(reg), "bad")
        assert result.failed
        assert "key rejected" in result.detail

    async def test_reach_does_not_fall_back(self):
        """The whole point. `registry.stream()` exists to hide a broken brain
        behind a working one -- which is exactly wrong when the question being
        asked is "is this brain broken"."""
        reg = registry(
            bad=FakeBrain("bad", fail=BrainExhausted("429")),
            good=FakeBrain("good", text="I answered instead."),
            default="bad",
            fallback=["bad", "good"],
        )
        results = await selftest.reach(reg)
        assert only(results, "bad").failed, "a working fallback masked the failure"
        assert only(results, "good").ok

    async def test_an_unconfigured_brain_is_skipped_not_failed(self):
        """No API key is not a broken brain. Reporting it red would make a
        first run look like a disaster."""
        parked = FakeBrain("parked")
        parked.missing = lambda: "$NOPE_API_KEY is not set"
        result = only(await selftest.reach(registry(parked=parked, echo=EchoBrain())), "parked")
        assert result.skipped and result.ok
        assert "NOPE_API_KEY" in result.detail

    async def test_silence_is_a_failure(self):
        """No error and no text: a model id that returns nothing, or a stream
        shape we do not parse. `brains list` would call this healthy."""
        result = only(await selftest.reach(registry(mute=FakeBrain("mute", text=""))), "mute")
        assert result.failed
        assert "no text" in result.detail

    async def test_skip_leaves_a_brain_alone(self):
        pricey = FakeBrain("pricey")
        results = await selftest.reach(registry(pricey=pricey, echo=EchoBrain()), skip=["pricey"])
        assert only(results, "pricey").skipped
        assert pricey.calls == 0, "a skipped brain must not be called"

    async def test_only_tests_one_brain(self):
        other = FakeBrain("other")
        results = await selftest.reach(registry(echo=EchoBrain(), other=other), only="echo")
        assert [r.name for r in results] == ["echo"]
        assert other.calls == 0


# -- classify --------------------------------------------------------------


class TestClassify:
    """The regression that has bitten twice: Google answering 400 for a bad
    key, and a model-404 marked fatal. Both ended a turn that four working
    brains could have answered."""

    async def test_a_retryable_rejection_passes(self):
        reg = registry(p=http_brain("p", BrainUnavailable("the API key was rejected")))
        assert only(await selftest.classify(reg), "p").ok

    async def test_a_fatal_rejection_fails(self):
        reg = registry(p=http_brain("p", BrainRefused("HTTP 400: bad key")))
        result = only(await selftest.classify(reg), "p")
        assert result.failed
        assert "FATAL" in result.detail

    async def test_a_non_brain_error_fails(self):
        """A raw httpx error is not a BrainError, so `registry.stream()` never
        catches it and the turn dies with a traceback. This is how the
        ProxyError hole was found."""
        reg = registry(p=http_brain("p", httpx.ProxyError("403 Forbidden")))
        result = only(await selftest.classify(reg), "p")
        assert result.failed
        assert "chain cannot handle" in result.detail

    async def test_accepting_a_junk_key_fails(self):
        result = only(await selftest.classify(registry(p=http_brain("p", None))), "p")
        assert result.failed
        assert "invalid key was accepted" in result.detail

    async def test_an_unreachable_provider_is_skipped_not_passed(self):
        """Blocked egress and no wifi both raise BrainUnavailable, which is
        correct behaviour -- but reading that as "classification works" would
        be a pass for a test that never ran."""
        blocked = BrainUnavailable("unreachable")
        blocked.__cause__ = httpx.ProxyError("403 Forbidden")
        result = only(await selftest.classify(registry(p=http_brain("p", blocked))), "p")
        assert result.skipped
        assert "not tested" in result.detail

    async def test_cli_brains_are_skipped(self):
        """Breaking a subscription login on purpose is not something a test
        gets to do."""
        result = only(await selftest.classify(registry(c=FakeBrain("c"))), "c")
        assert result.skipped
        assert "no API key" in result.detail


# -- chain -----------------------------------------------------------------


class TestChain:
    async def test_an_exhausted_brain_hands_over(self):
        reg = registry(
            a=FakeBrain("a", text="from a"),
            b=FakeBrain("b", text="from b"),
            default="a",
            fallback=["a", "b"],
        )
        result = only(await selftest.chain(reg), "a gives out")
        assert result.ok
        assert "rate limited" in result.detail

    async def test_a_refusal_does_not_walk_the_chain(self):
        """A malformed request fails identically everywhere. Retrying it down
        the chain would spend every quota you own on the same error."""
        b = FakeBrain("b", text="from b")
        reg = registry(a=FakeBrain("a"), b=b, default="a", fallback=["a", "b"])
        # The helper directly, not the whole phase: the handover check runs
        # first and calls `b` for a legitimate reason of its own.
        result = await selftest._refusal_stops_the_chain(reg, ["a", "b"])
        assert result.ok
        assert b.calls == 0, "the chain retried a request that fails everywhere"

    async def test_routing_state_is_put_back(self):
        """A successful fallback moves `active` on purpose. Leaving it moved
        would mean running the test changed which brain answers next."""
        reg = registry(
            a=FakeBrain("a"), b=FakeBrain("b"), default="a", fallback=["a", "b"]
        )
        await selftest.chain(reg)
        assert reg.active == "a" and reg.pinned is None

    async def test_the_real_brains_are_restored(self):
        reg = registry(a=FakeBrain("a"), b=FakeBrain("b"), default="a", fallback=["a", "b"])
        original = reg.get("a")
        await selftest.chain(reg)
        assert reg.get("a") is original

    async def test_one_brain_cannot_test_a_chain(self):
        result = only(await selftest.chain(registry(a=FakeBrain("a"))), "fallback")
        assert result.skipped
        assert "two usable brains" in result.detail

    async def test_a_silent_switch_is_a_failure(self):
        """You have to hear that it happened. A swap nobody announces is how
        "which brain are you on" started getting surprising answers."""
        reg = registry(
            a=FakeBrain("a"), b=FakeBrain("b"), default="a", fallback=["a", "b"]
        )

        async def mute(prompt, ctx, *, brain=None):
            yield "text", "from b"
            reg._active = "b"

        reg.stream = mute
        assert only(await selftest.chain(reg), "a gives out").failed

    async def test_deep_walks_every_position(self):
        reg = registry(
            a=FakeBrain("a"), b=FakeBrain("b"), c=FakeBrain("c"),
            default="a", fallback=["a", "b", "c"],
        )
        labels = [r.name for r in await selftest.chain(reg, deep=True)]
        assert "a gives out" in labels and "b gives out" in labels


# -- switch ----------------------------------------------------------------


class TestSwitching:
    """Runs against the shipped config, so the aliases are the real ones."""

    def _build(self):
        from evie.config import PACKAGE_DEFAULTS, load_registry

        return lambda: load_registry(PACKAGE_DEFAULTS / "brains.yaml")

    def test_every_shipped_phrasing_is_handled_locally(self):
        bad = [r for r in selftest.switching(self._build()) if r.failed]
        assert not bad, "\n".join(f"{r.name}: {r.detail}" for r in bad)

    def test_it_catches_a_swap_that_costs_a_model_call(self):
        """The failure mode this exists for: "Switched to OpenRouter" spoken
        by a model, 7 seconds after asking, with nothing switched."""
        from evie.config import PACKAGE_DEFAULTS, load_registry

        def leaky():
            reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
            real = reg.use

            def charged(name, **kw):
                reg._counts[reg.resolve(name)].bump()
                return real(name, **kw)

            reg.use = charged
            return reg

        bad = [r for r in selftest.switching(leaky) if r.failed]
        assert bad and all("still called" in r.detail for r in bad)

    def test_it_catches_a_phrasing_that_reaches_a_model(self):
        from evie.config import PACKAGE_DEFAULTS, load_registry

        def deaf():
            reg = load_registry(PACKAGE_DEFAULTS / "brains.yaml")
            reg.resolve = lambda name: (_ for _ in ()).throw(
                __import__("evie.brains.registry", fromlist=["UnknownBrain"]).UnknownBrain(name)
            )
            return reg

        assert [r for r in selftest.switching(deaf) if r.failed]


class TestFailures:
    def test_skipped_results_are_not_failures(self):
        results = [
            selftest.Result("reach", "a", True, ""),
            selftest.Result("reach", "b", True, "", skipped=True),
            selftest.Result("reach", "c", False, "broken"),
        ]
        assert [r.name for r in selftest.failures(results)] == ["c"]


class TestChainAcceptsARealWalk:
    """The first live run reported a failure when the chain had worked.

    groq was forced to fail, gemini_api was genuinely 403'd, and openrouter
    answered. Two brains down and still an answer is the design succeeding.
    The assertion demanded the *next* brain specifically, and called it a
    failure.
    """

    async def test_it_accepts_a_two_step_walk(self):
        reg = registry(
            a=FakeBrain("a"),
            b=FakeBrain("b", fail=BrainUnavailable("403 denied")),
            c=FakeBrain("c", text="from c"),
            default="a",
            fallback=["a", "b", "c"],
        )
        result = only(await selftest.chain(reg), "a gives out")
        assert result.ok, result.detail
        assert "a → b → c" in result.detail

    async def test_a_skipped_brain_must_still_be_announced(self):
        """Or a second failure passes unmentioned behind the first, and you
        never learn that gemini has been dead for a week."""
        reg = registry(
            a=FakeBrain("a"),
            b=FakeBrain("b", fail=BrainUnavailable("403")),
            c=FakeBrain("c", text="from c"),
            default="a",
            fallback=["a", "b", "c"],
        )

        async def half_mute(prompt, ctx, *, brain=None):
            yield "notice", "a is rate limited. Switching to b."
            yield "text", "from c"
            reg._active = "c"

        reg.stream = half_mute
        result = only(await selftest.chain(reg), "a gives out")
        assert result.failed
        assert "never told about b" in result.detail

    async def test_landing_outside_the_chain_still_fails(self):
        reg = registry(
            a=FakeBrain("a"), b=FakeBrain("b"), default="a", fallback=["a", "b"]
        )

        async def wrong(prompt, ctx, *, brain=None):
            yield "notice", "a is rate limited. Switching to b."
            yield "text", "from somewhere"
            reg._active = "a"

        reg.stream = wrong
        assert only(await selftest.chain(reg), "a gives out").failed


class TestClassifyTellsEmptyFromAccepted:
    """The first live run said `github` "accepted a deliberately invalid key".

    That branch fired whenever no exception was raised and threw away what
    came back, so a 200 carrying a real answer and a 200 carrying nothing
    produced the same verdict — accusing the provider of something it had
    probably not done.
    """

    async def test_text_back_means_the_key_really_was_accepted(self):
        result = only(await selftest.classify(registry(p=http_brain("p", None))), "p")
        assert result.failed
        assert "accepted the junk key" in result.detail, "it should quote what came back"

    async def test_nothing_back_is_reported_as_nothing_back(self):
        class Silent(OpenAICompatBrain):
            async def stream(self, prompt, ctx):
                return
                yield ""

        reg = registry(
            p=Silent(HttpBrainSpec("p", "https://x.invalid/v1", "m", api_key_env="K"))
        )
        result = only(await selftest.classify(reg), "p")
        assert result.skipped, "the chain handles this; it is not E.V.I.E. misbehaving"
        assert "no content" in result.detail
        assert "accepted" not in result.detail
