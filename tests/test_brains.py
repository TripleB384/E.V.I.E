"""The brain seam, the hot swap, and the fallback chain.

The fallback chain is the part that makes free tiers usable, so it gets the
most attention here: it has to move on when a brain gives out, and it has to
refuse to move on once a brain has started talking.
"""

import pytest

from evie.brains import (
    BrainExhausted,
    BrainRefused,
    BrainRegistry,
    BrainStatus,
    BrainUnavailable,
    Context,
    EchoBrain,
    Health,
    Turn,
)
from evie.brains.registry import UnknownBrain


class FakeBrain:
    """A brain that says what you tell it to, or fails how you tell it to."""

    def __init__(self, name, *, fail=None, text="ok", agentic=False, fail_after=0):
        self.name = name
        self.agentic = agentic
        self.fail = fail
        self.text = text
        self.fail_after = fail_after
        self.calls = 0
        self.prompts = []

    async def stream(self, prompt, ctx):
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail and not self.fail_after:
            raise self.fail
        for i, word in enumerate(self.text.split()):
            if self.fail and i >= self.fail_after:
                raise self.fail
            yield word + " "

    async def health(self):
        return BrainStatus(Health.OK if not self.fail else Health.EXHAUSTED)


def registry(**brains):
    opts = {k: v for k, v in brains.items() if k in ("default", "fallback", "quick", "aliases")}
    impls = {k: v for k, v in brains.items() if k not in opts}
    opts.setdefault("default", next(iter(impls)))
    return BrainRegistry(impls, **opts)


async def collect(reg, prompt="hi", ctx=None, **kw):
    text, notices = [], []
    async for kind, chunk in reg.stream(prompt, ctx or Context(), **kw):
        (text if kind == "text" else notices).append(chunk)
    return "".join(text).strip(), notices


class TestResolution:
    def test_resolves_aliases(self):
        reg = registry(
            gemini_cli=FakeBrain("gemini_cli"),
            aliases={"google": "gemini_cli", "gemini": "gemini_cli"},
        )
        assert reg.resolve("google") == "gemini_cli"
        assert reg.resolve("GEMINI") == "gemini_cli"

    def test_tolerates_transcription_noise(self):
        # Whisper writes "grok" for "groq" constantly.
        reg = registry(groq=FakeBrain("groq"))
        assert reg.resolve("groq brain") == "groq"

    def test_rejects_nonsense(self):
        reg = registry(groq=FakeBrain("groq"))
        with pytest.raises(UnknownBrain):
            reg.resolve("the smallest font")

    def test_default_must_exist(self):
        with pytest.raises(KeyError):
            BrainRegistry({"a": FakeBrain("a")}, default="b")


class TestSwapping:
    def test_swap_changes_active(self):
        reg = registry(a=FakeBrain("a"), b=FakeBrain("b"), default="a")
        result = reg.use("b")
        assert result.changed and reg.active == "b"
        assert result.spoken() == "Switched to b."

    def test_swap_to_current_is_a_noop(self):
        reg = registry(a=FakeBrain("a"), default="a")
        result = reg.use("a")
        assert not result.changed
        assert "Already" in result.spoken()

    def test_reset_returns_to_default(self):
        reg = registry(a=FakeBrain("a"), b=FakeBrain("b"), default="a")
        reg.use("b")
        reg.reset()
        assert reg.active == "a"


class TestFallbackChain:
    async def test_moves_on_when_rate_limited(self):
        dead = FakeBrain("claude", fail=BrainExhausted("429"))
        alive = FakeBrain("groq", text="from groq")
        reg = registry(claude=dead, groq=alive, default="claude", fallback=["claude", "groq"])

        text, notices = await collect(reg)
        assert text == "from groq"
        assert alive.calls == 1
        assert notices and "rate limited" in notices[0]
        assert "groq" in notices[0]

    async def test_adopts_the_brain_that_worked(self):
        reg = registry(
            claude=FakeBrain("claude", fail=BrainUnavailable("not logged in")),
            groq=FakeBrain("groq"),
            default="claude",
            fallback=["claude", "groq"],
        )
        await collect(reg)
        assert reg.active == "groq", "shouldn't retry a dead brain on every turn"

    async def test_never_switches_mid_answer(self):
        # Splicing two models' replies together would be worse than failing.
        half = FakeBrain("claude", text="one two three", fail=BrainExhausted("429"), fail_after=2)
        backup = FakeBrain("groq")
        reg = registry(claude=half, groq=backup, default="claude", fallback=["claude", "groq"])

        with pytest.raises(BrainExhausted):
            await collect(reg)
        assert backup.calls == 0

    async def test_does_not_retry_a_real_error(self):
        # A malformed request fails everywhere; trying three brains just wastes time.
        reg = registry(
            claude=FakeBrain("claude", fail=BrainRefused("bad request")),
            groq=FakeBrain("groq"),
            default="claude",
            fallback=["claude", "groq"],
        )
        with pytest.raises(BrainRefused):
            await collect(reg)

    async def test_raises_when_everything_is_down(self):
        reg = registry(
            a=FakeBrain("a", fail=BrainExhausted("429")),
            b=FakeBrain("b", fail=BrainUnavailable("offline")),
            default="a",
            fallback=["a", "b"],
        )
        with pytest.raises(BrainUnavailable):
            await collect(reg)

    async def test_explicit_brain_overrides_active(self):
        a, b = FakeBrain("a"), FakeBrain("b", text="from b")
        reg = registry(a=a, b=b, default="a")
        text, _ = await collect(reg, brain="b")
        assert text == "from b" and a.calls == 0


class TestUsageCounting:
    async def test_counts_requests_per_brain(self):
        reg = registry(a=FakeBrain("a"), default="a")
        await collect(reg)
        await collect(reg)
        assert reg.used_today("a") == 2

    async def test_headroom_shows_limit_when_set(self):
        reg = BrainRegistry(
            {"a": FakeBrain("a")}, default="a", daily_limits={"a": 1000}
        )
        await collect(reg)
        assert reg.headroom("a") == "1/1000"


class TestContext:
    def test_handoff_summary_carries_the_thread(self):
        ctx = Context(transcript=[Turn("user", "when is it due"), Turn("assistant", "Thursday")])
        summary = ctx.handoff_summary()
        assert "when is it due" in summary and "Thursday" in summary

    def test_empty_context_has_no_summary(self):
        assert Context().handoff_summary() == ""

    def test_as_messages_includes_system(self):
        msgs = Context(system="be brief", transcript=[Turn("user", "hi")]).as_messages()
        assert msgs[0] == {"role": "system", "content": "be brief"}
        assert msgs[1]["content"] == "hi"


class TestEchoBrain:
    async def test_echoes(self):
        out = "".join([c async for c in EchoBrain().stream("hello", Context())])
        assert out.strip() == "You said: hello"

    async def test_is_always_healthy(self):
        assert (await EchoBrain().health()).usable


class TestCliBrainHealth:
    """`which` proves a binary exists. It proves nothing about being logged in.

    `evie brains list` reported claude as "ready" on a machine where the next
    call failed with "Not logged in". A health check must not claim more than
    it checked.
    """

    def _brain(self, command):
        from evie.brains import CliBrain, CliBrainSpec

        return CliBrain(CliBrainSpec(name="x", command=command))

    async def test_a_found_binary_is_installed_not_ready(self):
        status = await self._brain(["sh", "-c", "{prompt}"]).health()
        assert status.health is Health.UNVERIFIED
        assert status.health is not Health.OK, "cannot promise a login it never checked"
        assert status.usable, "still worth trying -- the real call reports the truth"

    async def test_a_missing_binary_is_missing(self):
        status = await self._brain(["definitely-not-real-xyz", "{prompt}"]).health()
        assert status.health is Health.MISSING
        assert not status.usable

    async def test_an_http_brain_without_a_key_is_not_usable(self):
        from evie.brains import HttpBrainSpec, OpenAICompatBrain

        brain = OpenAICompatBrain(
            HttpBrainSpec(name="g", base_url="http://x/v1", model="m",
                          api_key_env="DEFINITELY_UNSET_KEY_XYZ")
        )
        status = await brain.health()
        assert status.health is Health.UNAUTHENTICATED
        assert not status.usable
