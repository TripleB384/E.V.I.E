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

    missing = staticmethod(lambda: None)  # configured, unless a test says otherwise

    def __init__(self, name, *, fail=None, text="ok", agentic=False, fail_after=0,
                 model="fake-1"):
        self.name = name
        self.agentic = agentic
        self.fail = fail
        self.text = text
        self.fail_after = fail_after
        self.model = model
        self.calls = 0
        self.prompts = []

    def describe(self):
        return f"the {self.model} model"

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


class TestParking:
    """A brain you never set up should stay out of the way.

    Distinct from one that is down. With no API key there is nothing to try,
    so attempting it only produces a spoken "switching to..." for a brain the
    user never configured.
    """

    def _registry(self, **missing):
        brains = {}
        for name, reason in missing.items():
            brain = FakeBrain(name)
            brain.missing = (lambda r=reason: r) if reason else (lambda: None)
            brains[name] = brain
        return BrainRegistry(
            brains, default=next(iter(brains)), fallback=list(brains)
        )

    def test_unconfigured_brains_are_parked(self):
        reg = self._registry(groq=None, gemini_api="$GEMINI_API_KEY is not set")
        assert reg.usable() == ["groq"]
        assert reg.parked == {"gemini_api": "$GEMINI_API_KEY is not set"}

    def test_a_parked_brain_is_still_listed_and_nameable(self):
        """Parking governs routing, not existence -- naming one should
        explain what it needs, not claim it does not exist."""
        reg = self._registry(groq=None, gemini_api="$GEMINI_API_KEY is not set")
        assert "gemini_api" in reg.names()
        assert reg.resolve("gemini_api") == "gemini_api"
        assert reg.is_parked("gemini_api")

    async def test_the_chain_skips_parked_brains_silently(self):
        reg = self._registry(
            claude="claude is not installed",
            gemini_api="$GEMINI_API_KEY is not set",
            groq=None,
        )
        text, notices = await collect(reg, brain="groq")
        assert text == "ok"
        assert notices == [], "no announcement for brains that were never set up"

    async def test_naming_a_parked_brain_still_tries_it(self):
        """An explicit request deserves the brain's own error, not silence."""
        reg = self._registry(groq=None, gemini_api="$GEMINI_API_KEY is not set")
        reg.get("gemini_api").fail = BrainUnavailable("$GEMINI_API_KEY is not set")
        text, _ = await collect(reg, brain="gemini_api")
        assert text == "ok", "should fall through to the configured brain"


class TestPinning:
    def test_use_pins_by_default(self):
        reg = BrainRegistry(
            {"a": FakeBrain("a"), "b": FakeBrain("b")}, default="a",
            tiers={"simple": "a", "hard": "b"},
        )
        reg.use("b")
        assert reg.pinned == "b"
        assert reg.for_tier("simple") == "b", "a hand-picked brain must stick"

    def test_unpin_restores_tier_routing(self):
        reg = BrainRegistry(
            {"a": FakeBrain("a"), "b": FakeBrain("b")}, default="a",
            tiers={"simple": "a", "hard": "b"},
        )
        reg.use("b")
        reg.unpin()
        assert reg.for_tier("simple") == "a"

    def test_reset_also_releases_the_pin(self):
        reg = BrainRegistry(
            {"a": FakeBrain("a"), "b": FakeBrain("b")}, default="a",
            tiers={"simple": "a"},
        )
        reg.use("b")
        reg.reset()
        assert reg.pinned is None and reg.active == "a"


class TestErrorClassification:
    """Which failures are worth trying another brain for.

    Providers disagree on status codes: Google answers an invalid key with
    400, most others with 401. Reading the status alone made a bad Google key
    a fatal BrainRefused, so one wrong credential killed a request the
    fallback chain could have answered.
    """

    def _classify(self, status, body):
        from evie.brains.openai_compat import _from_status

        return _from_status(status, body)

    def test_googles_400_for_a_bad_key_is_recoverable(self):
        body = '{"error": {"code": 400, "message": "Please pass a valid API key"}}'
        exc = self._classify(400, body)
        assert isinstance(exc, BrainUnavailable)
        assert exc.retryable_elsewhere, "the chain must be able to move on"

    def test_the_message_says_what_to_check(self):
        exc = self._classify(400, '{"message": "Please pass a valid API key"}')
        said = str(exc).lower()
        assert "right provider" in said and "quotes" in said and "expired" in said

    def test_a_401_is_still_recoverable(self):
        assert isinstance(self._classify(401, "unauthorized"), BrainUnavailable)

    def test_quota_language_reads_as_exhausted_at_any_status(self):
        exc = self._classify(403, '{"message": "Quota exceeded for this project"}')
        assert isinstance(exc, BrainExhausted)

    def test_a_retired_model_is_recoverable(self):
        """I originally asserted the opposite here, reasoning that a bad model
        name "fails identically everywhere". It does not: a model id is
        per-provider configuration. Google retiring gemini-2.5-flash ended a
        real turn that Groq would have answered."""
        exc = self._classify(
            404,
            '{"error":{"code":404,"message":"models/gemini-2.5-flash is no longer '
            'available to new users. Please update to models/gemini-3.6-flash"}}',
        )
        assert isinstance(exc, BrainUnavailable)
        assert exc.retryable_elsewhere
        assert "brains.yaml" in str(exc), "should say where to change it"

    def test_an_unknown_model_name_is_recoverable_too(self):
        exc = self._classify(400, '{"error": {"message": "model not found: gpt-9"}}')
        assert isinstance(exc, BrainUnavailable)

    def test_a_malformed_request_stays_fatal(self):
        """Nothing here is provider-specific, so three more round trips would
        produce three more identical rejections."""
        exc = self._classify(400, '{"error": {"message": "messages: field required"}}')
        assert isinstance(exc, BrainRefused)
        assert not exc.retryable_elsewhere

    def test_429_is_exhausted(self):
        assert isinstance(self._classify(429, "slow down"), BrainExhausted)


class TestDescribe:
    """What goes into the system prompt when a brain is asked what it is.

    It has to be true. A model left to answer from its own weights said
    "GPT-4" three times while running on Groq's gpt-oss-120b.
    """

    def test_an_http_brain_names_its_model_and_host(self):
        from evie.brains import HttpBrainSpec, OpenAICompatBrain

        brain = OpenAICompatBrain(
            HttpBrainSpec("groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b")
        )
        said = brain.describe()
        assert "openai/gpt-oss-120b" in said
        assert "api.groq.com" in said

    def test_a_cli_brain_does_not_invent_a_model_id(self):
        """Which model `claude -p` picks comes from that tool's own config
        and can change between calls, so naming one here would be a guess."""
        from evie.brains import CliBrain, CliBrainSpec

        said = CliBrain(CliBrainSpec("claude", ["claude", "-p", "{prompt}"])).describe()
        assert "claude" in said
        assert "sonnet" not in said and "opus" not in said

    def test_the_registry_describes_through_an_alias(self):
        reg = registry(groq=FakeBrain("groq", model="gpt-oss"), aliases={"fast": "groq"})
        assert "gpt-oss" in reg.describe("fast")

    def test_a_brain_without_describe_falls_back_to_its_name(self):
        class Older:
            name = "old"
            agentic = False

            def missing(self):
                return None

            async def stream(self, prompt, ctx):
                yield "hi"

            async def health(self):
                raise NotImplementedError

        assert registry(old=Older()).describe("old") == "old"


class TestTransportFailuresAreSurvivable:
    """A failure to get the bytes through must be a BrainError, not a raw one.

    `registry.stream()` only catches `BrainError`. Anything else escapes the
    fallback loop entirely, so the turn dies with a traceback while four
    working brains sit untouched behind it.

    Found by `evie brains test`: a 403 from an egress proxy took out four
    brains at once. `stream()` was catching only `ConnectError` and
    `TimeoutException`, which leaves ProxyError, ReadError and
    RemoteProtocolError — a dropped connection mid-answer — uncaught.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "error",
        [
            "ProxyError",           # any proxy, corporate or otherwise
            "ReadError",            # connection dropped mid-stream
            "RemoteProtocolError",  # server hung up
            "UnsupportedProtocol",  # a malformed base_url in brains.yaml
            "ConnectError",
        ],
    )
    async def test_an_http_transport_error_is_a_brain_error(self, monkeypatch, error):
        import httpx

        from evie.brains import BrainError, HttpBrainSpec, OpenAICompatBrain

        raised = getattr(httpx, error)("nope")

        class Boom:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **kw):
                raise raised

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Boom())
        brain = OpenAICompatBrain(HttpBrainSpec("p", "https://x.invalid/v1", "m"))

        with pytest.raises(BrainError) as exc:
            async for _ in brain.stream("hi", Context()):
                pass
        assert exc.value.retryable_elsewhere, "the chain must be allowed to move on"

    async def test_the_chain_survives_a_proxy_error(self, monkeypatch):
        """The payoff: the whole point of classifying it correctly."""
        import httpx

        from evie.brains import HttpBrainSpec, OpenAICompatBrain

        class Boom:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **kw):
                raise httpx.ProxyError("403 Forbidden")

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Boom())
        reg = registry(
            blocked=OpenAICompatBrain(HttpBrainSpec("blocked", "https://x.invalid/v1", "m")),
            backup=FakeBrain("backup", text="Still here."),
            default="blocked",
            fallback=["blocked", "backup"],
        )
        text, notices = await collect(reg)
        assert text.strip() == "Still here."
        assert notices and "blocked" in notices[0]

    async def test_a_cli_that_cannot_be_spawned_is_a_brain_error(self, monkeypatch):
        """`shutil.which` is not a guarantee: a broken symlink, a missing
        execute bit, or an upgrade swapping the binary between the check and
        the spawn all raise OSError, which is not a BrainError."""
        import asyncio as _asyncio

        from evie.brains import BrainUnavailable, CliBrain, CliBrainSpec

        monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/" + exe)

        async def cannot_spawn(*a, **kw):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", cannot_spawn)
        brain = CliBrain(CliBrainSpec("c", ["nope", "{prompt}"]))

        with pytest.raises(BrainUnavailable) as exc:
            async for _ in brain.stream("hi", Context()):
                pass
        assert "Permission denied" in str(exc.value)


class TestAnEmptyAnswerIsAFailure:
    """A brain that returns nothing must not end the turn in silence.

    `stream()` used to `return` when a brain yielded nothing and raised
    nothing: no text, no error, no fallback. Worse than a crash, because a
    crash at least names itself — E.V.I.E. simply said nothing and nothing was
    recorded as wrong. Reachable from an HTTP 200 with an empty or
    non-conforming body, which is how some gateways answer a rejected key,
    and from a CLI brain that exits 0 having printed nothing.
    """

    async def test_an_empty_brain_hands_over_to_the_next(self):
        reg = registry(
            mute=FakeBrain("mute", text=""),
            backup=FakeBrain("backup", text="Still here."),
            default="mute",
            fallback=["mute", "backup"],
        )
        text, notices = await collect(reg)
        assert text.strip() == "Still here."
        assert notices and "empty" in notices[0].lower()
        assert reg.active == "backup"

    async def test_an_empty_answer_from_the_last_brain_raises(self):
        """An error is better than silence. Returning nothing here would look
        exactly like her choosing not to reply."""
        reg = registry(mute=FakeBrain("mute", text=""), default="mute")
        with pytest.raises(BrainUnavailable, match="empty"):
            await collect(reg)

    async def test_every_brain_empty_still_raises(self):
        reg = registry(
            a=FakeBrain("a", text=""),
            b=FakeBrain("b", text=""),
            default="a",
            fallback=["a", "b"],
        )
        with pytest.raises(BrainUnavailable):
            await collect(reg)

    async def test_a_brain_that_answers_is_untouched(self):
        reg = registry(a=FakeBrain("a", text="fine"), default="a")
        text, notices = await collect(reg)
        assert text.strip() == "fine"
        assert not notices


class TestAccountDeniedIsNotAKeyProblem:
    """Telling someone to re-check a key that is perfectly fine wastes their
    afternoon. Google answers `403 PERMISSION_DENIED: Your project has been
    denied access` when the project is blocked — the key is valid and beside
    the point."""

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "body",
        [
            '{"error":{"code":403,"message":"Your project has been denied access. '
            'Please contact support.","status":"PERMISSION_DENIED"}}',
            '{"error":{"message":"Your account has been suspended"}}',
            '{"error":{"message":"This organization is not authorized to use this model"}}',
        ],
    )
    async def test_it_blames_the_account_not_the_key(self, body):
        from evie.brains.openai_compat import _from_status

        exc = _from_status(403, body)
        assert isinstance(exc, BrainUnavailable), "the chain must still move on"
        assert "not the key" in str(exc)
        assert "stray quotes" not in str(exc)

    async def test_an_ordinary_bad_key_still_says_so(self):
        from evie.brains.openai_compat import _from_status

        exc = _from_status(401, '{"error":{"message":"Invalid API key provided"}}')
        assert "stray quotes" in str(exc)
        assert "not the key" not in str(exc)


class TestTheSystemPromptFollowsTheBrainThatAnswers:
    """The chain is the only place that knows a switch happened.

    Built once by the caller, the prompt describes the brain it *expected* to
    answer. Live, that meant groq being told it was named claude and that the
    vault was its working directory, after claude turned out to be logged out.
    """

    class Spy(FakeBrain):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.seen: list[str] = []

        async def stream(self, prompt, ctx):
            self.seen.append(ctx.system)
            async for chunk in super().stream(prompt, ctx):
                yield chunk

    def _registry(self):
        return BrainRegistry(
            {"a": self.Spy("a", fail=BrainExhausted("out of quota")),
             "b": self.Spy("b")},
            default="a",
            fallback=["a", "b"],
        )

    async def test_each_brain_is_asked_for_its_own(self):
        reg = self._registry()
        ctx = Context(system="original")
        await collect(reg, ctx=ctx, system_for=lambda n: f"you are {n}")
        assert reg.get("a").seen == ["you are a"]
        assert reg.get("b").seen == ["you are b"]

    async def test_it_is_rebuilt_before_the_call_not_after(self):
        """Asserting on ctx afterwards would pass even if the prompt were
        rebuilt too late to reach anyone."""
        reg = self._registry()
        ctx = Context(system="original")
        await collect(reg, ctx=ctx, system_for=lambda n: f"you are {n}")
        assert ctx.system == "you are b"

    async def test_without_it_nothing_changes(self):
        """Every other caller -- the chain phase of the self-test among them --
        passes no hook and must behave exactly as before."""
        reg = self._registry()
        ctx = Context(system="original")
        await collect(reg, ctx=ctx)
        assert ctx.system == "original"
        assert reg.get("b").seen == ["original"]
