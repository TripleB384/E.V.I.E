"""The `evie` command.

Every subcommand below `run` exists to isolate one layer, so when something
breaks you can find out which half is at fault in one command instead of
guessing: `ask` is the brain path with no audio, `say` is the audio path with
no brain, `doctor` is everything that has to be true before either works.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .brains.base import Health

console = Console()
err = Console(stderr=True, style="bold red")

OVERRIDE_STUB = """# Your overrides, layered on top of the shipped defaults.
#
# Anything you set here wins. Anything you leave out keeps following
# evie/defaults/brains.yaml, so new providers and better defaults reach you
# without you doing anything. That is why this file starts empty instead of
# as a copy -- a copy freezes the day you ran `evie init`, and later
# improvements become invisible.
#
# See every available brain and where each setting came from:
#     evie brains list
#
# Examples, all commented out:
#
# default: claude              # who answers when nothing else applies
# quick: groq                  # short questions go here
#
# brains:
#   groq:
#     model: llama-3.3-70b-versatile   # change one field, keep the rest
#   ollama:
#     enabled: false                   # turn one off for good
#   my_server:                         # or add one of your own
#     kind: openai
#     base_url: http://192.168.1.50:8080/v1
#     model: whatever-i-am-running
#     aliases: [homelab]
"""

STUB_CONFIG = """# Your overrides, layered on top of evie/defaults/config.yaml.
# Set only what you want to change; the rest keeps following the defaults.
#
# hotkey: f13                # if Right Option is awkward in your terminal
# voice:
#   kokoro_voice: af_bella   # `evie say --help` lists the engines
#   speed: 1.15
# ears:
#   model: tiny.en           # faster, worse at names
"""


def _is_interactive() -> bool:
    """Whether this is a real terminal session.

    Its own function because the alternative is unpatchable: a CLI test
    runner replaces sys.stdin, so a direct sys.stdin.isatty() call can only
    ever report the runner's pipe.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _fail(message: str, hint: str = "") -> None:
    err.print(f"✗ {escape(message)}")
    if hint:
        console.print(f"  [dim]{escape(hint)}[/]")
    sys.exit(1)


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="evie")
@click.pass_context
def main(ctx: click.Context) -> None:
    """E.V.I.E. — a voice assistant with swappable brains."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


# -- the loop ------------------------------------------------------------


@main.command()
@click.option("--debug", is_flag=True, help="Print per-stage latency after each turn.")
@click.option("--brain", default=None, help="Start on a specific brain.")
def run(debug: bool, brain: str | None) -> None:
    """Start the voice loop. Hold the hotkey to talk."""
    from .assistant import Assistant

    # A global hotkey listener needs a real terminal session. Run from an
    # agent's shell tool, a CI job or a pipe and the key press never arrives --
    # the loop just sits there looking broken. Say so before spending thirty
    # seconds loading Whisper and Kokoro to reach the same silence.
    if not _is_interactive():
        _fail(
            "`evie run` needs an interactive terminal.",
            "Open Terminal or iTerm directly, then: "
            "cd ~/E.V.I.E && source .venv/bin/activate && evie run\n"
            "  (`evie ask` and `evie say` work fine from a script or an agent's shell.)",
        )

    try:
        assistant = Assistant.load()
        if brain:
            assistant.registry.use(brain)
    except Exception as exc:
        _fail(str(exc), "Run `evie doctor` to see what is missing.")

    try:
        from .loop import VoiceLoop
    except ImportError as exc:
        _fail(f"voice extras missing: {exc}", "uv pip install -e '.[voice]'")

    # pynput only warns on stderr and carries on, so without this E.V.I.E.
    # announces herself ready and then ignores every key press.
    from .audio.capture import accessibility_trusted, request_accessibility

    if accessibility_trusted() is False:
        app = os.environ.get("TERM_PROGRAM", "your terminal")
        err.print("✗ macOS has not granted this terminal Accessibility access.")
        console.print(
            f"  [yellow]The hotkey cannot work without it, so there is no point starting.[/]\n"
            f"  Asking macOS to show its prompt now — it adds [bold]{app}[/] to the list "
            f"for you,\n  which is easier than finding the '+' button.\n"
        )
        request_accessibility()
        console.print(
            "  Then: turn the switch [bold]on[/], [bold]quit the terminal with Cmd-Q[/], "
            "reopen it, and run `evie run` again.\n"
            "  [dim]The quit matters: macOS only applies this to a newly launched "
            "process.[/]"
        )
        sys.exit(1)

    try:
        asyncio.run(VoiceLoop(assistant, debug=debug).run())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/]")
    except Exception as exc:
        _fail(str(exc), "Run `evie doctor` for the usual causes.")


# -- isolating the brain path -------------------------------------------


@main.command()
@click.argument("prompt", required=False)
@click.option("--brain", default=None, help="Force a specific brain for this question.")
def ask(prompt: str | None, brain: str | None) -> None:
    """Ask in text, with no audio. Isolates the brain path."""
    from .assistant import Assistant

    text = prompt or sys.stdin.read().strip()
    if not text:
        _fail("nothing to ask", 'Try: evie ask "what is 2 plus 2"')

    async def go() -> None:
        assistant = Assistant.load()
        if brain:
            assistant.registry.use(brain)
        async for kind, chunk in assistant.respond(text):
            if kind == "text":
                console.print(chunk, end="")
            elif kind == "notice":
                console.print(f"\n[yellow]{chunk}[/]")
        console.print()

    try:
        asyncio.run(go())
    except Exception as exc:
        _fail(str(exc))


# -- isolating the audio path -------------------------------------------


@main.command()
@click.argument("text")
@click.option("--engine", default=None, help="kokoro, say, elevenlabs, null")
def say(text: str, engine: str | None) -> None:
    """Speak a line out loud. Isolates the TTS path."""
    from .audio import Speaker
    from .config import Settings
    from .text import for_speech
    from .voice import load_engine

    settings = Settings.load()
    try:
        tts = load_engine(engine or settings.voice.tts, settings.voice)
        samples = tts.synth(for_speech(text))
    except Exception as exc:
        _fail(str(exc))

    with Speaker(tts.sample_rate) as speaker:
        speaker.say(samples, tts.sample_rate)
        speaker.wait(timeout=60)
    console.print(f"[green]✓[/] spoke {len(samples) / tts.sample_rate:.1f}s via {tts.name}")


# -- brains --------------------------------------------------------------


@main.group(invoke_without_command=True)
@click.pass_context
def brains(ctx: click.Context) -> None:
    """Inspect and switch brains."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(brains_list)


@brains.command("list")
def brains_list() -> None:
    """Show every brain, whether it works, and today's usage."""
    from .config import find_config, load_registry

    try:
        registry = load_registry()
    except Exception as exc:
        _fail(str(exc))

    statuses = asyncio.run(registry.health())

    kinds = {"CliBrain": "cli", "OpenAICompatBrain": "http", "EchoBrain": "echo"}
    marks = {
        Health.OK: "[green]ready[/]",
        Health.UNVERIFIED: "[green]installed[/]",
        Health.UNAUTHENTICATED: "[yellow]no key[/]",
        Health.MISSING: "[red]unreachable[/]",
        Health.EXHAUSTED: "[yellow]rate limited[/]",
        Health.UNKNOWN: "[dim]unknown[/]",
    }

    # Which tier sends work here, so the table answers "when does this get
    # used" rather than only "does it exist".
    serves: dict[str, list[str]] = {}
    for tier, brain in registry.tiers.items():
        serves.setdefault(brain, []).append(tier)

    active = [n for n in registry.names() if not registry.is_parked(n)]
    table = Table(title="in the rotation", header_style="bold")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("tools")
    table.add_column("handles")
    table.add_column("status")
    table.add_column("today", justify="right")

    for name in active:
        impl = registry.get(name)
        status = statuses[name]
        flags = ""
        if name == registry.pinned:
            flags = " [bold yellow]📌[/]"
        elif name == registry.active:
            flags = " [bold green]←[/]"
        table.add_row(
            f"{name}{flags}",
            kinds.get(type(impl).__name__, "?"),
            "yes" if impl.agentic else "—",
            ", ".join(serves.get(name, [])) or "[dim]fallback only[/]",
            f"{marks[status.health]} [dim]{escape(status.detail)}[/]",
            registry.headroom(name),
        )
    console.print(table)

    if registry.parked:
        console.print("\n[dim]parked — not routed to, and skipped by the fallback "
                      "chain:[/]")
        for name, reason in registry.parked.items():
            console.print(f"  [dim]{name:12}[/] [yellow]{escape(reason)}[/]")

    console.print()
    if registry.pinned:
        console.print(
            f"[yellow]pinned to {registry.pinned}[/] — say [bold]\"auto\"[/] to let "
            f"her choose again"
        )
    else:
        order = " · ".join(
            f"{tier}→{registry.for_tier(tier)}" for tier in registry.TIER_ORDER
        )
        console.print(f"[dim]{order}[/]")
    # The effective chain, not the configured one: showing a parked brain
    # here would contradict the table directly above.
    effective = [n for n in registry.fallback if not registry.is_parked(n)]
    console.print(f"[dim]fallback {' → '.join(effective) or '—'}[/]")
    layers = getattr(registry, "sources", None) or [find_config("brains.yaml")]
    console.print("[dim]config: " + " + ".join(str(p) for p in layers) + "[/]")


def _edit_overrides(change) -> Path:
    """Apply `change` to your own override file, creating it if needed.

    Always your file, never the shipped defaults. Writing to whichever config
    happened to be found first would edit evie/defaults/brains.yaml inside the
    repo -- turning up in `git status` and getting clobbered by the next pull.
    """
    import yaml

    from .config import user_dir

    path = user_dir() / "brains.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _resolve_or_fail(name: str) -> str:
    from .config import load_registry

    try:
        return load_registry().resolve(name)
    except Exception as exc:
        _fail(str(exc), "Run `evie brains list` to see the options.")


@brains.command("use")
@click.argument("name")
def brains_use(name: str) -> None:
    """Set the default brain."""
    target = _resolve_or_fail(name)

    def change(data):
        data["default"] = target

    path = _edit_overrides(change)
    console.print(f"[green]✓[/] default brain is now [bold]{target}[/]")
    console.print(f"  [dim]{path}[/]")


@brains.command("disable")
@click.argument("name")
def brains_disable(name: str) -> None:
    """Hide a brain you do not want, without editing YAML by hand."""
    target = _resolve_or_fail(name)

    def change(data):
        data.setdefault("brains", {}).setdefault(target, {})["enabled"] = False

    path = _edit_overrides(change)
    console.print(f"[green]✓[/] [bold]{target}[/] is off")
    console.print(f"  [dim]{path} — `evie brains enable {target}` to undo[/]")


@brains.command("enable")
@click.argument("name")
def brains_enable(name: str) -> None:
    """Turn a brain back on."""
    import yaml

    from .config import PACKAGE_DEFAULTS, user_dir

    # The brain may be off precisely because it is absent from the merged
    # registry, so resolve against the shipped roster rather than the live one.
    shipped = yaml.safe_load((PACKAGE_DEFAULTS / "brains.yaml").read_text()) or {}
    known = set(shipped.get("brains") or {})

    # The override file is optional -- `evie init` writes an empty one, and a
    # user who deletes it is in a perfectly good state. Reading it blindly
    # crashed with FileNotFoundError for exactly that case.
    override = user_dir() / "brains.yaml"
    if override.is_file():
        local = yaml.safe_load(override.read_text()) or {}
        known |= set(local.get("brains") or {})

    if name not in known:
        _fail(
            f"no brain called {name!r}. Known: {', '.join(sorted(known))}",
            "Run `evie brains list` to see what is configured.",
        )

    def change(data):
        data.setdefault("brains", {}).setdefault(name, {})["enabled"] = True

    path = _edit_overrides(change)
    console.print(f"[green]✓[/] [bold]{name}[/] is on")
    console.print(f"  [dim]{path}[/]")


PHASES = ("switch", "classify", "chain", "reach")



@brains.command("test")
@click.option("--only", type=click.Choice(PHASES), default=None,
              help="Run one check. `switch` needs no keys and no network.")
@click.option("--brain", "one", default=None,
              help="Test one brain: its reach and how its provider rejects a bad key.")
@click.option("--skip", multiple=True, help="Leave a brain out. Repeatable.")
@click.option("--deep", is_flag=True,
              help="Walk every position in the fallback chain, not just the head.")
def brains_test(only: str | None, one: str | None, skip: tuple[str, ...], deep: bool) -> None:
    """Make real calls and check the brains, the chain and switching.

    Unlike `brains list`, which only pings /models or looks for a binary, this
    runs an actual inference on every brain -- roughly one request each. Cheap
    everywhere except `claude`, which boots a whole Claude Code session per
    call: 5-11s and real plan allowance. `--skip claude` leaves it out.
    """
    from .config import PACKAGE_DEFAULTS, load_registry
    from . import selftest

    try:
        registry = load_registry()
    except Exception as exc:
        _fail(str(exc))

    for name in (*skip, *( [one] if one else [] )):
        try:
            registry.resolve(name)
        except Exception as exc:
            _fail(str(exc), "Run `evie brains list` to see what is configured.")

    if only:
        wanted = [only]
    elif one:
        # `switch` and `chain` are properties of the whole roster, not of one
        # brain, so naming a brain means the two checks that are about it.
        wanted = ["classify", "reach"]
    else:
        wanted = list(PHASES)
    results: list = []

    async def go() -> None:
        if "switch" in wanted:
            results.extend(
                selftest.switching(lambda: load_registry(PACKAGE_DEFAULTS / "brains.yaml"))
            )
        if "classify" in wanted:
            results.extend(await selftest.classify(registry, only=one, skip=skip))
        if "chain" in wanted:
            results.extend(await selftest.chain(registry, deep=deep, skip=skip))
        if "reach" in wanted:
            results.extend(await selftest.reach(registry, only=one, skip=skip))

    titles = {
        "switch": "switching — is a brain swap still free?",
        "classify": "classify — is a real provider rejection survivable?",
        "chain": "chain — does she move on when a brain gives out?",
        "reach": "reach — can every brain actually answer?",
    }

    try:
        asyncio.run(go())
    except Exception as exc:
        _fail(str(exc))

    for phase in wanted:
        rows = [r for r in results if r.phase == phase]
        if not rows:
            continue
        table = Table(title=titles[phase], header_style="bold", title_justify="left")
        table.add_column("")
        table.add_column("what")
        table.add_column("result")
        table.add_column("took", justify="right")
        for r in rows:
            mark = "[dim]–[/]" if r.skipped else ("[green]✓[/]" if r.ok else "[red]✗[/]")
            style = "dim" if r.skipped else ("" if r.ok else "red")
            table.add_row(
                mark,
                escape(r.name),
                f"[{style}]{escape(r.detail)}[/]" if style else escape(r.detail),
                f"{r.seconds:.1f}s" if r.seconds else "",
            )
        console.print(table)
        console.print()

    bad = selftest.failures(results)
    passed = sum(1 for r in results if r.ok and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    summary = f"{passed} passed, {len(bad)} failed"
    if skipped:
        summary += f", {skipped} skipped [dim](not configured, or not testable here)[/]"

    if bad:
        console.print(f"[red]{summary}[/]")
        console.print("\n[dim]A brain green in `brains list` and red here means the "
                      "health check is overstating readiness — it never ran an "
                      "inference.[/]")
        sys.exit(1)
    console.print(f"[green]{summary}[/]")


# -- memory --------------------------------------------------------------


@main.group(invoke_without_command=True)
@click.pass_context
def memory(ctx: click.Context) -> None:
    """Inspect and back up what she remembers."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(memory_status)


@memory.command("status")
def memory_status() -> None:
    """Show what is in the vault, how big it is, and whether it is backed up."""
    from .config import Settings
    from .memory import Vault, VaultGit, stats

    settings = Settings.load()
    vault = Vault(settings.vault)
    if not vault.exists:
        _fail(f"no vault at {vault.root}", "Run `evie init`.")

    info = stats(vault)
    kb = info["bytes"] / 1024
    console.print(f"[bold]{vault.root}[/]")
    console.print(
        f"  {info['files']} notes · {info['days']} days logged · "
        f"[bold]{kb:.0f} KB[/] total"
    )
    if info["first"]:
        console.print(f"  from {info['first']} to {info['last']}")

    # Put the size in terms anyone can judge.
    if kb < 5000:
        console.print(
            f"  [dim]For scale: a decade at this rate is about "
            f"{kb * 10 / max(info['days'], 1) * 365 / 1024:.0f} MB. "
            f"Disk is not your constraint.[/]"
        )

    git = VaultGit(vault)
    if not git.initialized:
        console.print(
            "\n  [yellow]Not backed up.[/] One copy, one machine. To fix, make an "
            "empty\n  [bold]private[/] repo on GitHub and run:\n"
            "    [bold]evie memory setup git@github.com:you/evie-vault.git[/]"
        )
    elif remote := git.remote():
        console.print(f"\n  [green]backed up[/] → {remote}")
    else:
        console.print("\n  [yellow]local git only[/] — no remote set")


@memory.command("setup")
@click.argument("remote")
def memory_setup(remote: str) -> None:
    """Point the vault at a private git remote for free, versioned backup."""
    from .config import Settings
    from .memory import GitError, Vault, VaultGit

    vault = Vault(Settings.load().vault)
    if not vault.exists:
        _fail(f"no vault at {vault.root}", "Run `evie init`.")
    try:
        git = VaultGit(vault)
        git.setup(remote)
        console.print(f"[green]✓[/] {vault.root} → {remote}")
        console.print("  Now run [bold]evie memory sync[/] to push what she already has.")
    except GitError as exc:
        _fail(str(exc))


@memory.command("sync")
@click.option("--message", "-m", default=None, help="Commit message.")
def memory_sync(message: str | None) -> None:
    """Commit and push the vault."""
    from .config import Settings
    from .memory import GitError, Vault, VaultGit

    vault = Vault(Settings.load().vault)
    if not vault.exists:
        _fail(f"no vault at {vault.root}", "Run `evie init`.")
    try:
        console.print(f"[green]✓[/] {VaultGit(vault).sync(message)}")
    except GitError as exc:
        _fail(str(exc), "Check the remote exists and you can push to it.")


# -- setup and diagnosis -------------------------------------------------


@main.command()
@click.option("--owner", default=None, help="Your name, for EVIE.md.")
def init(owner: str | None) -> None:
    """Create the vault and copy the starter config into ~/.evie."""
    from .config import Settings, user_dir
    from .memory import Vault

    home = user_dir()
    home.mkdir(parents=True, exist_ok=True)
    # Deliberately a stub, not a copy of the defaults. A copy wins over the
    # shipped config forever, so it silently freezes your setup on the day you
    # ran this -- new brains never appear, and an API key you export has
    # nothing to read it.
    for name, stub in (("brains.yaml", OVERRIDE_STUB), ("config.yaml", STUB_CONFIG)):
        dst = home / name
        if not dst.exists():
            dst.write_text(stub)
            console.print(f"[green]✓[/] wrote {dst} [dim](empty — overrides only)[/]")

    settings = Settings.load()
    vault = Vault(settings.vault).ensure(owner or os.environ.get("USER", "you"))
    console.print(f"[green]✓[/] vault at {vault.root}")
    console.print("\nNext: [bold]evie doctor[/]")


@main.command()
@click.option("--fix", is_flag=True, help="Download any missing model files.")
def doctor(fix: bool) -> None:
    """Check everything that has to be true before `evie run` works."""
    import shutil as _shutil

    problems = 0

    def check(label: str, ok: bool, detail: str = "", hint: str = "") -> bool:
        nonlocal problems
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"{mark} {label}" + (f" [dim]{escape(detail)}[/]" if detail else ""))
        if not ok:
            problems += 1
            if hint:
                console.print(f"    [yellow]{escape(hint)}[/]")
        return ok

    console.print("[bold]config[/]")
    try:
        from .config import Settings, find_config, load_registry

        settings = Settings.load()
        registry = load_registry()
        source = find_config("brains.yaml")
        check("brains.yaml", True, f"{len(registry.names())} brains, default {registry.active}")
        console.print(f"  [dim]loaded from {source}[/]")
        if source.parent == Path.cwd():
            console.print(
                "  [yellow]This is a local override, not the shipped config. "
                "Delete it to pick up updates from git.[/]"
            )
    except Exception as exc:
        check("brains.yaml", False, "", str(exc))
        console.print("\n[red]Cannot continue without config.[/] Run `evie init`.")
        sys.exit(1)

    console.print("\n[bold]brains[/]")
    statuses = asyncio.run(registry.health())
    usable = [n for n, s in statuses.items() if s.usable]
    for name, status in statuses.items():
        console.print(
            f"  {'[green]✓[/]' if status.usable else '[yellow]-[/]'} {name} "
            f"[dim]{escape(status.detail)}[/]"
        )
    check("at least one brain usable", bool(usable), f"{len(usable)} ready",
          "Install a CLI (claude/gemini/codex) or set an API key env var.")

    console.print("\n[bold]python[/]")
    version = ".".join(str(n) for n in sys.version_info[:3])
    check(
        f"python {version}",
        sys.version_info >= (3, 11),
        "",
        "Kokoro needs onnxruntime, which only ships wheels for 3.11+. "
        "macOS ships 3.9. Recreate the venv with: uv venv --python 3.12",
    )

    console.print("\n[bold]python packages[/]")
    for mod, extra in (
        ("sounddevice", "audio in/out"),
        ("numpy", "audio buffers"),
        ("faster_whisper", "speech to text"),
        ("kokoro_onnx", "text to speech"),
        ("pynput", "push-to-talk hotkey"),
    ):
        try:
            __import__(mod)
            check(mod, True, extra)
        except ImportError:
            check(mod, False, extra, "uv pip install -e '.[voice]'")

    console.print("\n[bold]models[/]")
    try:
        from .voice.kokoro import DOWNLOADS, missing_files

        missing = missing_files()
        if missing and fix:
            _download(DOWNLOADS, missing)
            missing = missing_files()
        check("kokoro voice model", not missing,
              "" if not missing else ", ".join(p.name for p in missing),
              "Run `evie doctor --fix` to download (~350MB).")
    except Exception as exc:
        check("kokoro voice model", False, "", str(exc))

    from .ears.stt import MODEL_DIR as WHISPER_DIR

    console.print(
        f"  [dim]whisper model cache: {WHISPER_DIR} "
        f"({'present' if WHISPER_DIR.exists() else 'downloads on first run, ~500MB'})[/]"
    )

    console.print("\n[bold]audio devices[/]")
    try:
        import sounddevice as sd

        ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        outs = [d for d in sd.query_devices() if d["max_output_channels"] > 0]
        check("microphone", bool(ins), ins[0]["name"] if ins else "",
              "Grant your terminal Microphone access in System Settings > Privacy & Security.")
        check("speakers", bool(outs), outs[0]["name"] if outs else "")
    except Exception as exc:
        check("audio devices", False, "", f"{exc}")

    if not _is_interactive():
        console.print(
            "\n[yellow]Not a terminal session.[/] Checks below are unreliable here: this\n"
            "  shell doesn't read your ~/.zshrc, so exported API keys look unset, and\n"
            "  `evie run` can't receive hotkeys at all. Re-run in Terminal or iTerm."
        )

    if sys.platform == "darwin":
        console.print("\n[bold]macOS permissions[/]")
        try:
            from .audio.capture import accessibility_trusted

            trusted = accessibility_trusted()
        except Exception:
            trusted = None

        if trusted is None:
            console.print(
                "  [dim]Could not query Accessibility. Grant it to your terminal app "
                "(not Python) if the hotkey does nothing.[/]"
            )
        else:
            check(
                "Accessibility (global hotkey)",
                trusted,
                "granted to this terminal" if trusted else "",
                "Run `evie run` — it will ask macOS to add this terminal to the list, "
                "then turn the switch on, quit with Cmd-Q, and reopen.",
            )
        check("`say` fallback voice", _shutil.which("say") is not None)

    console.print("\n[bold]vault[/]")
    from .memory import Vault

    vault = Vault(settings.vault)
    check("memory vault", vault.exists, str(vault.root), "Run `evie init`.")

    console.print()
    if problems:
        console.print(f"[yellow]{problems} thing(s) need attention.[/]")
        sys.exit(1)
    console.print("[bold green]All clear. Run `evie run`.[/]")


def _download(downloads: dict[Path, str], missing: list[Path]) -> None:
    import httpx

    for path in missing:
        url = downloads[path]
        path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"  downloading {path.name} …")
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
            resp.raise_for_status()
            with path.open("wb") as fh:
                for block in resp.iter_bytes(1 << 20):
                    fh.write(block)
        console.print(f"  [green]✓[/] {path.name}")


if __name__ == "__main__":
    main()
