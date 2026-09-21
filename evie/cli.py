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

    table = Table(title="brains", header_style="bold")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("tools")
    table.add_column("status")
    table.add_column("today", justify="right")
    table.add_column("also called", style="dim")

    marks = {
        Health.OK: "[green]ready[/]",
        Health.UNVERIFIED: "[green]installed[/]",
        Health.UNAUTHENTICATED: "[yellow]not authenticated[/]",
        Health.MISSING: "[red]missing[/]",
        Health.EXHAUSTED: "[yellow]rate limited[/]",
        Health.UNKNOWN: "[dim]unknown[/]",
    }
    kinds = {"CliBrain": "cli", "OpenAICompatBrain": "http", "EchoBrain": "echo"}
    for name in registry.names():
        impl = registry.get(name)
        status = statuses[name]
        active = " [bold green]←[/]" if name == registry.active else ""
        table.add_row(
            f"{name}{active}",
            kinds.get(type(impl).__name__, "?"),
            "yes" if impl.agentic else "no",
            f"{marks[status.health]} [dim]{escape(status.detail)}[/]",
            registry.headroom(name),
            ", ".join(registry.aliases_for(name)[:4]),
        )

    console.print(table)
    console.print(
        f"[dim]default {registry.active} · quick {registry.quick or '—'} · "
        f"fallback {' → '.join(registry.fallback) or '—'}[/]"
    )
    console.print(f"[dim]config: {find_config('brains.yaml')}[/]")


@brains.command("use")
@click.argument("name")
def brains_use(name: str) -> None:
    """Set the default brain (writes it to brains.yaml)."""
    import yaml

    from .config import find_config, load_registry

    try:
        registry = load_registry()
        target = registry.resolve(name)
    except Exception as exc:
        _fail(str(exc), "Run `evie brains list` to see the options.")

    path = find_config("brains.yaml")
    data = yaml.safe_load(path.read_text())
    data["default"] = target
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    console.print(f"[green]✓[/] default brain is now [bold]{target}[/] ({path})")


# -- setup and diagnosis -------------------------------------------------


@main.command()
@click.option("--owner", default=None, help="Your name, for EVIE.md.")
def init(owner: str | None) -> None:
    """Create the vault and copy the starter config into ~/.evie."""
    import shutil

    from .config import PACKAGE_DEFAULTS, USER_DIR, Settings
    from .memory import Vault

    USER_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("brains.yaml", "config.yaml"):
        src = PACKAGE_DEFAULTS / name
        dst = USER_DIR / name
        if src.is_file() and not dst.exists():
            shutil.copy(src, dst)
            console.print(f"[green]✓[/] wrote {dst}")

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
