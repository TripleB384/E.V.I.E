# E.V.I.E.

A voice-first personal assistant with **swappable brains**.

Hold a key, talk, and hear an answer back. Under the hood she can run on Claude,
ChatGPT, Gemini, Groq, a local model, or anything else that speaks the OpenAI
chat schema — and you can switch between them mid-conversation, out loud,
without losing the thread.

```
you:  Evie, what's due in CS 320 this week?
evie: The compiler project, Thursday at midnight. You've got the lexer done.
you:  Evie, switch to Gemini.
evie: Switched to gemini_cli.
you:  What was I just asking about?
evie: Your CS 320 deadline — the compiler project on Thursday.
```

## Why swappable brains

Most "build your own Jarvis" projects hardwire one provider. That's a bad deal
for two reasons: you're locked to one vendor's pricing, and you can't use the
free tiers you already have sitting there.

E.V.I.E. treats a brain as a thing that turns a prompt into a stream of text.
Two implementations cover every provider:

| Kind | What it is | Cost | Can it *do* things? |
|---|---|---|---|
| `cli` | `claude -p`, `codex exec`, `gemini -p` | Covered by a subscription or free tier | **Yes** — files, shell, web, MCP |
| `openai` | Any OpenAI-compatible endpoint | Free tier or pennies | No — conversation only |

Subscription-backed CLIs are the trick: they authenticate with the plan you
already pay for, so a request costs nothing at the margin, *and* they carry a
full agent loop so she can actually do your work rather than just describe it.

When a free tier runs dry, she walks the fallback chain and keeps going:

> *"Claude is rate limited. Switching to gemini_cli."*

## Install

A Mac (Apple Silicon recommended; Linux works, the macOS permission notes don't
apply) and **Python 3.11 or newer**.

That version floor is not a preference. The core of E.V.I.E. runs fine on 3.9 --
all tests pass there -- but Kokoro needs `onnxruntime`, which only publishes
wheels for cp311 and up. macOS ships Python 3.9, so you need a newer one, and
`uv` is the least painful way to get it.

**Run these one at a time.** Each has a checkpoint; don't move on without it.

**1. Install `uv`** (skip if `uv --version` already works):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Put it on your PATH.** The installer does *not* do this for the shell
you're already in, and the next command just fails with `command not found`:

```bash
source $HOME/.local/bin/env
```

*Checkpoint:* `uv --version` prints a version.

**3. Clone and create the environment:**

```bash
git clone https://github.com/TripleB384/E.V.I.E && cd E.V.I.E
```

```bash
uv venv --python 3.12
```

```bash
source .venv/bin/activate
```

*Checkpoint:* your prompt now starts with `(.venv)`.

**4. Install.** Note `voice,dev` -- `dev` is what provides `pytest`, so
installing only `[voice]` leaves you unable to run the tests:

```bash
uv pip install -e ".[voice,dev]"
```

Takes a minute or two; onnxruntime and the Whisper libraries are large.

*Checkpoint:*

```bash
pytest
```

385 tests pass, with no hardware and no credentials.

**5. Set up and go:**

```bash
evie init          # creates ~/.evie and your vault
evie doctor --fix  # checks everything, downloads the voice model (~350MB)
evie run           # hold Right Option and talk
```

<details>
<summary>Without <code>uv</code></summary>

You need a Python 3.11+ from somewhere else -- `brew install python@3.12`, or
python.org. Then `python3.12 -m venv .venv && source .venv/bin/activate &&
pip install -e ".[voice,dev]"`. Stock macOS Python 3.9 will not work for the
voice stack, for the onnxruntime reason above.

</details>

### Authenticating brains

E.V.I.E. ships with **no credentials** and never asks for your password. Each
brain authenticates itself, the same way you'd use it from a terminal:

| Brain | How |
|---|---|
| `claude` | `npm i -g @anthropic-ai/claude-code && claude` → `/login` |
| `gemini_cli` | `npm i -g @google/gemini-cli && gemini` → sign in (free, ~1,000 req/day) |
| `codex` | `npm i -g @openai/codex && codex` → sign in |
| `groq` | free key from [console.groq.com](https://console.groq.com) → `export GROQ_API_KEY=…` |
| `ollama` | `ollama pull qwen3:8b` — fully offline |

Turn a brain on by setting `enabled: true` in `brains.yaml`.

### Where keys live

**In your environment, and nowhere else.** `brains.yaml` records the *name* of
an environment variable, never its value:

```yaml
api_key_env: GROQ_API_KEY    # the name -- safe to commit
```

E.V.I.E. reads `os.environ` at call time and never writes a credential to disk.
That is what makes this config publishable, and it is enforced by a test:
`TestRepoHygiene` scans every tracked file for credential shapes on each run.

Put your keys in `~/.zshrc` so they persist across terminals:

```bash
echo 'export GROQ_API_KEY=your_key_here' >> ~/.zshrc
```

If a key ever reaches a chat, a screenshot, or a commit, rotate it rather than
assessing the blast radius — all of these providers issue free replacements in
about a minute.

## Using it

```bash
evie run --debug         # the voice loop, with per-stage latency
evie ask "..."           # text only — isolates the brain path
evie say "testing"       # audio only — isolates the TTS path
evie brains list         # health + today's usage for every brain
evie brains use gemini   # change the default
evie brains test         # make real calls — does any of this actually work?
evie doctor              # what's missing and how to fix it
```

`brains list` is cheap and optimistic: it pings `/models` or looks for a
binary, so a CLI that is installed but not logged in still shows green.
`brains test` is the one that asks for real. It runs an actual inference on
every brain, hands each provider a deliberately invalid key to check that a
rejection is survivable rather than fatal, forces the head of the fallback
chain to fail and confirms the next brain picks up, and replays the voice
switch commands to confirm they still cost no model call.

```bash
evie brains test --only switch   # needs no keys and no network
evie brains test --skip claude   # leave out the slow, plan-spending one
```

Roughly one request per brain, so it is nearly free — except `claude`, which
boots a whole Claude Code session per call (5–11s). It exits non-zero on any
failure.

Say these to her and no model is ever called — the swap is instant and free:

- *"switch to Gemini"* / *"use the fast one"* / *"go local"*
- *"who are you running on?"*
- *"list your brains"*
- *"go back"*

## Memory

Her memory is a folder of markdown files, not a vector database. You can read
it, fix it, and open it in Obsidian. Agentic brains run with the vault as their
working directory, so they search it with real file tools and there's no RAG
layer to go stale.

```
~/EVIE/vault/
  EVIE.md              who she is, what she knows about you
  daily/2026-09-20.md  rolling log
  classes/  business/  people/  projects/
```

## Canvas

```bash
evie canvas setup <host>   # point it at your school, once
evie canvas status         # check the URL and token, write nothing
evie canvas sync           # deadlines into the vault as markdown
```

Point it at your school — paste the URL straight from your browser bar, the
scheme and any `?login_success=1` are trimmed for you:

```bash
evie canvas setup yourdistrict.instructure.com
```

The token is **not** in config. Generate one in a browser — Account →
Settings → "+ New Access Token" — and put it in your shell as
`CANVAS_API_TOKEN`. Canvas shows it exactly once and it is password-
equivalent, so treat it like any other key here: environment only, never in
the repo. Some school districts disable token generation entirely; if the
"Approved Integrations" section is missing, that is the answer.

Sync writes `classes/upcoming.md` and `classes/<course>/deadlines.md`. Only
the block between the `<!-- evie:canvas -->` markers is replaced, so notes you
add to those files by hand survive.

**This is not an MCP server, deliberately.** MCP tools only reach agentic
brains, so every "what's due Thursday" would boot a Claude Code session —
5–11s and real plan allowance — to answer something the free brain does in
0.3s. Syncing into the vault means the fast brain can answer it, it works
offline, and it survives a Canvas outage.

If a sync writes nothing, `evie canvas sync --shape` prints what Canvas
actually sent.

## The stack

| Layer | Choice | Why |
|---|---|---|
| Wake | Push-to-talk (`pynput`) | Zero false triggers. A wake word is a later nicety, not a prerequisite. |
| Ears | `faster-whisper` `small.en` int8 | ~1s on Apple Silicon, and much better at names than `base`. |
| Brain | Swappable — see above | |
| Voice | Kokoro-82M via `kokoro-onnx` | Free, unlimited, offline, ~10× realtime on M-series. |

The thing that makes it feel fast isn't any one of those — it's that synthesis
starts on the **first complete sentence** instead of the last. See
`evie/text.py`.

Target budget, visible in `evie run --debug`:

| Stage | Target |
|---|---|
| Speech to text | < 1.0s |
| First token | < 1.5s |
| First audio | < 0.4s |
| **To first sound** | **< 3s** |

## Known gotchas

- **macOS global hotkeys** need Input Monitoring *and* Accessibility granted to
  your **terminal app**, not to Python. This is the number one reason
  push-to-talk silently does nothing. `evie doctor` checks it.
- **First run downloads ~850MB** of models (Whisper + Kokoro), once.
- **Free tiers move.** Groq dropped Llama from its free tier in Aug 2026. The
  fallback chain is the mitigation — assume any given tier will vanish.
- **Terms of service.** Running `claude -p` against *your own* subscription for
  *your own* assistant is normal use. Anthropic's Agent SDK terms prohibit
  *offering* claude.ai login or subscription rate limits to third parties — so
  E.V.I.E. ships with no credentials and every user authenticates their own CLIs.

## Development

```bash
uv pip install -e ".[voice,dev]"
pytest                                  # no audio hardware needed
evie ask "hello" --brain echo           # end-to-end with no model at all
```

The test suite covers the brain seam, the router, the fallback chain and the
sentence chunker — everything except the parts that need a microphone.

## License

GPL-2.0
