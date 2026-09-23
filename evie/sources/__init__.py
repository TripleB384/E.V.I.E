"""Where facts come from that E.V.I.E. did not learn by talking.

A source fetches from somewhere external and writes markdown into the vault.
It deliberately does not become a tool a brain calls: the vault is already
shared between every brain, so a fact written here is answerable by the fast
free one. Wiring Canvas in as an MCP server instead would have meant every
"what's due Thursday" booting a Claude Code session -- 5-11s and real plan
allowance -- to answer something groq does in 0.3s.
"""

from .canvas import Canvas, CanvasError, Deadline

__all__ = ["Canvas", "CanvasError", "Deadline"]
