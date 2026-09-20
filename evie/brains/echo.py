"""A brain with no provider behind it.

Exists so the voice loop, the router and the fallback chain can be exercised
end to end with no network, no API key and no subscription. `evie doctor` and
the test suite lean on it.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import Brain, BrainStatus, Context, Health


class EchoBrain:
    name = "echo"
    agentic = False

    def __init__(self, name: str = "echo", delay: float = 0.0) -> None:
        self.name = name
        self.delay = delay

    async def stream(self, prompt: str, ctx: Context) -> AsyncIterator[str]:
        for word in f"You said: {prompt}".split(" "):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield word + " "

    async def health(self) -> BrainStatus:
        return BrainStatus(Health.OK, "always available")


_: type[Brain] = EchoBrain
