"""Client-side pacing for hosted AI providers.

A local llama-server has no rate limit worth the name: it answers as fast as
the GPU allows and queues the rest. A hosted provider meters every request,
and on a free tier the meter is tight enough to matter -- Gemini Flash's free
tier has been ten requests a minute. Sending faster than that is not faster;
it buys a 429 per excess request, and each 429 is a round trip spent learning
nothing.

So the client paces itself under a known ceiling, and still treats a 429 as
information when one arrives anyway (another process sharing the key, a limit
lowered since the preset was written): the provider's own retry delay pushes
the next permitted request back for every caller, not only the one refused.
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import Awaitable, Callable
from time import monotonic

WINDOW_SECONDS = 60.0


class RequestRateLimiter:
    """At most `requests_per_minute` request starts in any rolling minute.

    A sliding window over actual start times, not a token bucket refilled at
    R/60 per second. The two agree on the average and differ on bursts, and the
    burst is what a provider's per-minute counter sees: a bucket would let a
    run that idled for a minute fire R requests at once and then pace, which is
    exactly what a minute-window counter accepts. The window does the same with
    nothing to tune.
    """

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        self.requests_per_minute = requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()
        self._not_before = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one more request may start, then record that it did."""
        async with self._lock:
            while True:
                now = self._clock()
                while self._starts and now - self._starts[0] >= WINDOW_SECONDS:
                    self._starts.popleft()
                wait = self._not_before - now
                if len(self._starts) >= self.requests_per_minute:
                    wait = max(wait, self._starts[0] + WINDOW_SECONDS - now)
                if wait <= 0:
                    self._starts.append(now)
                    return
                await self._sleep(wait)

    def defer(self, seconds: float) -> None:
        """Hold every caller back for `seconds`, because the provider said so."""
        if seconds <= 0:
            return
        self._not_before = max(self._not_before, self._clock() + seconds)


def backoff_delay(
    attempt: int,
    *,
    base: float = 2.0,
    cap: float = 60.0,
    jitter: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with full jitter, for retry `attempt` (1-based).

    Full jitter rather than a fixed doubling: several requests refused in the
    same instant would otherwise all come back in the same later instant and
    be refused together again.
    """
    ceiling = min(cap, base * (2 ** max(0, attempt - 1)))
    return ceiling * (0.5 + 0.5 * jitter())
