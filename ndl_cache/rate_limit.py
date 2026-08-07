"""
Client-side request pacing for the Nasdaq Data Link API.

Nasdaq publishes per-window call limits and enforces them by temporarily
disabling the account, not by throttling. Retrying a rejected request after a
fraction of a second therefore makes things worse: the account is already
suspended and every extra call extends the suspension. The defence has to be
proactive pacing plus a long cooldown once rejected.

Published limits for the Tables API (/api/v3/datatables), which is the surface
this client uses. These differ from the real-time REST API, whose 100
requests/second limit does not apply here:

    anonymous            20 calls / 10 min, 50 / day, shared globally
    authenticated        300 / 10 s, 2,000 / 10 min, 50,000 / day,
                         and a CONCURRENCY LIMIT OF ONE
    premium subscriber   5,000 / 10 min, 720,000 / day

The concurrency limit is not a rate at all: an authenticated non-premium key
may have exactly one call in flight plus one queued. Pacing cannot satisfy it,
only a semaphore can. Premium access is granted per dataset, so a key can be
premium for one table and authenticated for another.

Defaults track the authenticated tier rather than the premium tier, and below
even that, because the same API key is shared by other processes and machines
so this client cannot assume it owns the quota.

Override with NDL_RATE_LIMIT, a comma-separated list of ``calls/seconds``,
and NDL_MAX_CONCURRENCY:

    NDL_RATE_LIMIT=5/1,2000/600      # 5 per second and 2000 per 10 minutes
    NDL_RATE_LIMIT=                  # empty disables pacing entirely
    NDL_MAX_CONCURRENCY=2            # requests in flight
"""
import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """At most ``calls`` requests in any rolling ``seconds`` interval."""
    calls: int
    seconds: float


# Defaults track the authenticated tier, since premium is granted per dataset
# and a key can be premium for one table and not another. Headroom is modest
# rather than generous: setting these far below the limit throttles harder than
# the server would. 1,000/10min, for instance, works out to 1.7 requests per
# second, which is slower than issuing them one at a time.
#
# For a key that is premium across the tables in use, raise them:
#   NDL_RATE_LIMIT=280/10,4500/600,650000/86400
DEFAULT_WINDOWS = (
    Window(calls=250, seconds=10),       # 83% of the published 300/10s
    Window(calls=1800, seconds=600),     # 90% of the published 2,000/10min
    Window(calls=45000, seconds=86400),  # 90% of the published 50,000/day
)

# Two outstanding requests: one executing plus one queued, which is exactly
# what the documented concurrency limit allows.
#
# Two is the throughput-optimal point under that rule, not merely the safe one.
# With a single outstanding request the server sits idle for a full client
# round trip between calls, so throughput is 1/(service + round trip). Keeping
# one call queued means the server starts the next the instant the previous
# finishes, giving 1/service and hiding the round trip.
#
# Measured against the live Tables API, with per-request latency flat at
# ~610 ms throughout:
#
#     concurrency 1    1.76 req/s
#     concurrency 2    2.97 req/s     +69%
#     concurrency 3    4.28 req/s     +44%
#     concurrency 4    4.50 req/s      +5%
#
# Flat latency with rising throughput confirms the gain comes from hiding the
# round trip rather than from overloading the server. Three is the knee, and
# was not rejected, so the documented concurrency limit appears unenforced for
# premium keys. Two remains the default because premium is per dataset and the
# limit does apply to tables a key is not subscribed to.
DEFAULT_MAX_CONCURRENCY = 2

# How long to stand down after the server rejects a request. Nasdaq disables
# the account for minutes, so sub-second backoff is useless.
DEFAULT_PENALTY_SECONDS = 60.0
MAX_PENALTY_SECONDS = 900.0


def parse_windows(spec):
    """Parse a ``calls/seconds,calls/seconds`` specification."""
    windows = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            calls, seconds = part.split('/')
            windows.append(Window(calls=int(calls), seconds=float(seconds)))
        except ValueError:
            raise ValueError(
                f'NDL_RATE_LIMIT entry {part!r} is malformed; '
                f'expected "calls/seconds" such as "5/1,2000/600"')
    return tuple(windows)


def parse_retry_after(headers):
    """Seconds from a Retry-After header, or None if absent or unparseable."""
    value = None
    for key in headers:
        if key.lower() == 'retry-after':
            value = headers[key]
            break
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # The HTTP-date form is legal but Nasdaq does not use it, and guessing
        # wrong would under-wait. Fall back to the caller's default.
        return None


class RateLimiter:
    """
    Paces outgoing requests against one or more rolling windows.

    Shared by every concurrent fetcher on a client, so a rejection observed by
    one worker stops all of them.
    """

    def __init__(self, windows=None, clock=None, sleep=None,
                 penalty_seconds=DEFAULT_PENALTY_SECONDS,
                 max_concurrency=None):
        if windows is None:
            spec = os.environ.get('NDL_RATE_LIMIT')
            windows = DEFAULT_WINDOWS if spec is None else parse_windows(spec)
        if max_concurrency is None:
            max_concurrency = int(os.environ.get(
                'NDL_MAX_CONCURRENCY', DEFAULT_MAX_CONCURRENCY))
        if max_concurrency < 1:
            raise ValueError(
                f'NDL_MAX_CONCURRENCY must be at least 1, got {max_concurrency}')
        self.windows = tuple(windows)
        self.max_concurrency = max_concurrency
        self.penalty_seconds = penalty_seconds
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._calls = deque()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_concurrency)

    async def sleep(self, seconds):
        """
        Wait, using this limiter's clock.

        Retry backoff goes through here so that every wait in the client is
        driven by one injectable clock. Reaching the real asyncio.sleep for
        some waits and not others makes tests both slow and unable to assert
        on how long the client actually waited.
        """
        await self._sleep(seconds)

    def in_flight(self):
        """
        Context manager holding one concurrency slot for the duration of a
        request. Separate from acquire(), which only paces call starts: the
        server's concurrency limit counts calls that are open, not calls that
        were started.
        """
        return _InFlight(self._slots)

    @property
    def _horizon(self):
        return max((w.seconds for w in self.windows), default=0.0)

    def penalize(self, seconds=None):
        """
        Stand down for a while after a server-side rejection.

        Never shortens an existing cooldown: if another worker already
        triggered a longer one, that one still stands.
        """
        seconds = min(self.penalty_seconds if seconds is None else seconds,
                      MAX_PENALTY_SECONDS)
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)

    async def acquire(self):
        """Block until it is safe to issue one request, then record it."""
        while True:
            async with self._lock:
                now = self._clock()

                if now < self._blocked_until:
                    wait = self._blocked_until - now
                else:
                    wait = self._wait_for_windows(now)
                    if wait <= 0:
                        self._calls.append(now)
                        self._prune(now)
                        return

            # Sleep outside the lock so other workers can queue behind us
            # rather than serializing their waits end to end.
            await self._sleep(wait)

    def _wait_for_windows(self, now):
        """Seconds until a request would fit in every window; 0 if it fits now."""
        wait = 0.0
        for window in self.windows:
            cutoff = now - window.seconds
            in_window = [t for t in self._calls if t > cutoff]
            if len(in_window) >= window.calls:
                # The oldest call in this window has to age out first.
                oldest = in_window[-window.calls]
                wait = max(wait, oldest + window.seconds - now)
        return wait

    def _prune(self, now):
        horizon = self._horizon
        while self._calls and self._calls[0] <= now - horizon:
            self._calls.popleft()


class _InFlight:
    """Holds one concurrency slot for as long as a request is open."""

    def __init__(self, semaphore):
        self._semaphore = semaphore

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *args):
        self._semaphore.release()
        return False
