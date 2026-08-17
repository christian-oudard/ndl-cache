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

The quota belongs to the API key, not to the process holding it, so the call
history is kept in a small SQLite file shared by every process using the same
credential. Two scripts each pacing themselves to the published limits would
otherwise together exceed them, which is why parallel work previously had to be
serialised by hand. A rejection is shared the same way: one process being told
to stand down stands all of them down.

Override with NDL_RATE_LIMIT, a comma-separated list of ``calls/seconds``,
NDL_MAX_CONCURRENCY, and NDL_RATE_LIMIT_STATE:

    NDL_RATE_LIMIT=5/1,2000/600      # 5 per second and 2000 per 10 minutes
    NDL_RATE_LIMIT=                  # empty disables pacing entirely
    NDL_MAX_CONCURRENCY=2            # requests in flight
    NDL_RATE_LIMIT_STATE=/tmp/x.db   # where the shared history lives
    NDL_RATE_LIMIT_STATE=            # empty paces per process, as before
"""
import asyncio
import hashlib
import os
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


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


def default_state_path():
    """
    Where the shared call history lives.

    Deliberately not derived from NDL_CACHE_DB_PATH. Splitting the cache into
    one file per period is a normal thing to do, and the quota is still one
    quota; pacing has to follow the credential, not the cache file.
    """
    override = os.environ.get('NDL_RATE_LIMIT_STATE')
    if override is not None:
        return override or None
    cache_dir = Path.home() / '.cache' / 'ndl_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / 'rate_limit.sqlite')


class MemoryCallLog:
    """Call history for one process, used when no state file is configured."""

    def __init__(self):
        self._calls = deque()
        self._blocked_until = 0.0

    @contextmanager
    def transaction(self):
        yield

    def recent(self, cutoff):
        return [t for t in self._calls if t > cutoff]

    def record(self, now):
        self._calls.append(now)

    def prune(self, cutoff):
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def blocked_until(self):
        return self._blocked_until

    def block_until(self, when):
        self._blocked_until = max(self._blocked_until, when)


class SharedCallLog:
    """
    Call history shared by every process using one credential.

    SQLite rather than the DuckDB cache because DuckDB takes an exclusive lock
    on its file, so a second process cannot even open it, which is the case
    this exists to serve.

    Timestamps are wall clock, not monotonic: monotonic clocks are only
    comparable within a process.
    """

    def __init__(self, path, credential=None):
        self._key = hashlib.sha256(
            (credential or '').encode()).hexdigest()[:16]
        self._conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self._conn.execute('PRAGMA journal_mode = WAL')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS calls (key TEXT, ts REAL)')
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS calls_key_ts ON calls (key, ts)')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS cooldown '
            '(key TEXT PRIMARY KEY, until REAL)')

    @contextmanager
    def transaction(self):
        # IMMEDIATE takes the write lock up front, so the read that decides
        # whether a call fits cannot be overtaken by another process recording
        # one. A deferred transaction would let two processes both conclude
        # there is room for the last call in a window.
        self._conn.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self._conn.execute('ROLLBACK')
            raise
        self._conn.execute('COMMIT')

    def recent(self, cutoff):
        rows = self._conn.execute(
            'SELECT ts FROM calls WHERE key = ? AND ts > ? ORDER BY ts',
            (self._key, cutoff)).fetchall()
        return [row[0] for row in rows]

    def record(self, now):
        self._conn.execute('INSERT INTO calls (key, ts) VALUES (?, ?)',
                           (self._key, now))

    def prune(self, cutoff):
        self._conn.execute('DELETE FROM calls WHERE key = ? AND ts <= ?',
                           (self._key, cutoff))

    def blocked_until(self):
        row = self._conn.execute(
            'SELECT until FROM cooldown WHERE key = ?', (self._key,)).fetchone()
        return row[0] if row else 0.0

    def block_until(self, when):
        self._conn.execute(
            'INSERT INTO cooldown (key, until) VALUES (?, ?) '
            'ON CONFLICT (key) DO UPDATE SET until = max(until, excluded.until)',
            (self._key, when))


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
                 max_concurrency=None, log=None, credential=None):
        if windows is None:
            spec = os.environ.get('NDL_RATE_LIMIT')
            windows = DEFAULT_WINDOWS if spec is None else parse_windows(spec)
        if max_concurrency is None:
            max_concurrency = int(os.environ.get(
                'NDL_MAX_CONCURRENCY', DEFAULT_MAX_CONCURRENCY))
        if max_concurrency < 1:
            raise ValueError(
                f'NDL_MAX_CONCURRENCY must be at least 1, got {max_concurrency}')
        if log is None:
            path = default_state_path()
            log = (SharedCallLog(path, credential) if path
                   else MemoryCallLog())
        self.windows = tuple(windows)
        self.max_concurrency = max_concurrency
        self.penalty_seconds = penalty_seconds
        # Wall clock, not monotonic: the history is shared between processes
        # and monotonic clocks are only comparable within one.
        self._clock = clock or time.time
        self._sleep = sleep or asyncio.sleep
        self._log = log
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
        with self._log.transaction():
            self._log.block_until(self._clock() + seconds)

    async def acquire(self):
        """Block until it is safe to issue one request, then record it."""
        while True:
            async with self._lock:
                now = self._clock()
                with self._log.transaction():
                    blocked_until = self._log.blocked_until()
                    if now < blocked_until:
                        wait = blocked_until - now
                    else:
                        calls = self._log.recent(now - self._horizon)
                        wait = self._wait_for_windows(calls, now)
                        if wait <= 0:
                            self._log.record(now)
                            self._log.prune(now - self._horizon)
                            return

            # Sleep outside the lock so other workers can queue behind us
            # rather than serializing their waits end to end.
            await self._sleep(wait)

    def _wait_for_windows(self, calls, now):
        """Seconds until a request would fit in every window; 0 if it fits now."""
        wait = 0.0
        for window in self.windows:
            cutoff = now - window.seconds
            in_window = [t for t in calls if t > cutoff]
            if len(in_window) >= window.calls:
                # The oldest call in this window has to age out first.
                oldest = in_window[-window.calls]
                wait = max(wait, oldest + window.seconds - now)
        return wait


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
