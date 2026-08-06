"""
Tests for the request rate limiter.

Uses an injected clock and sleep so the tests exercise the pacing logic
without spending real time.
"""
import asyncio

import pytest

from ndl_cache.rate_limit import RateLimiter, Window, parse_retry_after


class FakeClock:
    """Monotonic clock whose sleep advances time instantly."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def time(self):
        return self.now

    async def sleep(self, seconds):
        assert seconds >= 0
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


def limiter(windows, clock, **kwargs):
    return RateLimiter(windows=windows, clock=clock.time, sleep=clock.sleep,
                       **kwargs)


class TestWindowPacing:

    async def test_requests_under_the_limit_do_not_wait(self):
        clock = FakeClock()
        rl = limiter([Window(calls=5, seconds=10)], clock)
        for _ in range(5):
            await rl.acquire()
        assert clock.slept == []

    async def test_exceeding_a_window_waits_for_it_to_roll_off(self):
        clock = FakeClock()
        rl = limiter([Window(calls=3, seconds=10)], clock)
        for _ in range(3):
            await rl.acquire()
        await rl.acquire()
        # The fourth call must wait for the first to age out of the window.
        assert clock.slept and clock.slept[-1] == pytest.approx(10.0)

    async def test_spacing_calls_out_avoids_any_wait(self):
        clock = FakeClock()
        rl = limiter([Window(calls=3, seconds=10)], clock)
        for _ in range(6):
            await rl.acquire()
            clock.advance(4)
        assert clock.slept == []

    async def test_the_tightest_window_governs(self):
        clock = FakeClock()
        rl = limiter([Window(calls=100, seconds=10),
                      Window(calls=5, seconds=600)], clock)
        for _ in range(5):
            await rl.acquire()
        await rl.acquire()
        assert clock.slept[-1] == pytest.approx(600.0)

    async def test_records_are_pruned_so_memory_does_not_grow(self):
        clock = FakeClock()
        rl = limiter([Window(calls=3, seconds=10)], clock)
        for _ in range(50):
            await rl.acquire()
            clock.advance(5)
        assert len(rl._calls) <= 3


class TestCooldown:

    async def test_penalty_blocks_every_caller(self):
        # A rejection seen by one worker must pause all of them, not just the
        # one that was rejected. Otherwise the others keep hammering a server
        # that has already said stop.
        clock = FakeClock()
        rl = limiter([Window(calls=100, seconds=10)], clock)
        rl.penalize(60.0)
        await rl.acquire()
        assert clock.slept[-1] == pytest.approx(60.0)

    async def test_penalty_expires(self):
        clock = FakeClock()
        rl = limiter([Window(calls=100, seconds=10)], clock)
        rl.penalize(60.0)
        clock.advance(61)
        await rl.acquire()
        assert clock.slept == []

    async def test_longer_penalty_replaces_shorter(self):
        clock = FakeClock()
        rl = limiter([Window(calls=100, seconds=10)], clock)
        rl.penalize(30.0)
        rl.penalize(120.0)
        await rl.acquire()
        assert clock.slept[-1] == pytest.approx(120.0)

    async def test_shorter_penalty_does_not_shorten_longer(self):
        clock = FakeClock()
        rl = limiter([Window(calls=100, seconds=10)], clock)
        rl.penalize(120.0)
        rl.penalize(5.0)
        await rl.acquire()
        assert clock.slept[-1] == pytest.approx(120.0)


class TestConfiguration:

    def test_defaults_stay_under_the_published_limits(self):
        # Tables API: 300/10s, 2,000/10min, 50,000/day for an authenticated
        # key. Defaults must leave headroom, since other processes on the same
        # account share the quota.
        rl = RateLimiter()
        by_seconds = {w.seconds: w.calls for w in rl.windows}
        assert by_seconds[10] <= 300
        assert by_seconds[600] <= 2000
        assert by_seconds[86400] <= 50000

    def test_defaults_are_not_slower_than_serial_requests(self):
        # A 10-minute window set too low throttles below what one-at-a-time
        # requests achieve, which is worse than no limiter at all. Measured
        # serial throughput is ~1.8 req/s.
        rl = RateLimiter()
        by_seconds = {w.seconds: w.calls for w in rl.windows}
        assert by_seconds[600] / 600 >= 1.8

    def test_windows_from_env(self, monkeypatch):
        monkeypatch.setenv('NDL_RATE_LIMIT', '7/10,100/600')
        rl = RateLimiter()
        assert [(w.calls, w.seconds) for w in rl.windows] == [(7, 10), (100, 600)]

    def test_malformed_env_is_rejected_loudly(self, monkeypatch):
        monkeypatch.setenv('NDL_RATE_LIMIT', 'not-a-limit')
        with pytest.raises(ValueError, match='NDL_RATE_LIMIT'):
            RateLimiter()

    async def test_no_windows_means_no_pacing(self):
        clock = FakeClock()
        rl = limiter([], clock)
        for _ in range(1000):
            await rl.acquire()
        assert clock.slept == []


class TestRetryAfter:

    def test_parses_numeric_seconds(self):
        assert parse_retry_after({'Retry-After': '120'}) == 120.0

    def test_header_name_is_case_insensitive(self):
        assert parse_retry_after({'retry-after': '30'}) == 30.0

    def test_missing_header_returns_none(self):
        assert parse_retry_after({}) is None

    def test_garbage_header_returns_none(self):
        assert parse_retry_after({'Retry-After': 'soon'}) is None


class TestConcurrency:
    """
    The documented authenticated-tier limit is one call in flight plus one
    queued. This is not a rate, so pacing cannot satisfy it.
    """

    def test_default_is_one_executing_plus_one_queued(self):
        # Two, not one: the documented limit permits a queued call, and
        # holding one there hides a full round trip. Measured at +69%
        # throughput over serial with latency unchanged.
        assert RateLimiter().max_concurrency == 2

    def test_concurrency_from_env(self, monkeypatch):
        monkeypatch.setenv('NDL_MAX_CONCURRENCY', '3')
        assert RateLimiter().max_concurrency == 3

    def test_zero_concurrency_is_rejected(self, monkeypatch):
        monkeypatch.setenv('NDL_MAX_CONCURRENCY', '0')
        with pytest.raises(ValueError, match='NDL_MAX_CONCURRENCY'):
            RateLimiter()

    async def test_slots_bound_simultaneous_requests(self):
        rl = RateLimiter(windows=[], max_concurrency=2)
        peak = 0
        current = 0

        async def request():
            nonlocal peak, current
            async with rl.in_flight():
                current += 1
                peak = max(peak, current)
                await asyncio.sleep(0)
                current -= 1

        await asyncio.gather(*[request() for _ in range(20)])
        assert peak == 2
