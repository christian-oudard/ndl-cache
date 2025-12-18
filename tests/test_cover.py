"""
Tests for the set-cover solver.

Uses small page limits (10 rows) to make the problem tractable for testing.
Assumes 1 ticker × 1 day = 1 row for simplicity.

Key constraint: We can select arbitrary sets of tickers, but only
contiguous date ranges (API uses date.gte/date.lte).
"""
import pytest
from cover import Gap, Request, solve_cover


class TestGap:
    def test_days_single(self):
        g = Gap('AAPL', '2024-01-01', '2024-01-01')
        assert g.days() == 1

    def test_days_range(self):
        g = Gap('AAPL', '2024-01-01', '2024-01-10')
        assert g.days() == 10


class TestRequest:
    def test_rows_single_ticker_single_day(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-01')
        assert r.rows() == 1

    def test_rows_multiple_tickers(self):
        r = Request(frozenset(['AAPL', 'MSFT']), '2024-01-01', '2024-01-05')
        assert r.rows() == 10  # 2 tickers × 5 days

    def test_covers_exact(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-05')
        g = Gap('AAPL', '2024-01-01', '2024-01-05')
        assert r.covers(g)

    def test_covers_subset(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-10')
        g = Gap('AAPL', '2024-01-03', '2024-01-05')
        assert r.covers(g)

    def test_not_covers_wrong_ticker(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-10')
        g = Gap('MSFT', '2024-01-01', '2024-01-05')
        assert not r.covers(g)

    def test_not_covers_outside_range(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-05')
        g = Gap('AAPL', '2024-01-03', '2024-01-10')
        assert not r.covers(g)


class TestSolveCover:
    """Test the greedy set-cover solver with max_rows=10."""

    def test_empty_gaps(self):
        """No gaps = no requests."""
        assert solve_cover([], max_rows=10) == []

    def test_single_gap_fits(self):
        """Single gap under limit = one request."""
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-05')]  # 5 rows
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert requests[0].covers(gaps[0])

    def test_single_gap_exact_limit(self):
        """Single gap at exactly the limit."""
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-10')]  # 10 rows
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert requests[0].covers(gaps[0])

    def test_single_gap_exceeds_limit(self):
        """Single gap over limit must be split by dates."""
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-15')]  # 15 rows
        requests = solve_cover(gaps, max_rows=10)

        # Need at least 2 requests
        assert len(requests) >= 2

        # All parts of the gap should be covered
        for day in range(1, 16):
            date = f'2024-01-{day:02d}'
            day_gap = Gap('AAPL', date, date)
            assert any(r.covers(day_gap) for r in requests), f"Day {date} not covered"

    def test_two_tickers_same_dates_fit(self):
        """Two tickers with same dates that fit together."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-04'),  # 4 rows
            Gap('MSFT', '2024-01-01', '2024-01-04'),  # 4 rows
        ]  # Total: 8 rows, fits in 10
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_two_tickers_same_dates_exceed(self):
        """Two tickers with same dates that don't fit together."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-06'),  # 6 rows
            Gap('MSFT', '2024-01-01', '2024-01-06'),  # 6 rows
        ]  # Total: 12 rows, needs 2 requests
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) >= 2  # Can't fit in 1
        # All cells must be covered
        needed = set().union(*(g.cells() for g in gaps))
        covered = set().union(*(r.cells() for r in requests))
        assert needed <= covered

    def test_batch_tickers_with_same_gap(self):
        """Multiple tickers with identical gaps should batch efficiently."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-02'),  # 2 rows
            Gap('MSFT', '2024-01-01', '2024-01-02'),  # 2 rows
            Gap('GOOGL', '2024-01-01', '2024-01-02'),  # 2 rows
            Gap('AMZN', '2024-01-01', '2024-01-02'),  # 2 rows
        ]  # Total: 8 rows, fits in one request
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_overfetch_is_acceptable(self):
        """Solver may overfetch to reduce request count."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-01'),  # Day 1
            Gap('AAPL', '2024-01-03', '2024-01-03'),  # Day 3 (gap on day 2)
        ]
        requests = solve_cover(gaps, max_rows=10)

        # Could be 1 request (days 1-3, overfetching day 2) or 2 requests
        # Greedy should prefer 1 request with overfetch
        assert len(requests) <= 2
        assert all(any(r.covers(g) for r in requests) for g in gaps)

    def test_different_date_ranges(self):
        """Tickers with completely different date ranges need separate requests."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-05'),  # 5 rows
            Gap('MSFT', '2024-06-01', '2024-06-05'),  # 5 rows, 150+ days apart
        ]
        requests = solve_cover(gaps, max_rows=10)

        # These can't be batched (would be 2 tickers × 150+ days > 10)
        assert len(requests) == 2
        assert all(any(r.covers(g) for r in requests) for g in gaps)

    def test_complex_scenario(self):
        """
        Complex scenario with overlapping and non-overlapping gaps.

        AAPL: days 1-3, 5-7 (two gaps)
        MSFT: days 2-4 (overlaps with AAPL's first gap)
        GOOGL: days 10-12 (separate)

        Optimal: ~2-3 requests depending on strategy
        """
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-03'),
            Gap('AAPL', '2024-01-05', '2024-01-07'),
            Gap('MSFT', '2024-01-02', '2024-01-04'),
            Gap('GOOGL', '2024-01-10', '2024-01-12'),
        ]
        requests = solve_cover(gaps, max_rows=10)

        # All cells must be covered
        needed = set().union(*(g.cells() for g in gaps))
        covered = set().union(*(r.cells() for r in requests))
        assert needed <= covered

        # Should be reasonably efficient (not 4 separate requests)
        assert len(requests) <= 3

    def test_many_small_gaps_batch_efficiently(self):
        """Many 1-day gaps for different tickers should batch."""
        gaps = [Gap(f'T{i}', '2024-01-01', '2024-01-01') for i in range(8)]
        requests = solve_cover(gaps, max_rows=10)

        # 8 tickers × 1 day = 8 rows, fits in one request
        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_respects_max_rows(self):
        """All returned requests must be under max_rows."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-10'),
            Gap('MSFT', '2024-01-01', '2024-01-10'),
            Gap('GOOGL', '2024-01-01', '2024-01-10'),
        ]
        requests = solve_cover(gaps, max_rows=10)

        for r in requests:
            assert r.rows() <= 10, f"Request {r} exceeds max_rows"
