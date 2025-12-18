"""
Tests for the set-cover solver.

Uses small page limits (10 rows) to make the problem tractable for testing.
Assumes 1 ticker × 1 day = 1 row for simplicity.

Key insight: Gaps only occur at boundaries (before synced_from or after synced_to).
"""
import pytest
from cover import Gap, Request, solve_cover, find_gaps


class TestGap:
    def test_days_single(self):
        g = Gap('AAPL', '2024-01-01', '2024-01-01', is_past_gap=True)
        assert g.days() == 1

    def test_days_range(self):
        g = Gap('AAPL', '2024-01-01', '2024-01-10', is_past_gap=True)
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
        g = Gap('AAPL', '2024-01-01', '2024-01-05', is_past_gap=True)
        assert r.covers(g)

    def test_covers_subset(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-10')
        g = Gap('AAPL', '2024-01-03', '2024-01-05', is_past_gap=True)
        assert r.covers(g)

    def test_not_covers_wrong_ticker(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-10')
        g = Gap('MSFT', '2024-01-01', '2024-01-05', is_past_gap=True)
        assert not r.covers(g)

    def test_not_covers_outside_range(self):
        r = Request(frozenset(['AAPL']), '2024-01-01', '2024-01-05')
        g = Gap('AAPL', '2024-01-03', '2024-01-10', is_past_gap=True)
        assert not r.covers(g)


class TestFindGaps:
    """Test the bounds-based gap finding."""

    def test_no_sync_bounds(self):
        """Ticker with no sync record = full query range gap."""
        gaps = find_gaps(['AAPL'], '2024-01-01', '2024-01-10', {})
        assert len(gaps) == 1
        assert gaps[0].ticker == 'AAPL'
        assert gaps[0].start == '2024-01-01'
        assert gaps[0].end == '2024-01-10'

    def test_fully_synced(self):
        """Query within sync bounds = no gaps."""
        sync_bounds = {'AAPL': ('2024-01-01', '2024-01-31')}
        gaps = find_gaps(['AAPL'], '2024-01-05', '2024-01-20', sync_bounds)
        assert len(gaps) == 0

    def test_gap_before_synced(self):
        """Query starts before synced_from = past gap."""
        sync_bounds = {'AAPL': ('2024-01-15', '2024-01-31')}
        gaps = find_gaps(['AAPL'], '2024-01-01', '2024-01-20', sync_bounds)
        assert len(gaps) == 1
        assert gaps[0].start == '2024-01-01'
        assert gaps[0].end == '2024-01-14'
        assert gaps[0].is_past_gap

    def test_gap_after_synced(self):
        """Query ends after synced_to = future gap."""
        sync_bounds = {'AAPL': ('2024-01-01', '2024-01-15')}
        gaps = find_gaps(['AAPL'], '2024-01-10', '2024-01-31', sync_bounds)
        assert len(gaps) == 1
        assert gaps[0].start == '2024-01-16'
        assert gaps[0].end == '2024-01-31'
        assert not gaps[0].is_past_gap

    def test_gaps_both_sides(self):
        """Query extends both before and after = two gaps."""
        sync_bounds = {'AAPL': ('2024-01-10', '2024-01-20')}
        gaps = find_gaps(['AAPL'], '2024-01-01', '2024-01-31', sync_bounds)
        assert len(gaps) == 2
        past_gap = next(g for g in gaps if g.is_past_gap)
        future_gap = next(g for g in gaps if not g.is_past_gap)
        assert past_gap.start == '2024-01-01'
        assert past_gap.end == '2024-01-09'
        assert future_gap.start == '2024-01-21'
        assert future_gap.end == '2024-01-31'

    def test_multiple_tickers_mixed(self):
        """Multiple tickers with different sync states."""
        sync_bounds = {
            'AAPL': ('2024-01-10', '2024-01-20'),  # Partial sync
            'MSFT': None,  # Not synced
        }
        gaps = find_gaps(['AAPL', 'MSFT'], '2024-01-01', '2024-01-31', sync_bounds)

        aapl_gaps = [g for g in gaps if g.ticker == 'AAPL']
        msft_gaps = [g for g in gaps if g.ticker == 'MSFT']

        assert len(aapl_gaps) == 2  # before and after
        assert len(msft_gaps) == 1  # full range


class TestSolveCover:
    """Test the boundary-aware set-cover solver with max_rows=10."""

    def test_empty_gaps(self):
        """No gaps = no requests."""
        assert solve_cover([], max_rows=10) == []

    def test_single_gap_fits(self):
        """Single gap under limit = one request."""
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-05', is_past_gap=True)]  # 5 rows
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert requests[0].covers(gaps[0])

    def test_single_gap_exact_limit(self):
        """Single gap at exactly the limit."""
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-10', is_past_gap=True)]  # 10 rows
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert requests[0].covers(gaps[0])

    def test_single_gap_exceeds_limit(self):
        """Single gap over limit - solver can't split, returns oversized request."""
        # Note: The solver doesn't split gaps; that's handled by _split_filters in cache.py
        gaps = [Gap('AAPL', '2024-01-01', '2024-01-15', is_past_gap=True)]  # 15 rows
        requests = solve_cover(gaps, max_rows=10)

        # Solver returns single request (splitting happens elsewhere)
        assert len(requests) == 1
        assert requests[0].covers(gaps[0])

    def test_two_tickers_same_dates_fit(self):
        """Two tickers with same dates that fit together."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-04', is_past_gap=True),  # 4 rows
            Gap('MSFT', '2024-01-01', '2024-01-04', is_past_gap=True),  # 4 rows
        ]  # Total: 8 rows, fits in 10
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_two_tickers_same_dates_exceed(self):
        """Two tickers with same dates that don't fit together."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-06', is_past_gap=True),  # 6 rows
            Gap('MSFT', '2024-01-01', '2024-01-06', is_past_gap=True),  # 6 rows
        ]  # Total: 12 rows, needs 2 requests
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) >= 2  # Can't fit in 1
        assert all(any(r.covers(g) for r in requests) for g in gaps)

    def test_batch_tickers_with_same_gap(self):
        """Multiple tickers with identical gaps should batch efficiently."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-02', is_past_gap=True),
            Gap('MSFT', '2024-01-01', '2024-01-02', is_past_gap=True),
            Gap('GOOGL', '2024-01-01', '2024-01-02', is_past_gap=True),
            Gap('AMZN', '2024-01-01', '2024-01-02', is_past_gap=True),
        ]  # Total: 8 rows, fits in one request
        requests = solve_cover(gaps, max_rows=10)

        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_past_and_future_gaps_separate(self):
        """Past and future gaps are batched separately."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-05', is_past_gap=True),   # past
            Gap('MSFT', '2024-01-01', '2024-01-05', is_past_gap=True),   # past
            Gap('AAPL', '2024-02-01', '2024-02-05', is_past_gap=False),  # future
            Gap('MSFT', '2024-02-01', '2024-02-05', is_past_gap=False),  # future
        ]
        requests = solve_cover(gaps, max_rows=10)

        # Should batch into 2 requests: one for past, one for future
        assert len(requests) == 2
        assert all(any(r.covers(g) for r in requests) for g in gaps)

    def test_different_length_past_gaps_batch(self):
        """Past gaps with different lengths can still batch (overfetch shorter ones)."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-05', is_past_gap=True),  # 5 days
            Gap('MSFT', '2024-01-01', '2024-01-03', is_past_gap=True),  # 3 days (shorter)
        ]
        requests = solve_cover(gaps, max_rows=10)

        # Should batch into 1 request covering 01-01 to 01-05 for both
        assert len(requests) == 1
        # Request covers the longer gap
        assert requests[0].covers(gaps[0])
        # And overfetches for the shorter gap
        assert requests[0].covers(gaps[1])

    def test_many_small_gaps_batch_efficiently(self):
        """Many 1-day gaps for different tickers should batch."""
        gaps = [Gap(f'T{i}', '2024-01-01', '2024-01-01', is_past_gap=True) for i in range(8)]
        requests = solve_cover(gaps, max_rows=10)

        # 8 tickers × 1 day = 8 rows, fits in one request
        assert len(requests) == 1
        assert all(requests[0].covers(g) for g in gaps)

    def test_respects_max_rows(self):
        """All returned requests must be under max_rows."""
        gaps = [
            Gap('AAPL', '2024-01-01', '2024-01-03', is_past_gap=True),
            Gap('MSFT', '2024-01-01', '2024-01-03', is_past_gap=True),
            Gap('GOOGL', '2024-01-01', '2024-01-03', is_past_gap=True),
            Gap('AMZN', '2024-01-01', '2024-01-03', is_past_gap=True),
        ]  # 4 tickers × 3 days = 12 rows
        requests = solve_cover(gaps, max_rows=10)

        for r in requests:
            assert r.rows() <= 10, f"Request {r} exceeds max_rows"
