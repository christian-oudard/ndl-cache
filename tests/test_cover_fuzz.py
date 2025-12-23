"""
Fuzz test for the set-cover solver.

Generates random sync bounds and validates the solver produces correct results.
"""
import random
import pytest
from datetime import datetime, timedelta
from ndl_cache.cover import Gap, Request, solve_cover, find_gaps


def random_date_range(start: str = '2024-01-01', end: str = '2024-12-31') -> tuple[str, str]:
    """Generate a random date range (start <= end)."""
    s = datetime.strptime(start, '%Y-%m-%d')
    e = datetime.strptime(end, '%Y-%m-%d')
    delta = (e - s).days
    d1 = s + timedelta(days=random.randint(0, delta))
    d2 = s + timedelta(days=random.randint(0, delta))
    if d1 > d2:
        d1, d2 = d2, d1
    return d1.strftime('%Y-%m-%d'), d2.strftime('%Y-%m-%d')


def generate_sync_bounds(
    tickers: list[str],
    query_start: str,
    query_end: str,
    sync_prob: float = 0.5,
) -> dict[str, tuple[str, str] | None]:
    """Generate random sync bounds for tickers."""
    bounds = {}
    for ticker in tickers:
        if random.random() < sync_prob:
            # Generate bounds that may or may not overlap with query
            synced_from, synced_to = random_date_range(query_start, query_end)
            bounds[ticker] = (synced_from, synced_to)
        else:
            bounds[ticker] = None
    return bounds


def validate_solution(gaps: list[Gap], requests: list[Request], max_rows: int) -> None:
    """Validate that the solution is correct."""
    # All gaps must be covered
    for gap in gaps:
        covered = any(r.covers(gap) for r in requests)
        assert covered, f"Gap {gap} not covered by any request"


@pytest.mark.parametrize("seed", range(20))
def test_cover_fuzz(seed):
    """Fuzz test: random scenarios at various scales."""
    random.seed(seed)

    # Vary parameters based on seed for diversity
    num_tickers = random.randint(1, 15)
    tickers = [f'T{i}' for i in range(num_tickers)]
    query_start, query_end = random_date_range('2024-01-01', '2024-03-31')
    sync_prob = random.uniform(0.0, 1.0)
    max_rows = random.choice([5, 10, 20, 50])

    sync_bounds = generate_sync_bounds(tickers, query_start, query_end, sync_prob)
    gaps = find_gaps(tickers, query_start, query_end, sync_bounds)

    if not gaps:
        return  # Nothing to test

    requests = solve_cover(gaps, max_rows)
    validate_solution(gaps, requests, max_rows)
