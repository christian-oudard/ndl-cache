"""
Fuzz test for the set-cover solver.

Generates random cache states and validates the solver produces correct results.
"""
import random
import pytest
from datetime import datetime, timedelta
from cover import Gap, Request, solve_cover, find_gaps


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


def generate_cached_cells(
    tickers: list[str],
    date_start: str,
    date_end: str,
    coverage_prob: float = 0.3,
) -> set[tuple[str, str]]:
    """Generate random cached cells with given coverage probability."""
    cached = set()
    s = datetime.strptime(date_start, '%Y-%m-%d')
    e = datetime.strptime(date_end, '%Y-%m-%d')

    current = s
    while current <= e:
        date_str = current.strftime('%Y-%m-%d')
        for ticker in tickers:
            if random.random() < coverage_prob:
                cached.add((ticker, date_str))
        current += timedelta(days=1)

    return cached


def validate_solution(gaps: list[Gap], requests: list[Request], max_rows: int) -> None:
    """Validate that the solution is correct."""
    # All requests must be under max_rows
    for r in requests:
        assert r.rows() <= max_rows, f"Request {r} exceeds max_rows ({r.rows()} > {max_rows})"

    # All gap cells must be covered
    needed = set()
    for g in gaps:
        needed.update(g.cells())

    covered = set()
    for r in requests:
        covered.update(r.cells())

    missing = needed - covered
    assert not missing, f"Missing cells: {list(missing)[:10]}..."


@pytest.mark.parametrize("seed", range(20))
def test_cover_fuzz(seed):
    """Fuzz test: random scenarios at various scales."""
    random.seed(seed)

    # Vary parameters based on seed for diversity
    num_tickers = random.randint(1, 15)
    tickers = [f'T{i}' for i in range(num_tickers)]
    date_start, date_end = random_date_range('2024-01-01', '2024-03-31')
    coverage_prob = random.uniform(0.1, 0.9)
    max_rows = random.choice([5, 10, 20, 50])

    cached = generate_cached_cells(tickers, date_start, date_end, coverage_prob)
    gaps = find_gaps(tickers, date_start, date_end, cached)

    if not gaps:
        return  # Nothing to test

    requests = solve_cover(gaps, max_rows)
    validate_solution(gaps, requests, max_rows)
