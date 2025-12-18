"""
Set-cover solver for minimizing API requests.

Given a set of gaps (ticker, date_start, date_end) that need to be fetched,
find the minimum number of requests that cover all gaps while staying under
the row limit per request.

Key constraint: We can select arbitrary sets of tickers, but only
contiguous date ranges (API uses date.gte/date.lte).

This is a set-cover variant solved with greedy heuristics.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Gap:
    """A gap that needs to be fetched: one ticker over a date range."""
    ticker: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD

    def days(self) -> int:
        """Number of calendar days in this gap."""
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        return (e - s).days + 1

    def cells(self) -> set[tuple[str, str]]:
        """Return set of (ticker, date) cells this gap covers."""
        result = set()
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        current = s
        while current <= e:
            result.add((self.ticker, current.strftime('%Y-%m-%d')))
            current += timedelta(days=1)
        return result


@dataclass(frozen=True)
class Request:
    """A request covering one or more tickers over a date range."""
    tickers: frozenset[str]
    start: str
    end: str

    def rows(self, rows_per_ticker_day: int = 1) -> int:
        """Estimated rows this request will return."""
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        days = (e - s).days + 1
        return len(self.tickers) * days * rows_per_ticker_day

    def covers(self, gap: Gap) -> bool:
        """Does this request fully cover the given gap?"""
        return (
            gap.ticker in self.tickers
            and gap.start >= self.start
            and gap.end <= self.end
        )

    def cells(self) -> set[tuple[str, str]]:
        """Return set of (ticker, date) cells this request covers."""
        result = set()
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        current = s
        while current <= e:
            date_str = current.strftime('%Y-%m-%d')
            for ticker in self.tickers:
                result.add((ticker, date_str))
            current += timedelta(days=1)
        return result


def find_gaps(
    tickers: list[str],
    start: str,
    end: str,
    cached: set[tuple[str, str]],
) -> list[Gap]:
    """
    Find gaps between a query and cached cells.

    Args:
        tickers: Tickers being queried
        start: Query start date (YYYY-MM-DD)
        end: Query end date (YYYY-MM-DD)
        cached: Set of (ticker, date) cells already cached

    Returns:
        List of gaps (contiguous date ranges per ticker) not in cache
    """
    gaps = []
    s = datetime.strptime(start, '%Y-%m-%d')
    e = datetime.strptime(end, '%Y-%m-%d')

    for ticker in tickers:
        current = s
        gap_start = None

        while current <= e:
            date_str = current.strftime('%Y-%m-%d')
            is_missing = (ticker, date_str) not in cached

            if is_missing and gap_start is None:
                gap_start = date_str
            elif not is_missing and gap_start is not None:
                prev_date = (current - timedelta(days=1)).strftime('%Y-%m-%d')
                gaps.append(Gap(ticker, gap_start, prev_date))
                gap_start = None

            current += timedelta(days=1)

        if gap_start is not None:
            gaps.append(Gap(ticker, gap_start, end))

    return gaps


def solve_cover(
    gaps: list[Gap],
    max_rows: int,
    rows_per_ticker_day: int = 1,
) -> list[Request]:
    """
    Find a set of requests that cover all gaps, minimizing request count.

    Uses a greedy heuristic: repeatedly pick the request that covers
    the most uncovered cells while staying under max_rows.

    Args:
        gaps: List of gaps to cover
        max_rows: Maximum rows per request
        rows_per_ticker_day: Rows returned per ticker per day (default 1)

    Returns:
        List of requests that cover all gaps
    """
    if not gaps:
        return []

    # Convert gaps to cells that need coverage
    remaining: set[tuple[str, str]] = set()
    for gap in gaps:
        remaining.update(gap.cells())

    requests = []

    while remaining:
        best_request = _find_best_request(remaining, max_rows, rows_per_ticker_day)
        requests.append(best_request)
        remaining -= best_request.cells()

    return requests


def _find_best_request(
    cells: set[tuple[str, str]],
    max_rows: int,
    rows_per_ticker_day: int,
) -> Request:
    """
    Find the request that covers the most cells while staying under max_rows.

    Strategy: Group cells by date, try different date ranges and ticker combinations.
    """
    # Group cells by date and by ticker
    by_date: dict[str, set[str]] = {}  # date -> set of tickers
    by_ticker: dict[str, set[str]] = {}  # ticker -> set of dates
    all_tickers: set[str] = set()
    all_dates: set[str] = set()

    for ticker, date in cells:
        by_date.setdefault(date, set()).add(ticker)
        by_ticker.setdefault(ticker, set()).add(date)
        all_tickers.add(ticker)
        all_dates.add(date)

    sorted_dates = sorted(all_dates)
    max_days = max_rows // rows_per_ticker_day

    best_request = None
    best_coverage = 0

    # Strategy: Try different date ranges
    for i, start_date in enumerate(sorted_dates):
        # Try expanding the date range
        for j in range(i, len(sorted_dates)):
            end_date = sorted_dates[j]

            s = datetime.strptime(start_date, '%Y-%m-%d')
            e = datetime.strptime(end_date, '%Y-%m-%d')
            days = (e - s).days + 1

            max_tickers_for_range = max_rows // (days * rows_per_ticker_day)
            if max_tickers_for_range < 1:
                break  # Can't fit even one ticker, no point extending further

            # Find tickers that have cells in this date range
            tickers_with_cells = set()
            for d_idx in range(i, j + 1):
                d = sorted_dates[d_idx]
                tickers_with_cells.update(by_date.get(d, set()))

            # Take up to max_tickers (prioritize tickers with more cells in range)
            ticker_cell_counts = []
            for t in tickers_with_cells:
                count = sum(1 for d_idx in range(i, j + 1)
                           if sorted_dates[d_idx] in by_ticker.get(t, set()))
                ticker_cell_counts.append((count, t))
            ticker_cell_counts.sort(reverse=True)

            selected_tickers = frozenset(
                t for _, t in ticker_cell_counts[:max_tickers_for_range]
            )

            if not selected_tickers:
                continue

            request = Request(selected_tickers, start_date, end_date)
            coverage = len(request.cells() & cells)

            if coverage > best_coverage:
                best_coverage = coverage
                best_request = request

    # Fallback: single cell
    if best_request is None:
        ticker, date = next(iter(cells))
        best_request = Request(frozenset([ticker]), date, date)

    return best_request
