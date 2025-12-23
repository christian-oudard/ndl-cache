"""
Set-cover solver for minimizing API requests.

Given sync bounds per ticker, find gaps (before/after synced ranges) and
batch them into minimal requests while staying under the row limit.

Key insight: Gaps only occur at boundaries (before synced_from or after synced_to),
so we can exploit this structure for efficient batching.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Gap:
    """A gap that needs to be fetched: one ticker over a date range."""
    ticker: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    is_past_gap: bool  # True if gap is before synced_from, False if after synced_to

    def days(self) -> int:
        """Number of calendar days in this gap."""
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        return (e - s).days + 1


@dataclass(frozen=True)
class Request:
    """A request covering one or more tickers over a date range."""
    tickers: frozenset[str]
    start: str
    end: str

    def days(self) -> int:
        """Number of calendar days in this request."""
        s = datetime.strptime(self.start, '%Y-%m-%d')
        e = datetime.strptime(self.end, '%Y-%m-%d')
        return (e - s).days + 1

    def rows(self, rows_per_ticker_day: int = 1) -> int:
        """Estimated rows this request will return."""
        return len(self.tickers) * self.days() * rows_per_ticker_day

    def covers(self, gap: Gap) -> bool:
        """Does this request fully cover the given gap?"""
        return (
            gap.ticker in self.tickers
            and gap.start >= self.start
            and gap.end <= self.end
        )


def _day_before(date_str: str) -> str:
    """Return the day before the given date."""
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d - timedelta(days=1)).strftime('%Y-%m-%d')


def _day_after(date_str: str) -> str:
    """Return the day after the given date."""
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d + timedelta(days=1)).strftime('%Y-%m-%d')


def find_gaps(
    tickers: list[str],
    query_start: str,
    query_end: str,
    sync_bounds: dict[str, tuple[str, str] | None],
) -> list[Gap]:
    """
    Find gaps between a query and synced bounds.

    Args:
        tickers: Tickers being queried
        query_start: Query start date (YYYY-MM-DD)
        query_end: Query end date (YYYY-MM-DD)
        sync_bounds: Dict mapping ticker -> (synced_from, synced_to) or None if not synced

    Returns:
        List of gaps (at most 2 per ticker: before and/or after synced range)
    """
    gaps = []

    for ticker in tickers:
        bounds = sync_bounds.get(ticker)

        if bounds is None:
            # Never synced - entire query range is a gap
            # Mark as past_gap (arbitrary, but consistent)
            gaps.append(Gap(ticker, query_start, query_end, is_past_gap=True))
            continue

        synced_from, synced_to = bounds

        # Gap before synced range
        if query_start < synced_from:
            gap_end = _day_before(synced_from)
            if gap_end >= query_start:  # Valid gap
                gaps.append(Gap(ticker, query_start, gap_end, is_past_gap=True))

        # Gap after synced range
        if query_end > synced_to:
            gap_start = _day_after(synced_to)
            if gap_start <= query_end:  # Valid gap
                gaps.append(Gap(ticker, gap_start, query_end, is_past_gap=False))

    return gaps


def solve_cover(
    gaps: list[Gap],
    max_rows: int,
    rows_per_ticker_day: int = 1,
) -> list[Request]:
    """
    Find a set of requests that cover all gaps, minimizing request count.

    Uses boundary-aware batching: past gaps share query_start, future gaps share query_end.

    Args:
        gaps: List of gaps to cover
        max_rows: Maximum rows per request
        rows_per_ticker_day: Rows returned per ticker per day (default 1)

    Returns:
        List of requests that cover all gaps
    """
    if not gaps:
        return []

    past_gaps = [g for g in gaps if g.is_past_gap]
    future_gaps = [g for g in gaps if not g.is_past_gap]

    requests = []
    requests.extend(_batch_gaps(past_gaps, max_rows, rows_per_ticker_day, is_past=True))
    requests.extend(_batch_gaps(future_gaps, max_rows, rows_per_ticker_day, is_past=False))

    return requests


def _batch_gaps(
    gaps: list[Gap],
    max_rows: int,
    rows_per_ticker_day: int,
    is_past: bool,
) -> list[Request]:
    """
    Batch gaps into requests, exploiting boundary structure.

    For past gaps: all share the same start date, sort by end date descending (longest first)
    For future gaps: all share the same end date, sort by start date ascending (longest first)

    Greedy algorithm: pack tickers into requests while staying under max_rows.
    """
    if not gaps:
        return []

    requests = []

    if is_past:
        # Past gaps: share start, vary by end. Sort by end descending (longest first).
        sorted_gaps = sorted(gaps, key=lambda g: g.end, reverse=True)
    else:
        # Future gaps: share end, vary by start. Sort by start ascending (longest first).
        sorted_gaps = sorted(gaps, key=lambda g: g.start)

    # Group gaps by their boundary date for efficient batching
    # Past gaps: group by end date, future gaps: group by start date
    remaining = list(sorted_gaps)

    while remaining:
        # Start a new request with the first (longest) gap
        first = remaining.pop(0)
        batch_tickers = [first.ticker]

        if is_past:
            # Request range: (shared start, this gap's end)
            req_start = first.start
            req_end = first.end
        else:
            # Request range: (this gap's start, shared end)
            req_start = first.start
            req_end = first.end

        req_days = (datetime.strptime(req_end, '%Y-%m-%d') -
                    datetime.strptime(req_start, '%Y-%m-%d')).days + 1
        current_rows = req_days * rows_per_ticker_day

        # Try to add more tickers that fit within the same date range
        still_remaining = []
        for gap in remaining:
            additional_rows = req_days * rows_per_ticker_day  # Each ticker adds req_days worth

            if current_rows + additional_rows <= max_rows:
                # Check if this gap fits within the request range
                if is_past:
                    # Past gap: must have end <= req_end (will be covered by overfetch)
                    if gap.end <= req_end:
                        batch_tickers.append(gap.ticker)
                        current_rows += additional_rows
                    else:
                        still_remaining.append(gap)
                else:
                    # Future gap: must have start >= req_start (will be covered by overfetch)
                    if gap.start >= req_start:
                        batch_tickers.append(gap.ticker)
                        current_rows += additional_rows
                    else:
                        still_remaining.append(gap)
            else:
                still_remaining.append(gap)

        remaining = still_remaining
        requests.append(Request(frozenset(batch_tickers), req_start, req_end))

    return requests
