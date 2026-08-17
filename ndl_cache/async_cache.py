"""
Async cache layer using aioduckdb for non-blocking DuckDB operations.

Provides async_query() for async access and query() for sync access.
"""
import asyncio
import os
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import aioduckdb
import duckdb
import pandas as pd

from .async_client import AsyncNDLClient, NDLError
from .cover import solve_cover, find_gaps
from .tables import TableDef, TICKERS


# Concurrency is enforced by the rate limiter's semaphore, which is the only
# thing that can satisfy a limit counted in open calls rather than in a rate.

# Rows per request, kept well under the provider's ten thousand row page.
NDL_SPLIT_THRESHOLD = 9000

# Requests are also bounded by URL length, independently of row count. A query
# for one date across a thousand tickers returns almost no rows but builds a
# ticker parameter of several kilobytes; Nasdaq answered a 10,032 character URL
# with 414 and an HTML error page, which surfaced as a bare "API request
# failed". Most servers cap the request line near 8 KB, so budget the ticker
# list well under that to leave room for the other parameters.
MAX_TICKER_PARAM_CHARS = 3000

# How many days at the end of a ticker's cached range to ask about when
# checking whether the provider has restated it. Wide enough to contain a
# trading day across a weekend and a holiday.
PROBE_DAYS = 7

# Ticker cap for tables whose row density is unknown, so that one request
# cannot run past the page limit. SF1 is about 600 rows per ticker across all
# dimensions, so this keeps a request near six pages.
MAX_TICKERS_UNKNOWN_DENSITY = 100

# Lock per table for entire query operations (read-fetch-write cycle).
#
# Problem: asyncio.run() creates a new event loop and closes it after each call.
# asyncio.Lock objects are bound to the event loop that was running when they
# were created. When asyncio.run() is called again with a new event loop, the
# locks are still bound to the old (closed) loop, causing:
#     RuntimeError: <asyncio.locks.Lock ...> is bound to a different event loop
#
# Solution: Store a weak reference to the event loop along with the lock. When
# getting a lock, we compare the actual loop objects (not just their ids, since
# Python can reuse memory addresses after garbage collection). If the stored
# loop is gone or different, we create a new lock for the current loop.
_table_query_locks: dict[str, tuple[weakref.ref, asyncio.Lock]] = {}


def _get_table_lock(table_name: str, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    """Get or create a lock for a specific table's query operations."""
    existing = _table_query_locks.get(table_name)
    if existing is not None:
        loop_ref, lock = existing
        # Check if this lock is for the current loop (same object, not just same id)
        if loop_ref() is loop:
            return lock
        # Old loop was garbage collected or this is a different loop - create new lock

    lock = asyncio.Lock()
    _table_query_locks[table_name] = (weakref.ref(loop), lock)
    return lock


@dataclass(frozen=True)
class QueryPlan:
    """
    What a query would fetch, without fetching it.

    A query about to pull sixteen years across fifteen thousand tickers looks
    exactly like one that returns from cache in a millisecond, and the only
    way to tell them apart used to be to run it and wait.
    """
    table: str
    cached_tickers: int
    fetch_tickers: int
    fetch_ranges: list[tuple[str, str]] = field(default_factory=list)
    requests: int = 0
    estimated_rows: int | None = None

    @property
    def satisfied(self) -> bool:
        """Whether the cache already holds everything asked for."""
        return self.requests == 0


def is_read_only() -> bool:
    """
    Whether to open the cache without contending for the write lock.

    A read-only connection serves whatever is cached and never syncs, so
    several analysis processes can share one cache file. It does not let you
    read while another process writes; DuckDB refuses that either way.
    """
    return os.environ.get('NDL_CACHE_READ_ONLY', '').lower() in ('1', 'true', 'yes')


def is_cache_disabled() -> bool:
    """Check if cache is disabled via environment variable."""
    return os.environ.get('NDL_CACHE_DISABLED', '').lower() in ('1', 'true', 'yes')


def get_db_path() -> str:
    """Get database path from NDL_CACHE_DB_PATH env var or default."""
    if 'NDL_CACHE_DB_PATH' in os.environ:
        return os.environ['NDL_CACHE_DB_PATH']
    cache_dir = Path.home() / '.cache' / 'ndl_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / 'cache.duckdb')


def _columns(cols) -> str:
    """A quoted column list for a SELECT, an INSERT or a key."""
    return ', '.join(_quote(col) for col in cols)


def _quote(identifier: str) -> str:
    """
    Quote a column name for SQL.

    SHARADAR/TICKERS has a column called `table`, which is a reserved word, so
    identifiers cannot be interpolated bare.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _effective_sync_date(date_str: str, delay_days: int) -> str:
    """Cap a date to account for data provider delays."""
    if delay_days <= 0:
        return date_str
    max_sync_date = (datetime.now() - timedelta(days=delay_days)).strftime('%Y-%m-%d')
    return min(date_str, max_sync_date)


class _CacheManager:
    """
    Internal cache manager for a specific table.

    Use async_query() or query() functions instead of this class directly.
    """

    def __init__(self, table: TableDef):
        self.table = table
        self._db_path = get_db_path()
        self._conn: aioduckdb.Connection | None = None
        self._ndl_client: AsyncNDLClient | None = None

    async def _get_conn_without_init(self) -> aioduckdb.Connection:
        """Get or create connection without table initialization.

        Retries with backoff to handle Windows file lock delays when a previous
        process recently closed the database.
        """
        if self._conn is not None:
            return self._conn

        max_retries = 5
        base_delay = 0.1  # 100ms initial delay
        last_error = None

        for attempt in range(max_retries):
            try:
                self._conn = await aioduckdb.connect(
                    self._db_path, read_only=is_read_only())
                return self._conn
            except duckdb.IOException as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    continue
                # Final attempt failed
                wal_file = Path(self._db_path + '.wal')
                if wal_file.exists():
                    raise duckdb.IOException(
                        f"Database is locked. Only one process can access the cache at a time.\n"
                        f"If no other process is running, delete stale lock files:\n"
                        f"  rm {self._db_path}.wal*"
                    ) from e
                raise

        # Should not reach here, but just in case
        raise last_error  # type: ignore

    async def _get_conn(self) -> aioduckdb.Connection:
        """Get or create the async DuckDB connection with table initialization."""
        if self._conn is None:
            await self._get_conn_without_init()
            # Creating the bookkeeping table is a write, which a read-only
            # connection cannot do and a read-only caller does not need.
            if not is_read_only():
                await self._ensure_sync_bounds_table()
        return self._conn

    async def _get_ndl_client(self) -> AsyncNDLClient:
        """Get or create the async NDL client."""
        if self._ndl_client is None:
            self._ndl_client = AsyncNDLClient()
        return self._ndl_client

    async def close(self):
        """Close connections."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        if self._ndl_client is not None:
            await self._ndl_client.close()
            self._ndl_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_sync_bounds_table(self):
        """Create sync_bounds table if it doesn't exist."""
        conn = await self._get_conn_without_init()
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table.sync_bounds_table_name()} (
                ticker VARCHAR PRIMARY KEY,
                synced_from DATE,
                synced_to DATE,
                max_lastupdated DATE,
                last_staleness_check DATE
            )
        """)

    async def _ensure_data_table(self, data_columns: list[str]):
        """
        Create the data table, rebuilding it if an older one lacks columns
        this version needs.

        Only tables held in full are rebuilt. Their contents are refetched on
        the same call, so dropping costs one request; a per-ticker table would
        lose work that cannot be recovered so cheaply, and is left alone.

        The tickers table gained a `table` column when its key became
        (table, ticker), and a cache written before that has no way to accept
        the new rows.
        """
        conn = await self._get_conn()
        cols = list(self.table.index_columns) + data_columns
        name = self.table.safe_table_name()

        if (self.table.full_refresh_days is not None
                and await self._table_exists(name)):
            cursor = await conn.execute(f'DESCRIBE {name}')
            existing = {row[0] for row in await cursor.fetchall()}
            if not set(cols) <= existing:
                await conn.execute(f'DROP TABLE {name}')

        col_defs = [f'{_quote(col)} {self.table.column_types.get(col, "DOUBLE")}'
                    for col in cols]
        pk = _columns(self.table.index_columns)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk})
            )
        """)

    async def _get_sync_bounds(self, tickers: list[str]) -> dict[str, dict | None]:
        """Get sync bounds for given tickers."""
        if not tickers:
            return {}

        conn = await self._get_conn()
        placeholders = ', '.join(['?'] * len(tickers))
        cursor = await conn.execute(f"""
            SELECT ticker, synced_from, synced_to, max_lastupdated, last_staleness_check
            FROM {self.table.sync_bounds_table_name()}
            WHERE ticker IN ({placeholders})
        """, tickers)
        result = await cursor.fetchall()

        bounds = {ticker: None for ticker in tickers}
        for ticker, synced_from, synced_to, max_lastupdated, last_staleness_check in result:
            bounds[ticker] = {
                'synced_from': str(synced_from)[:10] if synced_from else None,
                'synced_to': str(synced_to)[:10] if synced_to else None,
                'max_lastupdated': str(max_lastupdated)[:10] if max_lastupdated else None,
                'last_staleness_check': str(last_staleness_check)[:10] if last_staleness_check else None,
            }

        return bounds

    async def _update_sync_bounds(self, ticker: str, from_date: str, to_date: str, max_lastupdated: str | None = None):
        """Update sync bounds for a ticker, expanding the existing range."""
        conn = await self._get_conn()
        effective_to = _effective_sync_date(to_date, self.table.sync_delay_days)

        cursor = await conn.execute(f"""
            SELECT synced_from, synced_to, max_lastupdated
            FROM {self.table.sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker])
        existing = await cursor.fetchone()

        if existing:
            old_from, old_to = str(existing[0])[:10], str(existing[1])[:10]
            old_max_lastupdated = str(existing[2])[:10] if existing[2] else None
            new_from = min(from_date, old_from)
            new_to = max(effective_to, old_to)
            if max_lastupdated and (not old_max_lastupdated or max_lastupdated > old_max_lastupdated):
                new_max_lastupdated = max_lastupdated
            else:
                new_max_lastupdated = old_max_lastupdated
        else:
            new_from = from_date
            new_to = effective_to
            new_max_lastupdated = max_lastupdated

        today = datetime.now().strftime('%Y-%m-%d')
        if existing:
            await conn.execute(f"""
                UPDATE {self.table.sync_bounds_table_name()}
                SET synced_from = ?, synced_to = ?, max_lastupdated = ?, last_staleness_check = ?
                WHERE ticker = ?
            """, [new_from, new_to, new_max_lastupdated, today, ticker])
        else:
            await conn.execute(f"""
                INSERT INTO {self.table.sync_bounds_table_name()}
                (ticker, synced_from, synced_to, max_lastupdated, last_staleness_check)
                VALUES (?, ?, ?, ?, ?)
            """, [ticker, new_from, new_to, new_max_lastupdated, today])

    async def _mark_ticker_synced(self, ticker: str, max_lastupdated: str | None = None):
        """Mark a ticker as synced for tables without date columns."""
        conn = await self._get_conn()
        today = datetime.now().strftime('%Y-%m-%d')

        cursor = await conn.execute(f"""
            SELECT max_lastupdated FROM {self.table.sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker])
        existing = await cursor.fetchone()

        if existing and existing[0]:
            old_max = str(existing[0])[:10]
            if max_lastupdated and max_lastupdated > old_max:
                new_max = max_lastupdated
            else:
                new_max = old_max
        else:
            new_max = max_lastupdated

        if existing:
            await conn.execute(f"""
                UPDATE {self.table.sync_bounds_table_name()}
                SET max_lastupdated = ?, last_staleness_check = ?
                WHERE ticker = ?
            """, [new_max, today, ticker])
        else:
            await conn.execute(f"""
                INSERT INTO {self.table.sync_bounds_table_name()}
                (ticker, synced_from, synced_to, max_lastupdated, last_staleness_check)
                VALUES (?, NULL, NULL, ?, ?)
            """, [ticker, new_max, today])

    async def _invalidate_ticker(self, ticker: str):
        """Delete all cached data and sync bounds for a ticker."""
        conn = await self._get_conn()

        cursor = await conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{self.table.safe_table_name()}'
        """)
        result = await cursor.fetchone()
        table_exists = result[0] > 0

        if table_exists:
            await conn.execute(f"""
                DELETE FROM {self.table.safe_table_name()}
                WHERE ticker = ?
            """, [ticker])

        await conn.execute(f"""
            DELETE FROM {self.table.sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker])

    async def _probe_windows(self, tickers: list[str],
                             sync_bounds: dict) -> dict[tuple[str, str], list[str]]:
        """
        The date window to ask the provider about, per group of tickers.

        Sharadar restates a ticker by rewriting `lastupdated` on every row it
        holds for that ticker: MSFT carries one single value, 2026-05-21,
        across 1999 to 2024. Reading the watermark therefore does not need the
        whole history, which is what made this check cost a hundred pages. A
        few days at the end of what is cached carries the same value.

        Tickers are grouped by the window so that the usual case, where every
        ticker was synced to the same date, is one request.
        """
        windows: dict[tuple[str, str], list[str]] = {}
        for ticker in tickers:
            bounds = sync_bounds[ticker]
            synced_to = bounds.get('synced_to')
            synced_from = bounds.get('synced_from')
            if not synced_to:
                continue
            start = (datetime.strptime(synced_to, '%Y-%m-%d')
                     - timedelta(days=PROBE_DAYS - 1)).strftime('%Y-%m-%d')
            if synced_from:
                start = max(start, synced_from)
            windows.setdefault((start, synced_to), []).append(ticker)
        return windows

    async def _cached_watermarks(self, tickers: list[str], start: str,
                                 end: str) -> dict[str, str]:
        """Highest lastupdated this cache holds per ticker over a window."""
        conn = await self._get_conn()
        date_col = self.table.date_column
        placeholders = ', '.join(['?'] * len(tickers))
        cursor = await conn.execute(f"""
            SELECT ticker, MAX(lastupdated) FROM {self.table.safe_table_name()}
            WHERE ticker IN ({placeholders})
              AND {_quote(date_col)} BETWEEN ? AND ?
            GROUP BY ticker
        """, [*tickers, start, end])
        return {t: str(lu)[:10] for t, lu in await cursor.fetchall() if lu}

    async def _check_and_invalidate_stale(self, tickers: list[str]):
        """Check if cached data is stale and invalidate if needed.

        Compares the provider's `lastupdated` watermark against this cache's
        own, over the same narrow window at the end of what is held, so that
        the two are like for like and neither side reads the whole history.
        """
        if not tickers:
            return

        conn = await self._get_conn()
        today = datetime.now().strftime('%Y-%m-%d')
        sync_bounds = await self._get_sync_bounds(tickers)

        tickers_to_check = []
        for ticker in tickers:
            bounds = sync_bounds.get(ticker)
            if bounds is None:
                continue
            last_check = bounds.get('last_staleness_check')
            if last_check == today:
                continue
            tickers_to_check.append(ticker)

        if not tickers_to_check:
            return

        # ACTIONS has no lastupdated column at all, and asking for one is a
        # 403 rather than an empty result, so there is nothing to probe with.
        # Its rows are therefore cached and never refreshed; see IMPROVEMENTS.
        if (self.table.date_column is None
                or 'lastupdated' not in self.table.query_columns
                or not await self._data_table_exists()):
            await self._mark_checked(tickers_to_check, today)
            return

        client = await self._get_ndl_client()
        stale_tickers = []

        for (start, end), group in (
                await self._probe_windows(tickers_to_check, sync_bounds)).items():
            date_col = self.table.date_column
            try:
                df = await client.get_table(
                    self.table.name,
                    columns=['ticker', 'lastupdated'],
                    ticker=group,
                    paginate=True,
                    **{date_col: {'gte': start, 'lte': end}},
                )
            except Exception:
                # On error, skip staleness check but still update last_staleness_check
                continue

            if len(df) == 0 or 'lastupdated' not in df.columns:
                continue
            api = df.groupby('ticker')['lastupdated'].max()
            cached = await self._cached_watermarks(group, start, end)
            for ticker, api_lu in api.items():
                cached_lu = cached.get(ticker)
                if pd.notna(api_lu) and cached_lu and str(api_lu)[:10] > cached_lu:
                    stale_tickers.append(ticker)

        # Recorded for every ticker looked at, including ones the provider had
        # nothing to say about, so a check is not repeated within the day.
        await self._mark_checked(tickers_to_check, today)

        for ticker in set(stale_tickers) | set(
                await self._renamed_away(tickers_to_check)):
            await self._invalidate_ticker(ticker)

    async def _mark_checked(self, tickers: list[str], today: str):
        """Record that these tickers were checked for staleness today."""
        conn = await self._get_conn()
        placeholders = ', '.join(['?'] * len(tickers))
        await conn.execute(f"""
            UPDATE {self.table.sync_bounds_table_name()}
            SET last_staleness_check = ?
            WHERE ticker IN ({placeholders})
        """, [today, *tickers])

    async def _table_exists(self, name: str) -> bool:
        conn = await self._get_conn()
        cursor = await conn.execute(
            'SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?',
            [name])
        return (await cursor.fetchone())[0] > 0

    async def _data_table_exists(self) -> bool:
        return await self._table_exists(self.table.safe_table_name())

    async def _ensure_universe(self):
        """
        Make sure the tickers table is on hand, since it is what says whether
        a symbol still exists.

        Without this the rename check is dead code for anyone who only ever
        queries prices: nothing else populates that table, so the comparison
        silently finds nothing, forever. It refreshes at most daily and the
        warm case is a local read, so the cost is one fetch a day.

        Best effort on purpose. This is a check running alongside somebody
        else's query, so failing to fetch it should leave the check undone,
        not fail the price query that happened to trigger it.

        Runs on this manager's own connection. A second connection to the same
        file works, but the write lands outside this one's snapshot, so the
        table it just created is invisible here and the check quietly finds
        nothing.
        """
        universe = _CacheManager(TICKERS)
        universe._conn = self._conn
        universe._ndl_client = await self._get_ndl_client()
        try:
            await universe._sync_full_table()
        except Exception:
            pass
        finally:
            # Borrowed, so neither is this manager's to close.
            universe._conn = None
            universe._ndl_client = None

    async def _renamed_away(self, tickers: list[str]) -> list[str]:
        """
        Tickers holding cached rows whose symbol no longer exists.

        When a company is renamed the provider moves its entire history to the
        new symbol and the old one stops existing: SEP has no rows at all for
        FB, and META carries them back to 2012. A cache filled before the
        rename keeps serving the old copy forever, and nothing else notices,
        because a probe for FB comes back empty and empty reads as "nothing to
        report" rather than "this symbol is gone".

        Detected against the cached tickers table, so it costs no request. The
        old name is recoverable from the ACTIONS row `tickerchangefrom`, whose
        `contraticker` holds it.

        The lookup has to be restricted to this table's own universe, because
        symbols are reassigned across them. FB is a ProShares ETF now, listed
        under SFP in 2025, so asking whether the symbol exists anywhere would
        answer yes and leave a decade of Facebook prices sitting in the equity
        cache.
        """
        if not tickers or self.table.tickers_table is None:
            return []
        await self._ensure_universe()
        if not await self._table_exists(TICKERS.safe_table_name()):
            return []
        conn = await self._get_conn()
        placeholders = ', '.join(['?'] * len(tickers))
        cursor = await conn.execute(f"""
            SELECT DISTINCT ticker FROM {TICKERS.safe_table_name()}
            WHERE "table" = ? AND ticker IN ({placeholders})
        """, [self.table.tickers_table, *tickers])
        listed = {row[0] for row in await cursor.fetchall()}
        return [t for t in tickers if t not in listed]

    def _estimate_rows_for_range(self, date_gte: str | None, date_lte: str | None) -> int:
        """Estimate number of rows per ticker for a date range."""
        if not (date_gte and date_lte):
            return 1
        # Caller should check rows_per_year is not None before calling
        assert self.table.rows_per_year is not None
        start = datetime.strptime(date_gte, '%Y-%m-%d')
        end = datetime.strptime(date_lte, '%Y-%m-%d')
        calendar_days = (end - start).days + 1
        return max(1, int(calendar_days * self.table.rows_per_year / 365))

    def _estimate_rows(self, filters: dict) -> int:
        """Estimate number of rows a filter set will return."""
        ticker = filters.get('ticker')
        n_tickers = len(ticker) if isinstance(ticker, list) else 1
        date_col = self.table.date_column
        est_rows_per_ticker = self._estimate_rows_for_range(
            filters.get(f'{date_col}_gte'),
            filters.get(f'{date_col}_lte')
        )
        return n_tickers * est_rows_per_ticker

    @staticmethod
    def _ticker_param_too_long(ticker) -> bool:
        """Whether a ticker filter would overflow the URL on its own."""
        if not isinstance(ticker, list) or len(ticker) <= 1:
            return False
        return len(','.join(ticker)) > MAX_TICKER_PARAM_CHARS

    @staticmethod
    def _tickers_per_url(ticker: list[str]) -> int:
        """How many tickers fit in one URL, using the longest as the estimate."""
        longest = max(len(t) for t in ticker)
        return max(1, MAX_TICKER_PARAM_CHARS // (longest + 1))

    def _split_filters(self, filters: dict, max_rows: int = NDL_SPLIT_THRESHOLD) -> list[dict]:
        """Split a filter set into chunks that each return < max_rows."""
        # Tables whose row density is unknown cannot be sized, but they still
        # have to be bounded: SF1 returns about 600 rows per ticker across all
        # dimensions, so two thousand tickers in one request is 119 pages,
        # past the page limit. That used to truncate silently and now raises,
        # so cap the ticker count instead of guessing a density.
        if self.table.rows_per_year is None:
            ticker = filters.get('ticker')
            if isinstance(ticker, list) and len(ticker) > MAX_TICKERS_UNKNOWN_DENSITY:
                return [
                    {**filters, 'ticker': chunk if len(chunk) > 1 else chunk[0]}
                    for chunk in (
                        ticker[i:i + MAX_TICKERS_UNKNOWN_DENSITY]
                        for i in range(0, len(ticker), MAX_TICKERS_UNKNOWN_DENSITY))
                ]
            return [filters]

        date_col = self.table.date_column
        ticker = filters.get('ticker')
        date_gte = filters.get(f'{date_col}_gte')
        date_lte = filters.get(f'{date_col}_lte')

        est_rows = self._estimate_rows(filters)
        # A request can be small in rows and still too long as a URL, so the
        # ticker parameter has to be checked before taking the early exit.
        if est_rows < max_rows and not self._ticker_param_too_long(ticker):
            return [filters]

        # Strategy 1: Split by tickers
        if isinstance(ticker, list) and len(ticker) > 1:
            est_rows_per_ticker = self._estimate_rows_for_range(date_gte, date_lte)
            tickers_per_chunk = max(1, max_rows // est_rows_per_ticker)
            # Whichever limit binds first, rows or URL length.
            tickers_per_chunk = min(
                tickers_per_chunk, self._tickers_per_url(ticker))

            chunks = []
            for i in range(0, len(ticker), tickers_per_chunk):
                chunk_tickers = ticker[i:i + tickers_per_chunk]
                chunk_filters = {**filters, 'ticker': chunk_tickers if len(chunk_tickers) > 1 else chunk_tickers[0]}
                chunks.extend(self._split_filters(chunk_filters, max_rows))
            return chunks

        # Strategy 2: Split by date range
        if date_gte and date_lte:
            start = datetime.strptime(date_gte, '%Y-%m-%d')
            end = datetime.strptime(date_lte, '%Y-%m-%d')
            calendar_days_per_chunk = max(1, int(max_rows * 365 / self.table.rows_per_year))

            chunks = []
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(chunk_start + timedelta(days=calendar_days_per_chunk - 1), end)
                chunk_filters = {
                    **filters,
                    f'{date_col}_gte': chunk_start.strftime('%Y-%m-%d'),
                    f'{date_col}_lte': chunk_end.strftime('%Y-%m-%d'),
                }
                chunks.append(chunk_filters)
                chunk_start = chunk_end + timedelta(days=1)
            return chunks

        return [filters]

    def _compute_optimal_fetches(
        self,
        tickers: list[str],
        date_gte: str,
        date_lte: str,
        sync_bounds_raw: dict[str, dict | None],
        max_rows: int = NDL_SPLIT_THRESHOLD,
    ) -> list[dict]:
        """Compute optimal fetch filter sets using set-cover solver."""
        sync_bounds = {}
        for ticker, bounds in sync_bounds_raw.items():
            if bounds is None:
                sync_bounds[ticker] = None
            else:
                sync_bounds[ticker] = (bounds['synced_from'], bounds['synced_to'])
        gaps = find_gaps(tickers, date_gte, date_lte, sync_bounds)

        if not gaps:
            return []

        requests = solve_cover(gaps, max_rows)

        date_col = self.table.date_column
        return [
            {
                'ticker': list(req.tickers) if len(req.tickers) > 1 else list(req.tickers)[0],
                f'{date_col}_gte': req.start,
                f'{date_col}_lte': req.end,
            }
            for req in requests
        ]

    async def plan(self, **filters) -> QueryPlan:
        """
        Work out what a query would fetch, without fetching anything.

        Reuses the same gap finding and splitting the query itself uses, so
        the request count is what would actually go out rather than a guess
        about it. Touches the network not at all, and so does not run the
        staleness check: this answers "what would this cost right now".
        """
        ticker_filter = filters.get('ticker')
        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        elif isinstance(ticker_filter, list):
            tickers = list(ticker_filter)
        else:
            tickers = []

        date_col = self.table.date_column
        date_gte = filters.get(f'{date_col}_gte') if date_col else None
        date_lte = filters.get(f'{date_col}_lte') if date_col else None

        if not tickers or not (date_gte and date_lte):
            return QueryPlan(table=self.table.name, cached_tickers=len(tickers),
                             fetch_tickers=0)

        bounds = await self._get_sync_bounds(tickers)
        fetches = self._compute_optimal_fetches(
            tickers, date_gte, date_lte, bounds)

        wanted = set()
        ranges = set()
        for fetch in fetches:
            wanted.update(self._tickers_in([fetch]))
            ranges.add((fetch[f'{date_col}_gte'], fetch[f'{date_col}_lte']))

        chunks = [c for fetch in fetches for c in self._split_filters(fetch)]
        rows = (sum(self._estimate_rows(c) for c in chunks)
                if self.table.rows_per_year is not None else None)

        return QueryPlan(
            table=self.table.name,
            cached_tickers=len(set(tickers) - wanted),
            fetch_tickers=len(wanted),
            fetch_ranges=sorted(ranges),
            requests=len(chunks),
            estimated_rows=rows,
        )

    async def fetch_from_ndl(self, **filters) -> pd.DataFrame:
        """Fetch data from NDL API using async client."""
        client = await self._get_ndl_client()

        # Convert our filter format to NDL format
        ndl_filters = {}
        range_filters = {}

        for key, value in filters.items():
            if key.endswith('_gte'):
                col = key[:-4]
                range_filters.setdefault(col, {})['gte'] = value
            elif key.endswith('_lte'):
                col = key[:-4]
                range_filters.setdefault(col, {})['lte'] = value
            else:
                ndl_filters[key] = value

        ndl_filters.update(range_filters)

        result = await client.get_table(
            self.table.name,
            columns=self.table.all_columns,
            paginate=True,
            **ndl_filters
        )

        return result

    async def _fetch_parallel(self, filter_sets: list[dict]) -> pd.DataFrame:
        """Fetch multiple filter sets concurrently."""
        if not filter_sets:
            return pd.DataFrame()

        # Split any oversized filter sets
        all_chunks = []
        for filters in filter_sets:
            all_chunks.extend(self._split_filters(filters))

        if len(all_chunks) == 1:
            return await self.fetch_from_ndl(**all_chunks[0])

        # Fetch all chunks concurrently
        results = await asyncio.gather(*[self.fetch_from_ndl(**f) for f in all_chunks])

        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True)

    async def _sync_parallel(self, filter_sets: list[dict]) -> int:
        """Fetch multiple filter sets concurrently and sync to cache."""
        if not filter_sets:
            return 0

        # Which tickers this cache already holds data for, read before the
        # fetch. A ticker that has never returned a row is not given a range
        # below; see _covered_ranges for why.
        known = {t for t, b in (await self._get_sync_bounds(
            self._tickers_in(filter_sets))).items() if b is not None}

        queried = await self._fetch_parallel(filter_sets)
        ticker_stats = self._per_ticker_stats(queried)

        # Rows first, coverage second, and never the other way round. A sync
        # bound written for data that never landed is invisible: later reads
        # return nothing, with no error and no refetch, because the cache
        # believes the range is covered. Measured on a real cache, 107 of 264
        # tickers in SEP claimed coverage with zero rows. Writing the rows
        # first means a failure leaves the range unclaimed and it is fetched
        # again.
        stored = 0
        if len(queried) > 0:
            data_columns = [c for c in self.table.query_columns
                            if c in queried.columns]
            await self._ensure_data_table(data_columns)
            cols = list(self.table.index_columns) + data_columns
            # Dedupe in case API returns duplicate rows
            store_df = queried[cols].drop_duplicates(
                subset=list(self.table.index_columns))
            await self._store(store_df, cols)
            stored = len(store_df)

        for ticker, (start, end) in self._covered_ranges(
                filter_sets, known | set(ticker_stats)).items():
            await self._update_sync_bounds(
                ticker, start, end,
                ticker_stats.get(ticker, {}).get('max_lastupdated'))

        # Tables with no date column, and fetches with no date range, still
        # record only what came back; there is no requested range to use.
        for ticker, stats in ticker_stats.items():
            if self.table.date_column is None:
                await self._mark_ticker_synced(ticker, stats.get('max_lastupdated'))
            elif stats.get('min_date') and stats.get('max_date'):
                await self._update_sync_bounds(
                    ticker, stats['min_date'], stats['max_date'],
                    stats.get('max_lastupdated'))

        return stored

    @staticmethod
    def _tickers_in(filter_sets: list[dict]) -> list[str]:
        """Every ticker named across a set of planned fetches."""
        tickers = set()
        for filters in filter_sets:
            value = filters.get('ticker')
            if isinstance(value, list):
                tickers.update(value)
            elif value:
                tickers.add(value)
        return sorted(tickers)

    def _covered_ranges(self, filter_sets: list[dict],
                        eligible: set[str]) -> dict[str, tuple[str, str]]:
        """
        The date range each ticker is now covered for, having asked for it.

        Coverage is what was *requested*, not what came back. A range with no
        rows in it is still answered: the provider was asked and said there is
        nothing there. Recording only the rows received leaves the empty parts
        permanently unsatisfied, so a cache holding exactly what is asked for
        still refetched the market holidays at either end on every call, and a
        delisted ticker refetched a range that grew with every step of a walk.

        This is the same idea the contiguous span already applies to holes in
        the middle, which is why those never had the problem.

        Only tickers in `eligible` get a range: ones this cache already holds
        data for, or which returned some now. A ticker that has never returned
        a row has no `lastupdated` watermark, so nothing could ever invalidate
        a wrong guess about it, and it stays unrecorded on purpose.

        The upper end is clamped by _update_sync_bounds against the provider's
        delay, so the leading edge is never claimed and rolls forward.
        """
        date_col = self.table.date_column
        if date_col is None:
            return {}

        ranges: dict[str, tuple[str, str]] = {}
        for filters in filter_sets:
            start = filters.get(f'{date_col}_gte')
            end = filters.get(f'{date_col}_lte')
            if not (start and end):
                continue
            for ticker in self._tickers_in([filters]):
                if ticker not in eligible:
                    continue
                held = ranges.get(ticker)
                ranges[ticker] = ((min(start, held[0]), max(end, held[1]))
                                  if held else (start, end))
        return ranges

    def _per_ticker_stats(self, queried: pd.DataFrame) -> dict[str, dict]:
        """Date range and max lastupdated per ticker in a fetched frame."""
        if len(queried) == 0:
            return {}

        wanted = {}
        if 'lastupdated' in queried.columns:
            wanted['max_lastupdated'] = ('lastupdated', 'max')
        date_col = self.table.date_column
        if date_col and date_col in queried.columns:
            wanted['min_date'] = (date_col, 'min')
            wanted['max_date'] = (date_col, 'max')
        if not wanted:
            return {}

        grouped = queried.groupby('ticker').agg(**wanted)
        return {
            ticker: {k: str(v)[:10] for k, v in row.items() if pd.notna(v)}
            for ticker, row in grouped.to_dict('index').items()
        }

    @asynccontextmanager
    async def _staged(self, store_df: pd.DataFrame):
        """
        Offer a frame to SQL as `_incoming` for the length of a write.

        Inserted in one statement rather than row by row. DuckDB is columnar,
        and an executemany of INSERT OR REPLACE pays a separate statement and
        index probe per row: writing one month of prices for 500 tickers that
        way took twenty seconds and got slower as the table grew, which is
        most of what made a warm cache feel slow.

        Everything inside must use execute_on_self. execute() runs on a fresh
        cursor, which aioduckdb duplicates from the connection, and which
        therefore cannot see anything registered on it.
        """
        conn = await self._get_conn()
        await conn.register('_incoming', store_df)
        try:
            yield conn
        finally:
            await conn.unregister('_incoming')

    async def _store(self, store_df: pd.DataFrame, cols: list[str]):
        """
        Merge a frame into the data table.

        INSERT OR REPLACE because parallel fetches and retries overlap, and
        the same row can legitimately arrive twice.
        """
        async with self._staged(store_df) as conn:
            await conn.execute_on_self(
                f'INSERT OR REPLACE INTO {self.table.safe_table_name()} '
                f'({_columns(cols)}) SELECT {_columns(cols)} FROM _incoming')

    async def _full_synced_at(self) -> str | None:
        """When the whole table was last replaced, or None if never."""
        conn = await self._get_conn()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                table_name VARCHAR PRIMARY KEY,
                full_synced_at DATE
            )
        """)
        cursor = await conn.execute(
            'SELECT full_synced_at FROM cache_meta WHERE table_name = ?',
            [self.table.name])
        row = await cursor.fetchone()
        return str(row[0])[:10] if row and row[0] else None

    async def _sync_full_table(self):
        """
        Replace the whole table if the copy on disk is older than the table's
        refresh window.

        For a table every caller wants in full, this is cheaper and simpler
        than per-ticker bookkeeping: one fetch, one refresh policy, and no
        sync bounds at all.
        """
        synced_at = await self._full_synced_at()
        if synced_at is not None:
            age = (datetime.now() - datetime.strptime(synced_at, '%Y-%m-%d')).days
            if age < self.table.full_refresh_days:
                return

        client = await self._get_ndl_client()
        fetched = await client.get_table(
            self.table.name, columns=self.table.all_columns, paginate=True)
        if len(fetched) == 0:
            raise NDLError(f'{self.table.name} returned no rows')

        data_columns = [c for c in self.table.query_columns
                        if c in fetched.columns]
        await self._ensure_data_table(data_columns)

        cols = list(self.table.index_columns) + data_columns
        store_df = fetched[cols].drop_duplicates(
            subset=list(self.table.index_columns))

        # Replaced wholesale rather than merged, so that rows the provider has
        # dropped do not linger in the cache forever, and in one transaction so
        # that a failure between the two leaves the old copy rather than an
        # empty table. Deleting first and discovering the problem second turned
        # a schema mismatch into 326 rows becoming 0.
        conn = await self._get_conn()
        await self._replace_all(store_df, cols)
        await conn.execute(
            'INSERT OR REPLACE INTO cache_meta (table_name, full_synced_at) '
            'VALUES (?, ?)',
            [self.table.name, datetime.now().strftime('%Y-%m-%d')])

    async def _replace_all(self, store_df: pd.DataFrame, cols: list[str]):
        """
        Swap the whole table's contents for a new frame, atomically.

        In one transaction so that a failure between the delete and the insert
        leaves the old copy rather than an empty table.
        """
        name = self.table.safe_table_name()
        async with self._staged(store_df) as conn:
            await conn.execute_on_self('BEGIN TRANSACTION')
            try:
                await conn.execute_on_self(f'DELETE FROM {name}')
                await conn.execute_on_self(
                    f'INSERT INTO {name} ({_columns(cols)}) '
                    f'SELECT {_columns(cols)} FROM _incoming')
            except BaseException:
                await conn.execute_on_self('ROLLBACK')
                raise
            await conn.execute_on_self('COMMIT')

    async def get_cached(self, **filters) -> pd.DataFrame:
        """Get data from local cache."""
        conn = await self._get_conn()

        where_clauses = []
        params = []
        for key, value in filters.items():
            if key.endswith('_gte'):
                where_clauses.append(f"{_quote(key[:-4])} >= ?")
                params.append(value)
            elif key.endswith('_lte'):
                where_clauses.append(f"{_quote(key[:-4])} <= ?")
                params.append(value)
            elif isinstance(value, list):
                placeholders = ', '.join(['?'] * len(value))
                where_clauses.append(f"{_quote(key)} IN ({placeholders})")
                params.extend(value)
            else:
                where_clauses.append(f"{_quote(key)} = ?")
                params.append(value)

        where = ' AND '.join(where_clauses) if where_clauses else '1=1'

        cursor = await conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{self.table.safe_table_name()}'
        """)
        result = await cursor.fetchone()
        table_exists = result[0] > 0

        if not table_exists:
            return pd.DataFrame()

        cursor = await conn.execute(f"""
            SELECT * FROM {self.table.safe_table_name()}
            WHERE {where}
            ORDER BY {_columns(self.table.index_columns)}
        """, params)

        rows = await cursor.fetchall()
        if not rows:
            return pd.DataFrame()

        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        if len(df) > 0:
            for col in self.table.index_columns:
                if self.table.column_types.get(col) == 'DATE':
                    df[col] = pd.to_datetime(df[col])
            df = df.set_index(list(self.table.index_columns))

        return df

    async def query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """Query data from cache, fetching from NDL if not cached."""
        ticker_filter = filters.get('ticker')
        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        elif isinstance(ticker_filter, list):
            tickers = ticker_filter
        else:
            tickers = []

        if is_cache_disabled():
            if tickers:
                fetch_filters = dict(filters)
                fetch_filters['ticker'] = tickers
                result = await self._fetch_parallel([fetch_filters])
                if not result.empty and self.table.index_columns:
                    index_cols = [c for c in self.table.index_columns if c in result.columns]
                    if index_cols:
                        result = result.set_index(index_cols)
                return result
            return pd.DataFrame()

        if is_read_only():
            # Serve what is held and sync nothing, so several readers can
            # share one cache file.
            return self._select_columns(await self.get_cached(**filters), columns)

        # Lock the entire read-fetch-write cycle per table to prevent race conditions
        loop = asyncio.get_running_loop()
        lock = _get_table_lock(self.table.name, loop)
        async with lock:
            if self.table.full_refresh_days is not None:
                await self._sync_full_table()
                return self._select_columns(
                    await self.get_cached(**filters), columns)

            if tickers:
                await self._check_and_invalidate_stale(tickers)

            if self.table.date_column is None:
                if tickers:
                    sync_bounds = await self._get_sync_bounds(tickers)
                    unsynced = [t for t in tickers if sync_bounds.get(t) is None]
                    if unsynced:
                        await self._sync_parallel([{'ticker': t} for t in unsynced])
            else:
                date_gte = filters.get(f'{self.table.date_column}_gte')
                date_lte = filters.get(f'{self.table.date_column}_lte')

                if tickers and date_gte and date_lte:
                    sync_bounds_raw = await self._get_sync_bounds(tickers)
                    optimal_fetches = self._compute_optimal_fetches(tickers, date_gte, date_lte, sync_bounds_raw)
                    if optimal_fetches:
                        await self._sync_parallel(optimal_fetches)
                elif tickers and not date_gte and not date_lte:
                    sync_bounds = await self._get_sync_bounds(tickers)
                    unsynced = [t for t in tickers if sync_bounds.get(t) is None]
                    if unsynced:
                        await self._sync_parallel([{'ticker': t} for t in unsynced])

            result = await self.get_cached(**filters)

        return self._select_columns(result, columns)

    @staticmethod
    def _select_columns(result: pd.DataFrame,
                        columns: list[str] | str | None) -> pd.DataFrame:
        """Narrow a result to the requested columns."""
        if len(result) == 0:
            return pd.DataFrame()
        if columns is None:
            return result
        if isinstance(columns, str):
            columns = [columns]
        return result[[c for c in columns if c in result.columns]]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

async def async_query(
    table: TableDef,
    /,
    columns: list[str] | str | None = None,
    explain: bool = False,
    **filters,
) -> pd.DataFrame | QueryPlan:
    """
    Query data from a Sharadar table asynchronously.

    Args:
        table: Table definition (e.g., SEP, SFP, SF1)
        columns: Columns to return (None for all)
        **filters: Query filters (ticker, date_gte, date_lte, etc.)

    Returns:
        DataFrame indexed by the table's index columns

    Example:
        from ndl_cache import SEP, async_query

        df = await async_query(SEP, ticker='AAPL', date_gte='2024-01-01', date_lte='2024-12-31')
    """
    async with _CacheManager(table) as mgr:
        if explain:
            return await mgr.plan(**filters)
        return await mgr.query(columns=columns, **filters)


def query(
    table: TableDef,
    /,
    columns: list[str] | str | None = None,
    explain: bool = False,
    **filters,
) -> pd.DataFrame | QueryPlan:
    """
    Query data from a Sharadar table synchronously.

    Args:
        table: Table definition (e.g., SEP, SFP, SF1)
        columns: Columns to return (None for all)
        **filters: Query filters (ticker, date_gte, date_lte, etc.)

    Returns:
        DataFrame indexed by the table's index columns

    Example:
        from ndl_cache import SEP, query

        df = query(SEP, ticker='AAPL', date_gte='2024-01-01', date_lte='2024-12-31')
    """
    return asyncio.run(
        async_query(table, columns=columns, explain=explain, **filters))


async def async_validate_sync_bounds(
    table: TableDef,
    fix: bool = False,
) -> list[str]:
    """
    Find tickers claiming a synced range while holding no rows.

    That combination is invisible in normal use: reads return nothing, with no
    error and no refetch, because the cache believes the range is covered, and
    a caller cannot tell it from a ticker that genuinely has no data. Measured
    on a real cache, 107 of 264 tickers in SEP and 111 of 2,253 in DAILY.

    Coverage is recorded from the range that was requested, so it legitimately
    reaches past the first and last row a ticker has; a range wider than the
    data is normal and is not reported. Holding *nothing* is not, because a
    range is only ever recorded for a ticker that returned rows.

    Opened read-only unless asked to fix, so inspecting never takes the write
    lock and any number of readers can look at once. It does not let you read
    a cache another process holds for writing; DuckDB refuses that either way.

    Args:
        table: Table definition (e.g., SEP, SFP)
        fix: If True, drop the claim so the range is fetched again

    Returns:
        The tickers found, in order.

    Example:
        from ndl_cache import SEP, validate_sync_bounds

        for ticker in validate_sync_bounds(SEP, fix=True):
            print(f'{ticker} claimed coverage with no rows')
    """
    db_path = get_db_path()
    if not Path(db_path).exists():
        return []
    conn = duckdb.connect(db_path, read_only=not fix)
    try:
        data_table = table.safe_table_name()
        sync_table = table.sync_bounds_table_name()
        present = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name IN (?, ?)
        """, [data_table, sync_table]).fetchone()[0]
        if present < 2:
            return []

        empty = [row[0] for row in conn.execute(f"""
            SELECT b.ticker FROM {sync_table} b
            WHERE b.synced_from IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM {data_table} d WHERE d.ticker = b.ticker)
            ORDER BY b.ticker
        """).fetchall()]

        if fix and empty:
            conn.executemany(
                f'DELETE FROM {sync_table} WHERE ticker = ?',
                [[ticker] for ticker in empty])
        return empty
    finally:
        conn.close()


def validate_sync_bounds(table: TableDef, fix: bool = False) -> list[str]:
    """
    Find tickers claiming a synced range while holding no rows (sync version).

    See async_validate_sync_bounds for details.
    """
    return asyncio.run(async_validate_sync_bounds(table, fix=fix))
