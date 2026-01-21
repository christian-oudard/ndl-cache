"""
Async cache layer using aioduckdb for non-blocking DuckDB operations.

Provides async_query() for async access and query() for sync access.
"""
import asyncio
import os
import weakref
from datetime import datetime, timedelta
from pathlib import Path

import aioduckdb
import duckdb
import pandas as pd

from .async_client import AsyncNDLClient
from .cover import solve_cover, find_gaps
from .tables import TableDef, TRADING_DAYS_PER_YEAR


# Optimal parallelization level based on benchmarking ~10k row requests
MAX_FETCH_WORKERS = 4

# NDL API page limit
NDL_PAGE_LIMIT = 10000

# Split threshold - stay well under page limit
NDL_SPLIT_THRESHOLD = 9000

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
                self._conn = await aioduckdb.connect(self._db_path)
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
        """Create data table if it doesn't exist."""
        conn = await self._get_conn()
        cols = list(self.table.index_columns) + data_columns
        col_defs = [f'{col} {self.table.column_types.get(col, "DOUBLE")}' for col in cols]
        pk = ', '.join(self.table.index_columns)

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table.safe_table_name()} (
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

    async def _check_and_invalidate_stale(self, tickers: list[str]):
        """Check if cached data is stale and invalidate if needed.

        Uses batched API queries to check multiple tickers at once, reducing
        the number of API calls from N to 1 (or a few batches for large lists).
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

        client = await self._get_ndl_client()

        # Batch query all tickers at once instead of one API call per ticker
        try:
            df = await client.get_table(
                self.table.name,
                columns=['ticker', 'lastupdated'],
                ticker=tickers_to_check,
                paginate=True
            )
        except Exception:
            # On error, skip staleness check but still update last_staleness_check
            df = None

        # Build map of ticker -> max lastupdated from API response
        api_lastupdated_map: dict[str, str] = {}
        if df is not None and len(df) > 0 and 'lastupdated' in df.columns:
            # Group by ticker and get max lastupdated for each
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker]
                max_lu = ticker_data['lastupdated'].max()
                if pd.notna(max_lu):
                    api_lastupdated_map[ticker] = str(max_lu)[:10]

        # Compare against cached values to find stale tickers
        stale_tickers = []
        for ticker in tickers_to_check:
            api_lastupdated = api_lastupdated_map.get(ticker)
            cached_lastupdated = sync_bounds[ticker].get('max_lastupdated')
            if api_lastupdated and cached_lastupdated and api_lastupdated > cached_lastupdated:
                stale_tickers.append(ticker)

            # Update last_staleness_check for all checked tickers
            await conn.execute(f"""
                UPDATE {self.table.sync_bounds_table_name()}
                SET last_staleness_check = ?
                WHERE ticker = ?
            """, [today, ticker])

        for ticker in stale_tickers:
            await self._invalidate_ticker(ticker)

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

    def _split_filters(self, filters: dict, max_rows: int = NDL_SPLIT_THRESHOLD) -> list[dict]:
        """Split a filter set into chunks that each return < max_rows."""
        # Skip splitting for tables with unknown row density (e.g., sparse ACTIONS table)
        if self.table.rows_per_year is None:
            return [filters]

        est_rows = self._estimate_rows(filters)
        if est_rows < max_rows:
            return [filters]

        date_col = self.table.date_column
        ticker = filters.get('ticker')
        date_gte = filters.get(f'{date_col}_gte')
        date_lte = filters.get(f'{date_col}_lte')

        # Strategy 1: Split by tickers
        if isinstance(ticker, list) and len(ticker) > 1:
            est_rows_per_ticker = self._estimate_rows_for_range(date_gte, date_lte)
            tickers_per_chunk = max(1, max_rows // est_rows_per_ticker)

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

        conn = await self._get_conn()
        queried = await self._fetch_parallel(filter_sets)

        # Extract actual date ranges and max_lastupdated per ticker from returned data
        # This ensures sync bounds reflect what was actually stored, not what was requested
        ticker_stats: dict[str, dict] = {}
        if len(queried) > 0:
            date_col = self.table.date_column
            for ticker in queried['ticker'].unique():
                ticker_data = queried[queried['ticker'] == ticker]
                stats: dict = {}

                if 'lastupdated' in ticker_data.columns:
                    max_lu = ticker_data['lastupdated'].max()
                    if pd.notna(max_lu):
                        stats['max_lastupdated'] = str(max_lu)[:10]

                if date_col and date_col in ticker_data.columns:
                    min_date = ticker_data[date_col].min()
                    max_date = ticker_data[date_col].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        stats['min_date'] = str(min_date)[:10]
                        stats['max_date'] = str(max_date)[:10]

                if stats:
                    ticker_stats[ticker] = stats

        # Update sync bounds based on ACTUAL data received, not requested ranges
        for ticker, stats in ticker_stats.items():
            if self.table.date_column is None:
                await self._mark_ticker_synced(ticker, stats.get('max_lastupdated'))
            else:
                min_date = stats.get('min_date')
                max_date = stats.get('max_date')
                if min_date and max_date:
                    await self._update_sync_bounds(
                        ticker, min_date, max_date, stats.get('max_lastupdated')
                    )

        if len(queried) == 0:
            return 0

        data_columns = [c for c in self.table.query_columns if c in queried.columns]
        await self._ensure_data_table(data_columns)

        store_df = queried[list(self.table.index_columns) + data_columns].copy()

        # Insert rows using executemany for aioduckdb compatibility
        cols = list(self.table.index_columns) + data_columns
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)

        # Dedupe in case API returns duplicate rows
        store_df = store_df.drop_duplicates(subset=list(self.table.index_columns))
        rows = [tuple(row) for row in store_df.itertuples(index=False, name=None)]

        # Use INSERT OR REPLACE to handle duplicates from parallel queries or retries.
        # This prevents constraint errors when overlapping data is fetched multiple times.
        await conn.executemany(f"""
            INSERT OR REPLACE INTO {self.table.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, rows)

        return len(store_df)

    async def get_cached(self, **filters) -> pd.DataFrame:
        """Get data from local cache."""
        conn = await self._get_conn()

        where_clauses = []
        params = []
        for key, value in filters.items():
            if key.endswith('_gte'):
                where_clauses.append(f"{key[:-4]} >= ?")
                params.append(value)
            elif key.endswith('_lte'):
                where_clauses.append(f"{key[:-4]} <= ?")
                params.append(value)
            elif isinstance(value, list):
                placeholders = ', '.join(['?'] * len(value))
                where_clauses.append(f"{key} IN ({placeholders})")
                params.extend(value)
            else:
                where_clauses.append(f"{key} = ?")
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
            ORDER BY {', '.join(self.table.index_columns)}
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

    def _find_tickers_with_gaps(self, df: pd.DataFrame, gap_threshold_days: int = 14) -> list[str]:
        """Find tickers with date gaps larger than threshold in cached data.

        This is a quick check on already-loaded data to detect corrupted cache entries.
        """
        if df.empty or self.table.date_column is None:
            return []

        date_col = self.table.date_column
        if date_col not in df.index.names:
            return []

        tickers_with_gaps = []

        # Get unique tickers from the index
        if 'ticker' in df.index.names:
            ticker_level = df.index.names.index('ticker')
            unique_tickers = df.index.get_level_values(ticker_level).unique()

            for ticker in unique_tickers:
                try:
                    ticker_data = df.loc[ticker] if ticker_level == 0 else df.xs(ticker, level='ticker')
                    if isinstance(ticker_data, pd.Series):
                        continue  # Only one row, no gaps possible

                    dates = ticker_data.index.get_level_values(date_col) if date_col in ticker_data.index.names else None
                    if dates is None or len(dates) < 2:
                        continue

                    # Sort dates and check for gaps
                    sorted_dates = sorted(dates)
                    for i in range(1, len(sorted_dates)):
                        gap = (sorted_dates[i] - sorted_dates[i-1]).days
                        if gap > gap_threshold_days:
                            tickers_with_gaps.append(ticker)
                            break
                except (KeyError, TypeError):
                    continue

        return tickers_with_gaps

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

        # Lock the entire read-fetch-write cycle per table to prevent race conditions
        loop = asyncio.get_running_loop()
        lock = _get_table_lock(self.table.name, loop)
        async with lock:
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

            # Quick gap check: if any tickers have large date gaps, invalidate and re-fetch
            if not result.empty and self.table.date_column:
                tickers_with_gaps = self._find_tickers_with_gaps(result)
                if tickers_with_gaps:
                    # Invalidate corrupted tickers and re-fetch
                    for ticker in tickers_with_gaps:
                        await self._invalidate_ticker(ticker)

                    # Re-run sync for the corrupted tickers
                    date_gte = filters.get(f'{self.table.date_column}_gte')
                    date_lte = filters.get(f'{self.table.date_column}_lte')
                    if date_gte and date_lte:
                        refetch_filters = [{
                            'ticker': tickers_with_gaps,
                            f'{self.table.date_column}_gte': date_gte,
                            f'{self.table.date_column}_lte': date_lte,
                        }]
                        await self._sync_parallel(refetch_filters)

                    # Re-fetch from cache
                    result = await self.get_cached(**filters)

        if len(result) == 0:
            return pd.DataFrame()

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            result = result[[c for c in columns if c in result.columns]]

        return result


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

async def async_query(
    table: TableDef,
    columns: list[str] | str | None = None,
    **filters,
) -> pd.DataFrame:
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
        return await mgr.query(columns=columns, **filters)


def query(
    table: TableDef,
    columns: list[str] | str | None = None,
    **filters,
) -> pd.DataFrame:
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
    return asyncio.run(async_query(table, columns=columns, **filters))


async def async_validate_sync_bounds(
    table: TableDef,
    fix: bool = False,
    gap_threshold_days: int = 14,
) -> list[dict]:
    """
    Validate sync bounds against actual cached data for a table.

    Detects sync bounds that claim a date range but the actual data is missing,
    has gaps, or doesn't match the claimed range.

    Args:
        table: Table definition (e.g., SEP, SFP)
        fix: If True, clear corrupted sync bounds so data will be re-fetched
        gap_threshold_days: Report gaps larger than this many days

    Returns:
        List of dicts describing issues found:
        [{'ticker': 'VT', 'issue': 'no_data', 'details': '...'}, ...]

    Example:
        from ndl_cache import SFP, async_validate_sync_bounds

        issues = await async_validate_sync_bounds(SFP, fix=True)
        for issue in issues:
            print(f"{issue['ticker']}: {issue['issue']}")
    """
    import duckdb

    db_path = get_db_path()
    conn = duckdb.connect(db_path)
    issues = []

    try:
        # Check if tables exist
        data_table = table.safe_table_name()
        sync_table = table.sync_bounds_table_name()

        tables_exist = conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name IN ('{data_table}', '{sync_table}')
        """).fetchone()[0]

        if tables_exist < 2:
            return []  # Tables don't exist yet

        # Get all sync bounds
        sync_bounds = conn.execute(f"""
            SELECT ticker, synced_from, synced_to
            FROM {sync_table}
        """).fetchall()

        date_col = table.date_column
        if date_col is None:
            return []  # Non-date tables don't have this issue

        for ticker, synced_from, synced_to in sync_bounds:
            # Get actual data range for this ticker
            result = conn.execute(f"""
                SELECT MIN({date_col}), MAX({date_col}), COUNT(*)
                FROM {data_table}
                WHERE ticker = ?
            """, [ticker]).fetchone()

            actual_min, actual_max, actual_count = result

            issue = None

            if actual_count == 0:
                issue = {
                    'ticker': ticker,
                    'issue': 'no_data',
                    'details': f'Sync bounds claim {synced_from} to {synced_to} but no data exists',
                }
            else:
                # Check for start mismatch (actual data starts later than claimed)
                if synced_from and actual_min:
                    synced_from_str = str(synced_from)[:10]
                    actual_min_str = str(actual_min)[:10]
                    if actual_min_str > synced_from_str:
                        from datetime import datetime
                        gap = (datetime.strptime(actual_min_str, '%Y-%m-%d') -
                               datetime.strptime(synced_from_str, '%Y-%m-%d')).days
                        if gap > gap_threshold_days:
                            issue = {
                                'ticker': ticker,
                                'issue': 'start_gap',
                                'details': f'Data starts at {actual_min_str} but sync claims {synced_from_str} (gap: {gap} days)',
                            }

                # Check for end mismatch (claimed end is later than actual data)
                if not issue and synced_to and actual_max:
                    synced_to_str = str(synced_to)[:10]
                    actual_max_str = str(actual_max)[:10]
                    if synced_to_str > actual_max_str:
                        from datetime import datetime
                        gap = (datetime.strptime(synced_to_str, '%Y-%m-%d') -
                               datetime.strptime(actual_max_str, '%Y-%m-%d')).days
                        if gap > gap_threshold_days:
                            issue = {
                                'ticker': ticker,
                                'issue': 'end_gap',
                                'details': f'Data ends at {actual_max_str} but sync claims {synced_to_str} (gap: {gap} days)',
                            }

            if issue:
                issues.append(issue)

                if fix:
                    # Clear sync bounds and data so it will be re-fetched
                    conn.execute(f"DELETE FROM {sync_table} WHERE ticker = ?", [ticker])
                    conn.execute(f"DELETE FROM {data_table} WHERE ticker = ?", [ticker])
                    issue['fixed'] = True

    finally:
        conn.close()

    return issues


def validate_sync_bounds(
    table: TableDef,
    fix: bool = False,
    gap_threshold_days: int = 14,
) -> list[dict]:
    """
    Validate sync bounds against actual cached data for a table (sync version).

    See async_validate_sync_bounds for details.
    """
    return asyncio.run(async_validate_sync_bounds(table, fix=fix, gap_threshold_days=gap_threshold_days))
