"""
Async cache layer using aioduckdb for non-blocking DuckDB operations.

Provides async_query() for async access and query() for sync access.
"""
import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import aioduckdb
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

# Lock per table for entire query operations (read-fetch-write cycle)
_table_query_locks: dict[str, asyncio.Lock] = {}


def _get_table_lock(table_name: str) -> asyncio.Lock:
    """Get or create a lock for a specific table's query operations."""
    if table_name not in _table_query_locks:
        _table_query_locks[table_name] = asyncio.Lock()
    return _table_query_locks[table_name]


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
        """Get or create connection without table initialization."""
        if self._conn is None:
            self._conn = await aioduckdb.connect(self._db_path)
        return self._conn

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
        """Check if cached data is stale and invalidate if needed."""
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

        async def check_ticker(ticker: str) -> tuple[str, bool]:
            try:
                df = await client.get_table(
                    self.table.name,
                    columns=['lastupdated'],
                    ticker=ticker,
                    paginate=False
                )
                if len(df) > 0 and 'lastupdated' in df.columns:
                    api_lastupdated = str(df['lastupdated'].max())[:10]
                    cached_lastupdated = sync_bounds[ticker].get('max_lastupdated')
                    is_stale = cached_lastupdated and api_lastupdated > cached_lastupdated
                    return (ticker, is_stale)
                return (ticker, False)
            except Exception:
                return (ticker, False)

        results = await asyncio.gather(*[check_ticker(t) for t in tickers_to_check])

        stale_tickers = []
        for ticker, is_stale in results:
            if is_stale:
                stale_tickers.append(ticker)
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

        # Extract max_lastupdated per ticker
        max_lastupdated_by_ticker = {}
        if len(queried) > 0 and 'lastupdated' in queried.columns:
            for ticker in queried['ticker'].unique():
                ticker_data = queried[queried['ticker'] == ticker]
                max_lu = ticker_data['lastupdated'].max()
                if pd.notna(max_lu):
                    max_lastupdated_by_ticker[ticker] = str(max_lu)[:10]

        # Update sync bounds
        today = datetime.now().strftime('%Y-%m-%d')
        for filters in filter_sets:
            ticker = filters.get('ticker')
            if ticker:
                tickers = ticker if isinstance(ticker, list) else [ticker]
                for t in tickers:
                    if self.table.date_column is None:
                        await self._mark_ticker_synced(t, max_lastupdated_by_ticker.get(t))
                    else:
                        date_gte = filters.get(f'{self.table.date_column}_gte', '1900-01-01')
                        date_lte = filters.get(f'{self.table.date_column}_lte', today)
                        await self._update_sync_bounds(t, date_gte, date_lte, max_lastupdated_by_ticker.get(t))

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

        await conn.executemany(f"""
            INSERT INTO {self.table.safe_table_name()} ({col_names})
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
        lock = _get_table_lock(self.table.name)
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
