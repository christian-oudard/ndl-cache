import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import nasdaqdatalink as ndl
import pandas as pd
import warnings

from .cover import solve_cover, find_gaps

# Optimal parallelization level based on benchmarking ~10k row requests
# Higher values hit server-side throttling with diminishing returns
MAX_FETCH_WORKERS = 4

# NDL API page limit - requests returning more get paginated (slow) or truncated
NDL_PAGE_LIMIT = 10000

# Split threshold - stay well under page limit so hitting 10k is always an error
# This buffer accounts for estimation inaccuracy (holidays, new listings, etc.)
NDL_SPLIT_THRESHOLD = 9000

# Approximate trading days per calendar year
TRADING_DAYS_PER_YEAR = 252

def get_db_path() -> str:
    """Get database path from NDL_CACHE_DB_PATH env var or default to ~/.cache/ndl_cache/."""
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


class CachedTable:
    """Base class for cached NDL tables."""
    table_name: str
    index_columns: list[str]
    query_columns: list[str]
    date_column: str = 'date'  # Column used for date-based filtering and sync bounds
    column_types: dict[str, str] = {}  # Override column types (default: DOUBLE)
    rows_per_year: int = TRADING_DAYS_PER_YEAR  # Expected rows per ticker per year (for request size estimation)
    sync_delay_days: int = 0  # Don't mark recent dates as synced (data may still appear)

    def __init__(self):
        self.conn = duckdb.connect(get_db_path())
        self._ensure_sync_bounds_table()

    def _ensure_sync_bounds_table(self):
        """Create sync_bounds table if it doesn't exist."""
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._sync_bounds_table_name()} (
                ticker VARCHAR PRIMARY KEY,
                synced_from DATE,
                synced_to DATE,
                max_lastupdated DATE,
                last_staleness_check DATE
            )
        """)

    def _ensure_data_table(self, data_columns: list[str]):
        """Create data table if it doesn't exist, with given columns."""
        cols = self.index_columns + data_columns
        col_defs = [f'{col} {self.column_types.get(col, "DOUBLE")}' for col in cols]
        pk = ', '.join(self.index_columns)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._safe_table_name()} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk})
            )
        """)

    def _safe_table_name(self) -> str:
        return self.table_name.replace('/', '_').lower()

    def _sync_bounds_table_name(self) -> str:
        return f"{self._safe_table_name()}_sync_bounds"

    def _get_sync_bounds(self, tickers: list[str]) -> dict[str, dict | None]:
        """
        Get sync bounds for given tickers.
        Returns dict mapping ticker -> {synced_from, synced_to, max_lastupdated, last_staleness_check} or None if not synced.
        """
        if not tickers:
            return {}

        placeholders = ', '.join(['?'] * len(tickers))
        result = self.conn.execute(f"""
            SELECT ticker, synced_from, synced_to, max_lastupdated, last_staleness_check
            FROM {self._sync_bounds_table_name()}
            WHERE ticker IN ({placeholders})
        """, tickers).fetchall()

        bounds = {ticker: None for ticker in tickers}
        for ticker, synced_from, synced_to, max_lastupdated, last_staleness_check in result:
            bounds[ticker] = {
                'synced_from': str(synced_from)[:10] if synced_from else None,
                'synced_to': str(synced_to)[:10] if synced_to else None,
                'max_lastupdated': str(max_lastupdated)[:10] if max_lastupdated else None,
                'last_staleness_check': str(last_staleness_check)[:10] if last_staleness_check else None,
            }

        return bounds

    def _update_sync_bounds(self, ticker: str, from_date: str, to_date: str, max_lastupdated: str | None = None):
        """
        Update sync bounds for a ticker, expanding the existing range.
        Caps to_date by sync_delay_days to avoid marking recent dates as synced.
        Also updates max_lastupdated if provided.
        """
        effective_to = _effective_sync_date(to_date, self.sync_delay_days)

        # Get existing bounds
        existing = self.conn.execute(f"""
            SELECT synced_from, synced_to, max_lastupdated
            FROM {self._sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker]).fetchone()

        if existing:
            old_from, old_to = str(existing[0])[:10], str(existing[1])[:10]
            old_max_lastupdated = str(existing[2])[:10] if existing[2] else None
            new_from = min(from_date, old_from)
            new_to = max(effective_to, old_to)
            # Update max_lastupdated if new one is greater
            if max_lastupdated and (not old_max_lastupdated or max_lastupdated > old_max_lastupdated):
                new_max_lastupdated = max_lastupdated
            else:
                new_max_lastupdated = old_max_lastupdated
        else:
            new_from = from_date
            new_to = effective_to
            new_max_lastupdated = max_lastupdated

        # Set last_staleness_check to today when syncing (we just got fresh data)
        today = datetime.now().strftime('%Y-%m-%d')
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {self._sync_bounds_table_name()}
            (ticker, synced_from, synced_to, max_lastupdated, last_staleness_check)
            VALUES (?, ?, ?, ?, ?)
        """, [ticker, new_from, new_to, new_max_lastupdated, today])

    def _mark_ticker_synced(self, ticker: str, max_lastupdated: str | None = None):
        """
        Mark a ticker as synced for tables without date columns (e.g., TICKERS table).
        Uses NULL for synced_from/synced_to but still tracks max_lastupdated and staleness.
        """
        today = datetime.now().strftime('%Y-%m-%d')

        # Check for existing entry to preserve max_lastupdated if newer
        existing = self.conn.execute(f"""
            SELECT max_lastupdated FROM {self._sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker]).fetchone()

        if existing and existing[0]:
            old_max = str(existing[0])[:10]
            if max_lastupdated and max_lastupdated > old_max:
                new_max = max_lastupdated
            else:
                new_max = old_max
        else:
            new_max = max_lastupdated

        self.conn.execute(f"""
            INSERT OR REPLACE INTO {self._sync_bounds_table_name()}
            (ticker, synced_from, synced_to, max_lastupdated, last_staleness_check)
            VALUES (?, NULL, NULL, ?, ?)
        """, [ticker, new_max, today])

    def _check_and_invalidate_stale(self, tickers: list[str]):
        """
        Check if cached data is stale and invalidate if needed.
        Only checks once per day per ticker to minimize API calls.

        Staleness is detected by comparing cached max_lastupdated with
        the current lastupdated from the API.
        """
        if not tickers:
            return

        today = datetime.now().strftime('%Y-%m-%d')
        sync_bounds = self._get_sync_bounds(tickers)

        # Find tickers that need staleness check (not checked today and have cached data)
        tickers_to_check = []
        for ticker in tickers:
            bounds = sync_bounds.get(ticker)
            if bounds is None:
                continue  # Not cached yet, will be fetched fresh
            last_check = bounds.get('last_staleness_check')
            if last_check == today:
                continue  # Already checked today
            tickers_to_check.append(ticker)

        if not tickers_to_check:
            return

        # Query API for one recent row per ticker to get current lastupdated
        # Use parallel requests to speed up cold start
        def check_ticker_staleness(ticker):
            """Check if a ticker's cached data is stale. Returns (ticker, is_stale)."""
            try:
                # Get one row to check lastupdated
                row = ndl.get_table(
                    self.table_name,
                    ticker=ticker,
                    qopts={'columns': ['lastupdated']},
                    paginate=False
                )
                if len(row) > 0 and 'lastupdated' in row.columns:
                    api_lastupdated = str(row['lastupdated'].max())[:10]
                    cached_lastupdated = sync_bounds[ticker].get('max_lastupdated')
                    is_stale = cached_lastupdated and api_lastupdated > cached_lastupdated
                    return (ticker, is_stale)
                return (ticker, False)
            except Exception:
                # If API check fails, skip this ticker
                return (ticker, False)

        # Run staleness checks in parallel (same worker count as fetch operations)
        stale_tickers = []
        with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
            results = list(executor.map(check_ticker_staleness, tickers_to_check))

        for ticker, is_stale in results:
            if is_stale:
                stale_tickers.append(ticker)
            # Update last_staleness_check regardless of staleness
            self.conn.execute(f"""
                UPDATE {self._sync_bounds_table_name()}
                SET last_staleness_check = ?
                WHERE ticker = ?
            """, [today, ticker])

        # Invalidate stale tickers by deleting their cached data and sync bounds
        for ticker in stale_tickers:
            self._invalidate_ticker(ticker)

    def _invalidate_ticker(self, ticker: str):
        """Delete all cached data and sync bounds for a ticker."""
        # Delete from data table
        table_exists = self.conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{self._safe_table_name()}'
        """).fetchone()[0] > 0

        if table_exists:
            self.conn.execute(f"""
                DELETE FROM {self._safe_table_name()}
                WHERE ticker = ?
            """, [ticker])

        # Delete from sync bounds
        self.conn.execute(f"""
            DELETE FROM {self._sync_bounds_table_name()}
            WHERE ticker = ?
        """, [ticker])

    def _estimate_rows_for_range(self, date_gte: str | None, date_lte: str | None) -> int:
        """Estimate number of rows per ticker for a date range."""
        if not (date_gte and date_lte):
            return 1
        start = datetime.strptime(date_gte, '%Y-%m-%d')
        end = datetime.strptime(date_lte, '%Y-%m-%d')
        calendar_days = (end - start).days + 1
        return max(1, int(calendar_days * self.rows_per_year / 365))

    def _estimate_rows(self, filters: dict) -> int:
        """Estimate number of rows a filter set will return."""
        ticker = filters.get('ticker')
        n_tickers = len(ticker) if isinstance(ticker, list) else 1
        date_col = self.date_column
        est_rows_per_ticker = self._estimate_rows_for_range(
            filters.get(f'{date_col}_gte'),
            filters.get(f'{date_col}_lte')
        )
        return n_tickers * est_rows_per_ticker

    def _split_filters(self, filters: dict, max_rows: int = NDL_SPLIT_THRESHOLD) -> list[dict]:
        """
        Split a filter set into chunks that each return < max_rows.
        Splits by tickers first, then by date ranges if needed.
        """
        est_rows = self._estimate_rows(filters)
        if est_rows < max_rows:
            return [filters]

        date_col = self.date_column
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
                # Recursively split if still too large (e.g., very long date range)
                chunks.extend(self._split_filters(chunk_filters, max_rows))
            return chunks

        # Strategy 2: Split by date range (for single ticker with long history)
        if date_gte and date_lte:
            start = datetime.strptime(date_gte, '%Y-%m-%d')
            end = datetime.strptime(date_lte, '%Y-%m-%d')

            # Calculate days per chunk to stay under max_rows
            # calendar_days_per_chunk = max_rows * 365 / rows_per_year
            calendar_days_per_chunk = max(1, int(max_rows * 365 / self.rows_per_year))

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

        # Can't split further
        return [filters]

    def _compute_optimal_fetches(
        self,
        tickers: list[str],
        date_gte: str,
        date_lte: str,
        max_rows: int = NDL_SPLIT_THRESHOLD,
    ) -> list[dict]:
        """
        Compute optimal fetch filter sets using set-cover solver.
        Returns list of filter dicts that cover all gaps with minimal requests.
        """
        sync_bounds_raw = self._get_sync_bounds(tickers)
        # Convert to format expected by find_gaps: ticker -> (synced_from, synced_to) or None
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

        # Convert Request objects to filter dicts using table's date column
        date_col = self.date_column
        return [
            {
                'ticker': list(req.tickers) if len(req.tickers) > 1 else list(req.tickers)[0],
                f'{date_col}_gte': req.start,
                f'{date_col}_lte': req.end,
            }
            for req in requests
        ]

    def fetch_from_ndl(self, **filters) -> pd.DataFrame:
        """
        Fetch data from NDL API.
        Uses paginate=True as safety net to never lose data.
        Warns if pagination was needed (indicates splitting should have prevented this).
        """
        # Convert our filter format to NDL format
        ndl_filters = {}
        range_filters = {}  # For gte/lte style filters

        for key, value in filters.items():
            if key.endswith('_gte'):
                col = key[:-4]
                range_filters.setdefault(col, {})['gte'] = value
            elif key.endswith('_lte'):
                col = key[:-4]
                range_filters.setdefault(col, {})['lte'] = value
            else:
                ndl_filters[key] = value

        # Merge range filters
        ndl_filters.update(range_filters)

        result = ndl.get_table(
            self.table_name,
            qopts={'columns': self.index_columns + self.query_columns},
            paginate=True,  # Safety net - never lose data
            **ndl_filters
        )

        # Track estimation accuracy
        estimated = self._estimate_rows(filters)
        actual = len(result)

        # Warn if pagination was needed (splitting should have prevented this)
        if actual >= NDL_PAGE_LIMIT:
            warnings.warn(
                f"NDL request returned {actual} rows (required pagination). "
                f"Estimated {estimated}, splitting threshold is {NDL_SPLIT_THRESHOLD}. "
                f"Filters: {filters}",
                UserWarning
            )
        # Warn if estimate was significantly off (>50% error)
        elif estimated > 0:
            error_ratio = abs(actual - estimated) / estimated
            if error_ratio > 0.5 and actual > 100:  # Only care about non-trivial requests
                warnings.warn(
                    f"NDL row estimate was off by {error_ratio:.0%}: "
                    f"estimated {estimated}, got {actual}. Filters: {filters}",
                    UserWarning
                )

        return result

    def _fetch_parallel(self, filter_sets: list[dict]) -> pd.DataFrame:
        """
        Fetch multiple filter sets in parallel using a worker pool.
        Workers pull from the queue until all requests are complete.
        Automatically splits any filter sets that would exceed page limit.
        Returns combined DataFrame.
        """
        if not filter_sets:
            return pd.DataFrame()

        # Split any oversized filter sets to stay under page limit
        all_chunks = []
        for filters in filter_sets:
            all_chunks.extend(self._split_filters(filters))

        if len(all_chunks) == 1:
            return self.fetch_from_ndl(**all_chunks[0])

        with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
            results = list(executor.map(lambda f: self.fetch_from_ndl(**f), all_chunks))

        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True)

    def _sync_parallel(self, filter_sets: list[dict]) -> int:
        """
        Fetch multiple filter sets in parallel and sync to cache.
        Also updates sync bounds for each ticker in each filter set.
        Returns total rows synced.
        """
        if not filter_sets:
            return 0

        queried = self._fetch_parallel(filter_sets)

        # Extract max_lastupdated per ticker from queried data
        max_lastupdated_by_ticker = {}
        if len(queried) > 0 and 'lastupdated' in queried.columns:
            for ticker in queried['ticker'].unique():
                ticker_data = queried[queried['ticker'] == ticker]
                max_lu = ticker_data['lastupdated'].max()
                if pd.notna(max_lu):
                    max_lastupdated_by_ticker[ticker] = str(max_lu)[:10]

        # Update sync bounds for all tickers in the filter sets
        # (even if we got no data - we still queried that range)
        today = datetime.now().strftime('%Y-%m-%d')
        for filters in filter_sets:
            ticker = filters.get('ticker')
            if ticker:
                tickers = ticker if isinstance(ticker, list) else [ticker]
                for t in tickers:
                    if self.date_column is None:
                        # Tables without date columns: just mark as synced
                        self._mark_ticker_synced(t, max_lastupdated_by_ticker.get(t))
                    else:
                        date_gte = filters.get(f'{self.date_column}_gte', '1900-01-01')
                        date_lte = filters.get(f'{self.date_column}_lte', today)
                        self._update_sync_bounds(t, date_gte, date_lte, max_lastupdated_by_ticker.get(t))

        if len(queried) == 0:
            return 0

        # Store data columns (everything except index columns)
        data_columns = [c for c in self.query_columns if c in queried.columns]

        # Ensure table exists with correct schema
        self._ensure_data_table(data_columns)

        # Build DataFrame for storage
        store_df = queried[self.index_columns + data_columns].copy()

        # Upsert into DuckDB
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {self._safe_table_name()}
            SELECT * FROM store_df
        """)

        return len(store_df)

    def sync(self, **filters):
        """Sync data from NDL to local cache. Splits large requests automatically."""
        # Use parallel fetch with auto-splitting to avoid pagination
        return self._sync_parallel([filters])

    def get_cached(self, **filters) -> pd.DataFrame:
        """Get data from local cache, indexed by index_columns."""
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

        # Check if table exists
        table_exists = self.conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{self._safe_table_name()}'
        """).fetchone()[0] > 0

        if not table_exists:
            return pd.DataFrame()

        df = self.conn.execute(f"""
            SELECT * FROM {self._safe_table_name()}
            WHERE {where}
            ORDER BY {', '.join(self.index_columns)}
        """, params).df()

        if len(df) > 0:
            # Convert DATE columns to datetime before indexing
            for col in self.index_columns:
                if self.column_types.get(col) == 'DATE':
                    df[col] = pd.to_datetime(df[col])
            df = df.set_index(self.index_columns)

        return df

    def query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Query data from cache, fetching from NDL if not cached.
        Uses set-cover solver to minimize API calls.
        Checks for stale data once per day per ticker.

        Returns DataFrame indexed by index_columns with requested data columns.
        """
        # Normalize ticker to list for consistent handling
        ticker_filter = filters.get('ticker')
        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        elif isinstance(ticker_filter, list):
            tickers = ticker_filter
        else:
            tickers = []

        # Check for stale data and invalidate if needed (once per day per ticker)
        if tickers:
            self._check_and_invalidate_stale(tickers)

        # Handle tables without date columns (e.g., TICKERS table)
        if self.date_column is None:
            if tickers:
                # Sync any tickers that haven't been synced yet
                sync_bounds = self._get_sync_bounds(tickers)
                unsynced = [t for t in tickers if sync_bounds.get(t) is None]
                if unsynced:
                    self._sync_parallel([{'ticker': t} for t in unsynced])
        else:
            # Get date filters using the table's date column
            date_gte = filters.get(f'{self.date_column}_gte')
            date_lte = filters.get(f'{self.date_column}_lte')

            # Sync data from NDL if needed
            if tickers and date_gte and date_lte:
                # Use cover solver to compute optimal fetches for gaps
                optimal_fetches = self._compute_optimal_fetches(tickers, date_gte, date_lte)
                if optimal_fetches:
                    self._sync_parallel(optimal_fetches)
            elif tickers and not date_gte and not date_lte:
                # No date filters - sync full history for unsynced tickers
                sync_bounds = self._get_sync_bounds(tickers)
                unsynced = [t for t in tickers if sync_bounds.get(t) is None]
                if unsynced:
                    self._sync_parallel([{'ticker': t} for t in unsynced])

        # Get final result from cache (already indexed by index_columns)
        result = self.get_cached(**filters)

        if len(result) == 0:
            return pd.DataFrame()

        # Select requested columns (or all if None)
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            result = result[[c for c in columns if c in result.columns]]

        return result


