import pandas as pd
import nasdaqdatalink as ndl
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import duckdb
import warnings

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


class CachedTable:
    """Base class for cached NDL tables."""
    table_name: str
    index_columns: list[str]
    query_columns: list[str]
    immutable_columns: list[str]

    def __init__(self, db_path: str = 'cache.duckdb'):
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self._ensure_table()

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        cols = self.index_columns + self.immutable_columns
        col_defs = []
        for col in cols:
            if col == 'date':
                col_defs.append(f'{col} DATE')
            elif col == 'ticker':
                col_defs.append(f'{col} VARCHAR')
            else:
                col_defs.append(f'{col} DOUBLE')

        pk = ', '.join(self.index_columns)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._safe_table_name()} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk})
            )
        """)

    def _safe_table_name(self) -> str:
        return self.table_name.replace('/', '_').lower()

    @staticmethod
    def immutable_data(queried: pd.DataFrame) -> pd.DataFrame:
        """Convert queried data to immutable storage format. Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def derived_data(immutable: pd.DataFrame) -> pd.DataFrame:
        """Derive output columns from immutable data. Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def _estimate_trading_days(date_gte: str | None, date_lte: str | None) -> int:
        """Estimate number of trading days in a date range."""
        if not (date_gte and date_lte):
            return 1
        start = datetime.strptime(date_gte, '%Y-%m-%d')
        end = datetime.strptime(date_lte, '%Y-%m-%d')
        calendar_days = (end - start).days + 1
        return max(1, int(calendar_days * TRADING_DAYS_PER_YEAR / 365))

    @staticmethod
    def _estimate_rows(filters: dict) -> int:
        """Estimate number of rows a filter set will return."""
        ticker = filters.get('ticker')
        n_tickers = len(ticker) if isinstance(ticker, list) else 1
        est_days = CachedTable._estimate_trading_days(
            filters.get('date_gte'),
            filters.get('date_lte')
        )
        return n_tickers * est_days

    @staticmethod
    def _split_filters(filters: dict, max_rows: int = NDL_SPLIT_THRESHOLD) -> list[dict]:
        """
        Split a filter set into chunks that each return < max_rows.
        Splits by tickers first, then by date ranges if needed.
        """
        est_rows = CachedTable._estimate_rows(filters)
        if est_rows < max_rows:
            return [filters]

        ticker = filters.get('ticker')
        date_gte = filters.get('date_gte')
        date_lte = filters.get('date_lte')

        # Strategy 1: Split by tickers
        if isinstance(ticker, list) and len(ticker) > 1:
            est_days = CachedTable._estimate_trading_days(date_gte, date_lte)
            tickers_per_chunk = max(1, max_rows // est_days)

            chunks = []
            for i in range(0, len(ticker), tickers_per_chunk):
                chunk_tickers = ticker[i:i + tickers_per_chunk]
                chunk_filters = {**filters, 'ticker': chunk_tickers if len(chunk_tickers) > 1 else chunk_tickers[0]}
                # Recursively split if still too large (e.g., very long date range)
                chunks.extend(CachedTable._split_filters(chunk_filters, max_rows))
            return chunks

        # Strategy 2: Split by date range (for single ticker with long history)
        if date_gte and date_lte:
            start = datetime.strptime(date_gte, '%Y-%m-%d')
            end = datetime.strptime(date_lte, '%Y-%m-%d')
            total_days = (end - start).days + 1

            # Calculate days per chunk to stay under max_rows
            # For single ticker: rows ≈ trading_days ≈ calendar_days * 252/365
            calendar_days_per_chunk = max(1, int(max_rows * 365 / TRADING_DAYS_PER_YEAR))

            chunks = []
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(chunk_start + timedelta(days=calendar_days_per_chunk - 1), end)
                chunk_filters = {
                    **filters,
                    'date_gte': chunk_start.strftime('%Y-%m-%d'),
                    'date_lte': chunk_end.strftime('%Y-%m-%d'),
                }
                chunks.append(chunk_filters)
                chunk_start = chunk_end + timedelta(days=1)
            return chunks

        # Can't split further
        return [filters]

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
        Returns total rows synced.
        """
        queried = self._fetch_parallel(filter_sets)
        if len(queried) == 0:
            return 0

        immutable = self.immutable_data(queried)

        # Add index columns
        for col in self.index_columns:
            immutable[col] = queried[col]

        # Reorder columns to match table schema
        col_order = self.index_columns + self.immutable_columns
        immutable = immutable[col_order]

        # Upsert into DuckDB
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {self._safe_table_name()}
            SELECT * FROM immutable
        """)

        return len(immutable)

    def sync(self, **filters):
        """Sync data from NDL to local cache. Splits large requests automatically."""
        # Use parallel fetch with auto-splitting to avoid pagination
        return self._sync_parallel([filters])

    def get_cached(self, **filters) -> pd.DataFrame:
        """Get data from local cache."""
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

        return self.conn.execute(f"""
            SELECT * FROM {self._safe_table_name()}
            WHERE {where}
        """, params).df()

    def _compute_date_gaps(self, cached_data: pd.DataFrame, date_gte: str, date_lte: str) -> list[tuple[str, str]]:
        """
        Compute date range gaps in cached data.
        Returns list of (gap_start, gap_end) tuples.
        """
        if not (date_gte and date_lte):
            return []

        if 'date' not in cached_data.columns or len(cached_data) == 0:
            return [(date_gte, date_lte)]

        cached_dates = set(str(d)[:10] for d in cached_data['date'])
        if not cached_dates:
            return [(date_gte, date_lte)]

        cached_min = min(cached_dates)
        cached_max = max(cached_dates)
        gaps = []

        # Gap before cached range
        if date_gte < cached_min:
            day_before_min = (datetime.strptime(cached_min, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            gaps.append((date_gte, day_before_min))

        # Gap after cached range
        if date_lte > cached_max:
            day_after_max = (datetime.strptime(cached_max, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            gaps.append((day_after_max, date_lte))

        # Gaps within cached range
        if cached_min <= date_gte and date_lte <= cached_max:
            start = datetime.strptime(date_gte, '%Y-%m-%d')
            end = datetime.strptime(date_lte, '%Y-%m-%d')
            gap_start = None
            current = start
            while current <= end:
                date_str = current.strftime('%Y-%m-%d')
                if date_str not in cached_dates:
                    if gap_start is None:
                        gap_start = date_str
                    gap_end = date_str
                else:
                    if gap_start is not None:
                        gaps.append((gap_start, gap_end))
                        gap_start = None
                current += timedelta(days=1)
            if gap_start is not None:
                gaps.append((gap_start, gap_end))

        return gaps

    def _fill_date_gaps(self, cached_data: pd.DataFrame, filters: dict) -> bool:
        """
        Check for date range gaps in cached data and fetch missing dates.
        Returns True if any data was fetched.
        """
        gaps = self._compute_date_gaps(
            cached_data,
            filters.get('date_gte'),
            filters.get('date_lte')
        )
        if not gaps:
            return False

        gap_filter_sets = [
            {**filters, 'date_gte': gap_start, 'date_lte': gap_end}
            for gap_start, gap_end in gaps
        ]
        self._sync_parallel(gap_filter_sets)
        return True

    def query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Query data, returning immutable and/or derived columns.
        Fetches from NDL if not cached, using set intersection to avoid re-fetching.
        """
        # Get from cache
        immutable = self.get_cached(**filters)

        # Determine what's missing and fetch it
        ticker_filter = filters.get('ticker')
        if isinstance(ticker_filter, list):
            # Multi-ticker query: prefer single overfetch call if under page limit
            date_gte = filters.get('date_gte')
            date_lte = filters.get('date_lte')

            # Check if any ticker has gaps
            has_gaps = False
            for ticker in ticker_filter:
                ticker_data = immutable[immutable['ticker'] == ticker] if len(immutable) > 0 else pd.DataFrame()
                if self._compute_date_gaps(ticker_data, date_gte, date_lte):
                    has_gaps = True
                    break

            if has_gaps:
                est_rows = self._estimate_rows(filters)

                if est_rows < NDL_SPLIT_THRESHOLD:
                    # Single overfetch call for entire rectangle (may split if needed)
                    self.sync(**filters)
                else:
                    # Large rectangle - collect only missing gaps to minimize data transfer
                    all_gap_filters = []
                    for ticker in ticker_filter:
                        ticker_data = immutable[immutable['ticker'] == ticker] if len(immutable) > 0 else pd.DataFrame()
                        ticker_filters = {**filters, 'ticker': ticker}
                        gaps = self._compute_date_gaps(ticker_data, date_gte, date_lte)
                        for gap_start, gap_end in gaps:
                            all_gap_filters.append({**ticker_filters, 'date_gte': gap_start, 'date_lte': gap_end})

                    if all_gap_filters:
                        self._sync_parallel(all_gap_filters)

                immutable = self.get_cached(**filters)
        elif len(immutable) == 0:
            # Cache empty: fetch everything
            self.sync(**filters)
            immutable = self.get_cached(**filters)
        else:
            # Check for date range gaps
            if self._fill_date_gaps(immutable, filters):
                immutable = self.get_cached(**filters)

        if len(immutable) == 0:
            return pd.DataFrame()

        # Derive computed columns
        derived = self.derived_data(immutable)

        # Build result from index + immutable + derived
        result = pd.DataFrame()
        for col in self.index_columns:
            result[col] = immutable[col]

        # Select requested columns (or all if None)
        if columns is None:
            # Return all: immutable + derived
            for col in self.immutable_columns:
                result[col] = immutable[col]
            for col in derived.columns:
                result[col] = derived[col]
        else:
            if isinstance(columns, str):
                columns = [columns]
            for col in columns:
                if col in immutable.columns:
                    result[col] = immutable[col]
                elif col in derived.columns:
                    result[col] = derived[col]

        return result.sort_values(self.index_columns).reset_index(drop=True)


class SEPTable(CachedTable):
    table_name = 'SHARADAR/SEP'
    index_columns = ['ticker', 'date']  # Put the `ticker` column first, for efficient ticker + date range queries.
    query_columns = ['open', 'low', 'high', 'close', 'volume', 'closeadj', 'closeunadj']
    immutable_columns = ['split_factor', 'split_dividend_factor',
                         'openunadj', 'lowunadj', 'highunadj', 'closeunadj', 'volumeunadj']

    @staticmethod
    def immutable_data(queried: pd.DataFrame) -> pd.DataFrame:
        immutable = pd.DataFrame()
        immutable['split_factor'] = queried['closeunadj'] / queried['close']
        immutable['split_dividend_factor'] = queried['closeunadj'] / queried['closeadj']
        for column in ['open', 'low', 'high', 'close']:
            immutable[f'{column}unadj'] = queried[column] * immutable['split_factor']
        immutable['volumeunadj'] = queried['volume'] / immutable['split_factor']
        return immutable

    @staticmethod
    def derived_data(immutable: pd.DataFrame) -> pd.DataFrame:
        derived = pd.DataFrame()
        for column in ['open', 'low', 'high', 'close']:
            derived[column] = immutable[f'{column}unadj'] / immutable['split_factor']
            derived[f'{column}adj'] = immutable[f'{column}unadj'] / immutable['split_dividend_factor']
        derived['volume'] = immutable['volumeunadj'] * immutable['split_factor']
        derived['volumeadj'] = immutable['volumeunadj'] * immutable['split_dividend_factor']
        return derived
