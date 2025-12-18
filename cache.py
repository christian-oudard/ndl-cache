import pandas as pd
import nasdaqdatalink as ndl
from pathlib import Path
from datetime import datetime, timedelta
import duckdb


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

    def fetch_from_ndl(self, **filters) -> pd.DataFrame:
        """Fetch data from NDL API."""
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

        return ndl.get_table(
            self.table_name,
            qopts={'columns': self.index_columns + self.query_columns},
            paginate=True,
            **ndl_filters
        )

    def sync(self, **filters):
        """Sync data from NDL to local cache."""
        queried = self.fetch_from_ndl(**filters)
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
            # Multi-ticker query: only fetch missing tickers
            cached_tickers = set(immutable['ticker'].unique()) if len(immutable) > 0 else set()
            missing_tickers = [t for t in ticker_filter if t not in cached_tickers]
            for ticker in missing_tickers:
                ticker_filters = {**filters, 'ticker': ticker}
                self.sync(**ticker_filters)
            if missing_tickers:
                immutable = self.get_cached(**filters)
        elif len(immutable) == 0:
            # Cache empty: fetch everything
            self.sync(**filters)
            immutable = self.get_cached(**filters)
        else:
            # Check for date range gaps
            date_gte = filters.get('date_gte')
            date_lte = filters.get('date_lte')
            if date_gte and date_lte and 'date' in immutable.columns:
                cached_dates = set(str(d)[:10] for d in immutable['date'])
                cached_min = min(cached_dates)
                cached_max = max(cached_dates)
                needs_refetch = False
                # Fetch dates before cached range
                if date_gte < cached_min:
                    day_before_min = (datetime.strptime(cached_min, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                    before_filters = {**filters, 'date_lte': day_before_min}
                    self.sync(**before_filters)
                    needs_refetch = True
                # Fetch dates after cached range
                if date_lte > cached_max:
                    day_after_max = (datetime.strptime(cached_max, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                    after_filters = {**filters, 'date_gte': day_after_max}
                    self.sync(**after_filters)
                    needs_refetch = True
                # Fetch gaps within cached range
                if cached_min <= date_gte and date_lte <= cached_max:
                    # Find contiguous gaps and fetch them
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
                                # Fetch this gap
                                gap_filters = {**filters, 'date_gte': gap_start, 'date_lte': gap_end}
                                self.sync(**gap_filters)
                                needs_refetch = True
                                gap_start = None
                        current += timedelta(days=1)
                    # Handle trailing gap
                    if gap_start is not None:
                        gap_filters = {**filters, 'date_gte': gap_start, 'date_lte': gap_end}
                        self.sync(**gap_filters)
                        needs_refetch = True
                if needs_refetch:
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
