"""
Sharadar table implementations.

Each table defines its schema and transformations between
queried data (from NDL API) and immutable storage format.
"""
import pandas as pd
import nasdaqdatalink as ndl
import duckdb
from cache import CachedTable, DB_PATH


class _PriceTableMixin:
    """
    Mixin for price tables (SEP, SFP) with OHLCV data.

    Stores immutable unadjusted prices and derives adjusted prices on read
    using corporate actions (splits, dividends) from ActionsTable.

    DataFrames are indexed by (ticker, date).
    """
    index_columns = ['ticker', 'date']
    query_columns = ['open', 'low', 'high', 'close', 'volume', 'closeadj', 'closeunadj']
    column_types = {'ticker': 'VARCHAR', 'date': 'DATE'}

    @staticmethod
    def immutable_data(queried: pd.DataFrame) -> pd.DataFrame:
        """Convert queried data to immutable storage format (unadjusted prices)."""
        immutable = pd.DataFrame()
        split_factor = queried['closeunadj'] / queried['close']
        for column in ['open', 'low', 'high', 'close']:
            immutable[f'{column}unadj'] = queried[column] * split_factor
        immutable['volumeunadj'] = queried['volume'] / split_factor
        return immutable

    def derived_data(self, immutable: pd.DataFrame) -> pd.DataFrame:
        """
        Compute split and dividend adjusted prices from ACTIONS data.

        Args:
            immutable: DataFrame indexed by (ticker, date) with unadjusted prices

        Returns:
            DataFrame with same index containing derived columns
        """
        if len(immutable) == 0:
            return pd.DataFrame()

        tickers = immutable.index.get_level_values('ticker').unique().tolist()
        split_factor = self._compute_split_factor(immutable.index, tickers)
        dividend_adj = self._compute_dividend_adjustment(immutable, tickers, split_factor)

        derived = pd.DataFrame(index=immutable.index)
        for column in ['open', 'low', 'high', 'close']:
            derived[column] = immutable[f'{column}unadj'] / split_factor
            derived[f'{column}adj'] = derived[column] * dividend_adj

        derived['volume'] = immutable['volumeunadj'] * split_factor
        derived['volumeadj'] = derived['volume'] * dividend_adj

        return derived

    def _compute_split_factor(self, index: pd.MultiIndex, tickers: list[str]) -> pd.Series:
        """
        Compute split factor for each (ticker, date) in index.
        factor = PRODUCT(split.value) for all splits after that date.

        Args:
            index: MultiIndex of (ticker, date)
            tickers: List of tickers to fetch splits for

        Returns:
            Series indexed by (ticker, date) with split factors
        """
        splits = ActionsTable().get_splits(tickers)

        if len(splits) == 0:
            return pd.Series(1.0, index=index)

        # Build cumulative split factor per ticker (from most recent split backwards)
        splits = splits.copy()
        splits['date'] = pd.to_datetime(splits['date'])
        splits = splits.sort_values(['ticker', 'date'], ascending=[True, False])
        splits['cumulative'] = splits.groupby('ticker')['value'].cumprod()

        # Build lookup: ticker -> list of (split_date, cumulative_factor) sorted ascending by date
        split_lookup = {}
        for ticker in splits['ticker'].unique():
            ticker_splits = splits[splits['ticker'] == ticker].sort_values('date')
            split_lookup[ticker] = list(zip(ticker_splits['date'], ticker_splits['cumulative']))

        # For each (ticker, date), find the first split after that date
        factors = []
        for ticker, date in index:
            ticker_splits = split_lookup.get(ticker, [])
            price_date = pd.to_datetime(date)
            factor = 1.0
            for split_date, cumulative in ticker_splits:
                if split_date > price_date:
                    factor = cumulative
                    break
            factors.append(factor)

        return pd.Series(factors, index=index)

    def _compute_dividend_adjustment(self, immutable: pd.DataFrame, tickers: list[str], split_factor: pd.Series) -> pd.Series:
        """
        Compute dividend adjustment for each (ticker, date).
        adjustment = PRODUCT(1 - yield) for all dividends after that date
        where yield = dividend_value / close_on_day_before_exdiv

        Uses ALL cached price data (not just the query range) to find close prices
        for dividend yield calculation.
        """
        dividends = ActionsTable().get_dividends(tickers)

        if len(dividends) == 0:
            return pd.Series(1.0, index=immutable.index)

        # Get ALL cached prices for these tickers to compute dividend yields
        all_cached = self._get_all_cached_prices(tickers)

        if len(all_cached) == 0:
            return pd.Series(1.0, index=immutable.index)

        # Compute split factors for all cached prices and get split-adjusted close
        all_split_factors = self._compute_split_factor(all_cached.index, tickers)
        all_cached = all_cached.copy()
        all_cached['close'] = all_cached['closeunadj'] / all_split_factors

        # For each dividend, find close on day before ex-div and compute (1 - yield)
        dividends = dividends.copy()
        dividends['date'] = pd.to_datetime(dividends['date'])

        div_adjustments = {}  # (ticker, div_date) -> adjustment factor
        for _, div in dividends.iterrows():
            ticker = div['ticker']
            div_date = div['date']
            # Get prices for this ticker before div date
            try:
                ticker_prices = all_cached.loc[ticker]
                prev_prices = ticker_prices[ticker_prices.index < div_date]
                if len(prev_prices) == 0:
                    div_adjustments[(ticker, div_date)] = 1.0
                    continue
                close_before = prev_prices['close'].iloc[-1]
                if close_before <= 0:
                    div_adjustments[(ticker, div_date)] = 1.0
                else:
                    div_adjustments[(ticker, div_date)] = 1 - div['value'] / close_before
            except KeyError:
                div_adjustments[(ticker, div_date)] = 1.0

        # For each price row, multiply adjustments for all dividends after that date
        factors = []
        for ticker, date in immutable.index:
            price_date = pd.to_datetime(date)
            factor = 1.0
            for (div_ticker, div_date), adj in div_adjustments.items():
                if div_ticker == ticker and div_date > price_date:
                    factor *= adj
            factors.append(factor)

        return pd.Series(factors, index=immutable.index)

    def _get_all_cached_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Get all cached prices for the given tickers, indexed by (ticker, date)."""
        if not tickers:
            return pd.DataFrame()

        placeholders = ', '.join(['?'] * len(tickers))
        df = self.conn.execute(f"""
            SELECT ticker, date, closeunadj
            FROM {self._safe_table_name()}
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, date
        """, tickers).df()

        if len(df) > 0:
            df = df.set_index(['ticker', 'date'])

        return df


class SEPTable(_PriceTableMixin, CachedTable):
    """SHARADAR/SEP - Daily equity prices (stocks)."""
    table_name = 'SHARADAR/SEP'


class SFPTable(_PriceTableMixin, CachedTable):
    """SHARADAR/SFP - Daily fund prices (ETFs, mutual funds)."""
    table_name = 'SHARADAR/SFP'


# All SF1 data columns (108 metrics from balance sheet, income statement, cash flow, etc.)
_SF1_DATA_COLUMNS = [
    'accoci', 'assets', 'assetsavg', 'assetsc', 'assetsnc', 'assetturnover',
    'bvps', 'calendardate', 'capex', 'cashneq', 'cashnequsd', 'consolinc',
    'cor', 'currentratio', 'de', 'debt', 'debtc', 'debtnc', 'debtusd',
    'deferredrev', 'depamor', 'deposits', 'divyield', 'dps', 'ebit', 'ebitda',
    'ebitdamargin', 'ebitdausd', 'ebitusd', 'ebt', 'eps', 'epsdil', 'epsusd',
    'equity', 'equityavg', 'equityusd', 'ev', 'evebit', 'evebitda', 'fcf',
    'fcfps', 'fiscalperiod', 'fxusd', 'gp', 'grossmargin', 'intangibles',
    'intexp', 'invcap', 'invcapavg', 'inventory', 'investments', 'investmentsc',
    'investmentsnc', 'lastupdated', 'liabilities', 'liabilitiesc', 'liabilitiesnc',
    'marketcap', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv', 'ncff',
    'ncfi', 'ncfinv', 'ncfo', 'ncfx', 'netinc', 'netinccmn', 'netinccmnusd',
    'netincdis', 'netincnci', 'netmargin', 'opex', 'opinc', 'payables',
    'payoutratio', 'pb', 'pe', 'pe1', 'ppnenet', 'prefdivis', 'price', 'ps',
    'ps1', 'receivables', 'reportperiod', 'retearn', 'revenue', 'revenueusd',
    'rnd', 'roa', 'roe', 'roic', 'ros', 'sbcomp', 'sgna', 'sharefactor',
    'sharesbas', 'shareswa', 'shareswadil', 'sps', 'tangibles', 'taxassets',
    'taxexp', 'taxliabilities', 'tbvps', 'workingcapital',
]


class SF1Table(CachedTable):
    """
    SHARADAR/SF1 - Core US Fundamentals.

    Contains income statement, balance sheet, cash flow, and derived metrics.
    Data is provided quarterly, annually, and trailing-twelve-months.

    Dimension codes:
    - ARQ: As-reported quarterly (excludes restatements)
    - MRQ: Most-recent quarterly (includes restatements)
    - ARY: As-reported annual
    - MRY: Most-recent annual
    - ART: As-reported trailing-twelve-months
    - MRT: Most-recent trailing-twelve-months
    """
    table_name = 'SHARADAR/SF1'
    index_columns = ['ticker', 'dimension', 'datekey']
    date_column = 'calendardate'  # Use calendardate for sync bounds (normalized date)
    query_columns = _SF1_DATA_COLUMNS
    rows_per_year = 18  # ~18 rows per ticker per year (4 quarterly + 1 annual across ~3-4 dimensions)
    column_types = {
        'ticker': 'VARCHAR',
        'dimension': 'VARCHAR',
        'datekey': 'DATE',
        'calendardate': 'DATE',
        'reportperiod': 'DATE',
        'lastupdated': 'DATE',
        'fiscalperiod': 'VARCHAR',
    }

    @staticmethod
    def immutable_data(queried: pd.DataFrame) -> pd.DataFrame:
        """For fundamentals, store data as-is (no transformation needed)."""
        return queried[_SF1_DATA_COLUMNS].copy()

    def derived_data(self, immutable: pd.DataFrame) -> pd.DataFrame:
        """No derived columns for fundamentals."""
        return pd.DataFrame()


class ActionsTable:
    """
    SHARADAR/ACTIONS - Corporate actions (splits, dividends, M&A, etc.).

    Unlike CachedTable, this syncs full ticker history without date ranges.
    Sync tracking is per-ticker (synced or not), not by date bounds.

    Key action types:
    - split: Stock split, value = new_shares/old_shares
    - dividend: Cash dividend, value = USD/share (split-adjusted)
    """
    table_name = 'SHARADAR/ACTIONS'
    index_columns = {'ticker': 'VARCHAR', 'date': 'DATE', 'action': 'VARCHAR'}
    data_columns = {'value': 'DOUBLE', 'name': 'VARCHAR', 'contraticker': 'VARCHAR', 'contraname': 'VARCHAR'}

    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
        self._ensure_table()

    def _safe_table_name(self) -> str:
        return self.table_name.replace('/', '_').lower()

    def _ensure_table(self):
        """Create data table and synced tickers tracking table."""
        all_columns = {**self.index_columns, **self.data_columns}
        col_defs = [f'{col} {typ}' for col, typ in all_columns.items()]
        pk = ', '.join(self.index_columns.keys())

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._safe_table_name()} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk})
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._safe_table_name()}_synced (
                ticker VARCHAR PRIMARY KEY,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def get_synced_tickers(self) -> set[str]:
        """Return set of tickers that have been synced."""
        result = self.conn.execute(
            f"SELECT ticker FROM {self._safe_table_name()}_synced"
        ).fetchall()
        return {row[0] for row in result}

    def ensure_synced(self, tickers: list[str]) -> int:
        """
        Ensure given tickers are synced. Returns count of newly synced.
        Syncs full history for each ticker (no date filtering).
        """
        if not tickers:
            return 0

        synced = self.get_synced_tickers()
        to_sync = [t for t in tickers if t not in synced]

        if not to_sync:
            return 0

        table = self._safe_table_name()
        cols = list(self.index_columns.keys()) + list(self.data_columns.keys())

        for ticker in to_sync:
            df = ndl.get_table(self.table_name, ticker=ticker, paginate=True)

            if len(df) > 0:
                df = df[cols]
                self.conn.execute(f"""
                    INSERT OR REPLACE INTO {table}
                    SELECT * FROM df
                """)

            self.conn.execute(f"""
                INSERT OR REPLACE INTO {table}_synced (ticker)
                VALUES (?)
            """, [ticker])

        return len(to_sync)

    def get_splits(self, tickers: list[str]) -> pd.DataFrame:
        """Get split actions for given tickers."""
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'date', 'value'])

        placeholders = ', '.join(['?'] * len(tickers))
        return self.conn.execute(f"""
            SELECT ticker, date, value
            FROM {self._safe_table_name()}
            WHERE ticker IN ({placeholders}) AND action = 'split'
            ORDER BY ticker, date
        """, tickers).df()

    def get_dividends(self, tickers: list[str]) -> pd.DataFrame:
        """Get dividend actions for given tickers."""
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'date', 'value'])

        placeholders = ', '.join(['?'] * len(tickers))
        return self.conn.execute(f"""
            SELECT ticker, date, value
            FROM {self._safe_table_name()}
            WHERE ticker IN ({placeholders}) AND action = 'dividend'
            ORDER BY ticker, date
        """, tickers).df()

    def query(self, **filters) -> pd.DataFrame:
        """
        Query actions data with filters.

        Supports filters: ticker (str or list), action (str or list)
        """
        ticker_filter = filters.get('ticker')
        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        elif isinstance(ticker_filter, list):
            tickers = ticker_filter
        else:
            tickers = []

        # Ensure tickers are synced before querying
        if tickers:
            self.ensure_synced(tickers)

        where_clauses = []
        params = []

        if tickers:
            placeholders = ', '.join(['?'] * len(tickers))
            where_clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)

        action_filter = filters.get('action')
        if action_filter:
            if isinstance(action_filter, str):
                where_clauses.append("action = ?")
                params.append(action_filter)
            elif isinstance(action_filter, list):
                placeholders = ', '.join(['?'] * len(action_filter))
                where_clauses.append(f"action IN ({placeholders})")
                params.extend(action_filter)

        where = ' AND '.join(where_clauses) if where_clauses else '1=1'

        return self.conn.execute(f"""
            SELECT * FROM {self._safe_table_name()}
            WHERE {where}
            ORDER BY ticker, date
        """, params).df()
