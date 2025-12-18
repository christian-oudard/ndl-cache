"""
Sharadar table implementations.

Each table defines its schema and transformations between
queried data (from NDL API) and immutable storage format.
"""
import pandas as pd
from cache import CachedTable


class _PriceTableMixin:
    """
    Mixin for price tables (SEP, SFP) with OHLCV data.

    Stores immutable unadjusted prices and derives adjusted prices on read.
    """
    index_columns = ['ticker', 'date']
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
    immutable_columns = _SF1_DATA_COLUMNS  # No transformation needed

    # ~18 rows per ticker per year (4 quarterly + 1 annual across ~3-4 dimensions with data)
    # Based on empirical testing: 270 rows / 15 years = 18
    rows_per_year = 18

    # Column types for non-DOUBLE columns
    column_types = {
        'fiscalperiod': 'VARCHAR',  # e.g., "2024-Q2", "2024-FY"
    }

    @staticmethod
    def immutable_data(queried: pd.DataFrame) -> pd.DataFrame:
        """For fundamentals, store data as-is (no transformation needed)."""
        return queried[_SF1_DATA_COLUMNS].copy()

    @staticmethod
    def derived_data(immutable: pd.DataFrame) -> pd.DataFrame:
        """No derived columns for fundamentals."""
        return pd.DataFrame()
