"""
Table definitions for Nasdaq Data Link Sharadar tables.

Each table is defined as a TableDef with its schema. Use query() or async_query()
to fetch data from these tables.
"""
from dataclasses import dataclass, field

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TableDef:
    """Definition of a Sharadar table schema."""
    name: str
    index_columns: tuple[str, ...]
    query_columns: tuple[str, ...]
    date_column: str | None = 'date'
    column_types: dict[str, str] = field(default_factory=dict)
    rows_per_year: int = TRADING_DAYS_PER_YEAR
    sync_delay_days: int = 0

    @property
    def all_columns(self) -> list[str]:
        """All columns (index + query)."""
        return list(self.index_columns) + list(self.query_columns)

    def safe_table_name(self) -> str:
        """Table name safe for SQL (e.g., 'sharadar_sep')."""
        return self.name.lower().replace('/', '_')

    def sync_bounds_table_name(self) -> str:
        """Name of the sync bounds tracking table."""
        return f"{self.safe_table_name()}_sync_bounds"


# Daily equity prices (stocks)
SEP = TableDef(
    name='SHARADAR/SEP',
    index_columns=('ticker', 'date'),
    query_columns=('open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj', 'lastupdated'),
    sync_delay_days=3,
    column_types={
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    },
)

# Daily fund prices (ETFs, mutual funds)
SFP = TableDef(
    name='SHARADAR/SFP',
    index_columns=('ticker', 'date'),
    query_columns=('open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj', 'lastupdated'),
    sync_delay_days=3,
    column_types={
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    },
)

# Core US Fundamentals
SF1 = TableDef(
    name='SHARADAR/SF1',
    index_columns=('ticker', 'dimension', 'datekey'),
    date_column='calendardate',
    query_columns=(
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
    ),
    rows_per_year=18,
    sync_delay_days=3,
    column_types={
        'ticker': 'VARCHAR',
        'dimension': 'VARCHAR',
        'datekey': 'DATE',
        'calendardate': 'DATE',
        'reportperiod': 'DATE',
        'lastupdated': 'DATE',
        'fiscalperiod': 'VARCHAR',
    },
)

# Daily valuation metrics
DAILY = TableDef(
    name='SHARADAR/DAILY',
    index_columns=('ticker', 'date'),
    query_columns=('marketcap', 'ev', 'pb', 'pe', 'ps', 'lastupdated'),
    sync_delay_days=3,
    column_types={
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    },
)

# Corporate actions (dividends, splits, spinoffs)
ACTIONS = TableDef(
    name='SHARADAR/ACTIONS',
    index_columns=('ticker', 'date', 'action'),
    query_columns=('name', 'value', 'contraticker', 'contraname'),
    sync_delay_days=3,
    column_types={
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'action': 'VARCHAR',
        'name': 'VARCHAR',
        'contraticker': 'VARCHAR',
        'contraname': 'VARCHAR',
    },
)

# Ticker metadata
TICKERS = TableDef(
    name='SHARADAR/TICKERS',
    index_columns=('ticker',),
    date_column=None,  # No date-based sync
    query_columns=(
        'name', 'exchange', 'category', 'cusips', 'siccode', 'sicsector',
        'sicindustry', 'famasector', 'famaindustry', 'sector', 'industry',
        'scalemarketcap', 'scalerevenue', 'currency', 'location', 'lastupdated',
        'firstadded', 'firstpricedate', 'lastpricedate', 'firstquarter',
        'lastquarter', 'secfilings', 'companysite', 'isdelisted', 'permaticker',
    ),
    column_types={
        'ticker': 'VARCHAR',
        'name': 'VARCHAR',
        'exchange': 'VARCHAR',
        'category': 'VARCHAR',
        'cusips': 'VARCHAR',
        'siccode': 'VARCHAR',
        'sicsector': 'VARCHAR',
        'sicindustry': 'VARCHAR',
        'famasector': 'VARCHAR',
        'famaindustry': 'VARCHAR',
        'sector': 'VARCHAR',
        'industry': 'VARCHAR',
        'scalemarketcap': 'VARCHAR',
        'scalerevenue': 'VARCHAR',
        'currency': 'VARCHAR',
        'location': 'VARCHAR',
        'lastupdated': 'DATE',
        'firstadded': 'DATE',
        'firstpricedate': 'DATE',
        'lastpricedate': 'DATE',
        'firstquarter': 'DATE',
        'lastquarter': 'DATE',
        'secfilings': 'VARCHAR',
        'companysite': 'VARCHAR',
        'isdelisted': 'VARCHAR',
        'permaticker': 'INTEGER',
    },
)
