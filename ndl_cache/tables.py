"""
Sharadar table implementations.

Each table defines its schema. Data is stored as-is from the API.
Staleness is tracked via lastupdated column and checked once per day.
"""
from .cache import CachedTable


class SEPTable(CachedTable):
    """
    SHARADAR/SEP - Daily equity prices (stocks).

    Stores all price data as-is from Sharadar, including pre-computed
    adjusted prices (closeadj, etc.).
    """
    table_name = 'SHARADAR/SEP'
    index_columns = ['ticker', 'date']
    query_columns = ['open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj', 'lastupdated']
    column_types = {
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    }


class SFPTable(CachedTable):
    """
    SHARADAR/SFP - Daily fund prices (ETFs, mutual funds).

    Stores all price data as-is from Sharadar.
    """
    table_name = 'SHARADAR/SFP'
    index_columns = ['ticker', 'date']
    query_columns = ['open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj', 'lastupdated']
    column_types = {
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    }


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
    query_columns = [
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


class DailyTable(CachedTable):
    """
    SHARADAR/DAILY - Daily valuation metrics.

    Includes market cap, price ratios, and other daily calculated values.
    """
    table_name = 'SHARADAR/DAILY'
    index_columns = ['ticker', 'date']
    query_columns = ['marketcap', 'ev', 'pb', 'pe', 'ps', 'lastupdated']
    column_types = {
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'lastupdated': 'DATE',
    }


class ActionsTable(CachedTable):
    """
    SHARADAR/ACTIONS - Corporate actions (dividends, splits, spinoffs).

    Action types include:
    - dividend: Cash dividend per share
    - split: Stock split ratio (e.g., 2.0 for 2-for-1 split)
    - spinoff: Spinoff events
    """
    table_name = 'SHARADAR/ACTIONS'
    index_columns = ['ticker', 'date', 'action']
    query_columns = ['name', 'value', 'contraticker', 'contraname']
    column_types = {
        'ticker': 'VARCHAR',
        'date': 'DATE',
        'action': 'VARCHAR',
        'name': 'VARCHAR',
        'contraticker': 'VARCHAR',
        'contraname': 'VARCHAR',
    }


class TickersTable(CachedTable):
    """
    SHARADAR/TICKERS - Ticker metadata.

    Contains category, exchange, and other static info for each ticker.
    This table is special: no date-based filtering, synced per ticker.
    """
    table_name = 'SHARADAR/TICKERS'
    index_columns = ['ticker']
    date_column = None  # No date-based sync - sync entire ticker or nothing
    query_columns = [
        'name', 'exchange', 'category', 'cusips', 'siccode', 'sicsector',
        'sicindustry', 'famasector', 'famaindustry', 'sector', 'industry',
        'scalemarketcap', 'scalerevenue', 'currency', 'location', 'lastupdated',
        'firstadded', 'firstpricedate', 'lastpricedate', 'firstquarter',
        'lastquarter', 'secfilings', 'companysite', 'isdelisted', 'permaticker',
    ]
    column_types = {
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
    }
