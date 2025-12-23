"""
ndl-cache: Cached access to Nasdaq Data Link Sharadar tables.

Usage:
    from ndl_cache import SEPTable, SFPTable, SF1Table, DailyTable, ActionsTable, TickersTable
    from ndl_cache import set_db_path, get_db_path

    # Optional: Set custom database path before creating tables
    set_db_path('/path/to/cache.duckdb')

    # Query equity prices
    sep = SEPTable()
    df = sep.query(columns=['close', 'volume'], ticker='AAPL', date_gte='2020-01-01', date_lte='2020-12-31')
"""

from . import cache
from .cache import (
    CachedTable,
    set_db_path,
    get_db_path,
)
from .tables import (
    SEPTable,
    SFPTable,
    SF1Table,
    DailyTable,
    ActionsTable,
    TickersTable,
)
from .prices import PriceData

__all__ = [
    'cache',
    'CachedTable',
    'set_db_path',
    'get_db_path',
    'SEPTable',
    'SFPTable',
    'SF1Table',
    'DailyTable',
    'ActionsTable',
    'TickersTable',
    'PriceData',
]
