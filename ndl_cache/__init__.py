"""
ndl-cache: Cached access to Nasdaq Data Link Sharadar tables.

Usage (sync):
    from ndl_cache import SEP, query

    df = query(SEP, ticker='AAPL', date_gte='2024-01-01', date_lte='2024-12-31')

Usage (async):
    from ndl_cache import SEP, async_query

    df = await async_query(SEP, ticker='AAPL', date_gte='2024-01-01', date_lte='2024-12-31')

Available tables:
    SEP - Daily equity prices (stocks)
    SFP - Daily fund prices (ETFs, mutual funds)
    SF1 - Core US Fundamentals
    DAILY - Daily valuation metrics
    ACTIONS - Corporate actions (dividends, splits, spinoffs)
    TICKERS - Ticker metadata

Optional: Set custom database path via environment variable:
    export NDL_CACHE_DB_PATH=/path/to/cache.duckdb
"""

# Table definitions
from .tables import (
    TableDef,
    SEP,
    SFP,
    SF1,
    DAILY,
    ACTIONS,
    TICKERS,
)

# Query functions
from .async_cache import (
    query,
    async_query,
    get_db_path,
    validate_sync_bounds,
    async_validate_sync_bounds,
)

# Price data multiplexer
from .prices import PriceData

# Async client (for advanced use)
from .async_client import (
    AsyncNDLClient,
    gather_tables,
    NDLError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
)

__all__ = [
    # Table definitions
    'TableDef',
    'SEP',
    'SFP',
    'SF1',
    'DAILY',
    'ACTIONS',
    'TICKERS',
    # Query functions
    'query',
    'async_query',
    'get_db_path',
    'validate_sync_bounds',
    'async_validate_sync_bounds',
    # Price data
    'PriceData',
    # Async client
    'AsyncNDLClient',
    'gather_tables',
    'NDLError',
    'AuthenticationError',
    'RateLimitError',
    'NotFoundError',
]
