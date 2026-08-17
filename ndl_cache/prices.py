"""
Price data multiplexer that routes queries to SEP or SFP.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .tables import SEP, SFP, TICKERS
from .async_cache import query as cache_query, async_query as cache_async_query


# SHARADAR/TICKERS lists a symbol once per source table, and the two price
# tables do not overlap: across 73,972 listings, no symbol appears under both
# SEP and SFP. So the listing itself says which table holds a symbol's prices,
# which is exactly what routing needs to know.
#
# The category does not say it. SFP holds ETD, UNIT, ETMF, IDX and MF rows
# besides the ETF, CEF and ETN ones, so choosing by a list of fund categories
# sends 623 funds to the equity table, where they have no rows at all and come
# back empty with no error.
EQUITY_TABLE = 'SEP'
FUND_TABLE = 'SFP'

UNKNOWN = 'Unknown'


class PriceData:
    """
    Routes price queries to SEP or SFP based on where the symbol is listed.

    Usage:
        prices = PriceData()
        df = prices.query(
            ticker=['AAPL', 'SPY'],  # Mix of equity and fund
            date_gte='2020-01-01',
            date_lte='2020-12-31',
        )
    """

    def __init__(self):
        self._source: dict[str, str | None] = {}
        self._category: dict[str, str] = {}

    def _unlisted(self, symbols: list[str]) -> list[str]:
        """Symbols this instance has not looked up yet."""
        return [s for s in symbols if s not in self._source]

    def _record(self, listings: pd.DataFrame, asked: list[str]):
        """
        Remember where each symbol is listed, and as what.

        A symbol appears once per source table, so the frame is indexed by
        (table, ticker) and reading it positionally takes the table name for
        the symbol. That left every category Unknown and every fund routed to
        the equity table.

        Symbols the provider does not list are recorded too, so that asking
        about one does not look unresolved and get asked again on every call.
        """
        if not listings.empty:
            for (table, ticker), row in listings.iterrows():
                category = row['category']
                if table in (EQUITY_TABLE, FUND_TABLE):
                    self._source[ticker] = table
                    self._category[ticker] = category
                elif ticker not in self._category:
                    self._category[ticker] = category
        for symbol in asked:
            self._source.setdefault(symbol, None)
            self._category.setdefault(symbol, UNKNOWN)

    def get_categories(self, symbols: list[str]) -> dict[str, str]:
        """
        Get category for each symbol, fetching listings not seen yet.
        Returns dict mapping symbol -> category.
        """
        unlisted = self._unlisted(symbols)
        if unlisted:
            self._record(cache_query(TICKERS, columns=['category'],
                                     ticker=unlisted), unlisted)
        return {s: self._category[s] for s in symbols}

    async def async_get_categories(self, symbols: list[str]) -> dict[str, str]:
        """
        Async version of get_categories.
        """
        unlisted = self._unlisted(symbols)
        if unlisted:
            self._record(await cache_async_query(TICKERS, columns=['category'],
                                                 ticker=unlisted), unlisted)
        return {s: self._category[s] for s in symbols}

    def split_by_category(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        Split symbols into equity and fund lists by where they are listed.
        Returns (equity_symbols, fund_symbols).
        """
        self.get_categories(symbols)
        return self._split(symbols)

    def _split(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        Symbols listed under SFP, and everything else.

        A symbol with no listing at all goes to the equity table: neither
        table holds it, so the choice does not change the answer, but dropping
        it here would lose a symbol the caller asked for without saying so.
        """
        equity = [s for s in symbols if self._source.get(s) != FUND_TABLE]
        fund = [s for s in symbols if self._source.get(s) == FUND_TABLE]
        return equity, fund

    @staticmethod
    def _tickers(filters: dict) -> list[str] | None:
        ticker_filter = filters.get('ticker')
        if ticker_filter is None:
            return None
        if isinstance(ticker_filter, str):
            return [ticker_filter]
        return list(ticker_filter)

    @staticmethod
    def _for(filters: dict, symbols: list[str]) -> dict:
        return {**filters,
                'ticker': symbols if len(symbols) > 1 else symbols[0]}

    @staticmethod
    def _combine(results: list[pd.DataFrame]) -> pd.DataFrame:
        results = [r for r in results if not r.empty]
        return pd.concat(results) if results else pd.DataFrame()

    def query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Query price data, routing to SEP or SFP based on where each symbol is
        listed.

        Parameters:
            columns: List of columns to return (or None for all)
            ticker: Single ticker or list of tickers
            date_gte: Start date (inclusive)
            date_lte: End date (inclusive)

        Returns DataFrame indexed by (ticker, date) with requested columns.
        """
        tickers = self._tickers(filters)
        if tickers is None:
            return pd.DataFrame()

        equity, fund = self.split_by_category(tickers)

        if equity and fund:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(cache_query, SEP, columns=columns,
                                    **self._for(filters, equity)),
                    executor.submit(cache_query, SFP, columns=columns,
                                    **self._for(filters, fund)),
                ]
                return self._combine([f.result() for f in futures])

        if equity:
            return cache_query(SEP, columns=columns, **self._for(filters, equity))
        if fund:
            return cache_query(SFP, columns=columns, **self._for(filters, fund))
        return pd.DataFrame()

    async def async_query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Async query for price data, routing to SEP or SFP based on where each
        symbol is listed. Uses asyncio.gather for true parallelism without
        thread conflicts.
        """
        tickers = self._tickers(filters)
        if tickers is None:
            return pd.DataFrame()

        await self.async_get_categories(tickers)
        equity, fund = self._split(tickers)

        if equity and fund:
            return self._combine(list(await asyncio.gather(
                cache_async_query(SEP, columns=columns,
                                  **self._for(filters, equity)),
                cache_async_query(SFP, columns=columns,
                                  **self._for(filters, fund)),
            )))

        if equity:
            return await cache_async_query(SEP, columns=columns,
                                           **self._for(filters, equity))
        if fund:
            return await cache_async_query(SFP, columns=columns,
                                           **self._for(filters, fund))
        return pd.DataFrame()
