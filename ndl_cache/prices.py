"""
Price data multiplexer that routes queries to SEP or SFP based on ticker category.
"""
import asyncio
import pandas as pd
from .tables import SEP, SFP, TICKERS
from .async_cache import query, async_query


# Categories that use SFP (fund prices) instead of SEP (equity prices)
FUND_CATEGORIES = ['ETF', 'CEF', 'ETN']


class PriceData:
    """
    Routes price queries to SEP or SFP based on symbol category.

    Usage:
        prices = PriceData()
        df = prices.query(
            ticker=['AAPL', 'SPY'],  # Mix of equity and fund
            date_gte='2020-01-01',
            date_lte='2020-12-31',
        )
    """

    def __init__(self):
        self._category_cache: dict[str, str] = {}

    def get_categories(self, symbols: list[str]) -> dict[str, str]:
        """
        Get category for each symbol. Queries API for unknown symbols.
        Returns dict mapping symbol -> category.
        """
        # Check cache first
        unknown = [s for s in symbols if s not in self._category_cache]

        if unknown:
            # Query tickers table for unknown symbols
            df = query(TICKERS, ticker=unknown)
            if not df.empty:
                for ticker, row in df.iterrows():
                    if isinstance(ticker, tuple):
                        ticker = ticker[0]  # Handle multi-index
                    self._category_cache[ticker] = row.get('category', 'Unknown')

        return {s: self._category_cache.get(s, 'Unknown') for s in symbols}

    async def async_get_categories(self, symbols: list[str]) -> dict[str, str]:
        """
        Async version of get_categories.
        """
        unknown = [s for s in symbols if s not in self._category_cache]

        if unknown:
            df = await async_query(TICKERS, ticker=unknown)
            if not df.empty:
                for ticker, row in df.iterrows():
                    if isinstance(ticker, tuple):
                        ticker = ticker[0]
                    self._category_cache[ticker] = row.get('category', 'Unknown')

        return {s: self._category_cache.get(s, 'Unknown') for s in symbols}

    def split_by_category(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        Split symbols into equity and fund lists based on category.
        Returns (equity_symbols, fund_symbols).
        """
        categories = self.get_categories(symbols)
        equity_symbols = []
        fund_symbols = []

        for symbol in symbols:
            if categories.get(symbol) in FUND_CATEGORIES:
                fund_symbols.append(symbol)
            else:
                equity_symbols.append(symbol)

        return equity_symbols, fund_symbols

    def query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Query price data, routing to SEP or SFP based on symbol category.

        Parameters:
            columns: List of columns to return (or None for all)
            ticker: Single ticker or list of tickers
            date_gte: Start date (inclusive)
            date_lte: End date (inclusive)

        Returns DataFrame indexed by (ticker, date) with requested columns.
        """
        from concurrent.futures import ThreadPoolExecutor
        from .async_cache import query as cache_query

        ticker_filter = filters.get('ticker')
        if ticker_filter is None:
            return pd.DataFrame()

        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        else:
            tickers = list(ticker_filter)

        equity_symbols, fund_symbols = self.split_by_category(tickers)

        # Query SEP and SFP in parallel if we have both equity and fund symbols
        if equity_symbols and fund_symbols:
            equity_filters = {**filters, 'ticker': equity_symbols if len(equity_symbols) > 1 else equity_symbols[0]}
            fund_filters = {**filters, 'ticker': fund_symbols if len(fund_symbols) > 1 else fund_symbols[0]}

            with ThreadPoolExecutor(max_workers=2) as executor:
                sep_future = executor.submit(cache_query, SEP, columns=columns, **equity_filters)
                sfp_future = executor.submit(cache_query, SFP, columns=columns, **fund_filters)

                sep_result = sep_future.result()
                sfp_result = sfp_future.result()

            results = []
            if not sep_result.empty:
                results.append(sep_result)
            if not sfp_result.empty:
                results.append(sfp_result)

            if not results:
                return pd.DataFrame()
            return pd.concat(results)

        # Only one type of symbol - no need for parallelism
        if equity_symbols:
            equity_filters = {**filters, 'ticker': equity_symbols if len(equity_symbols) > 1 else equity_symbols[0]}
            return cache_query(SEP, columns=columns, **equity_filters)

        if fund_symbols:
            fund_filters = {**filters, 'ticker': fund_symbols if len(fund_symbols) > 1 else fund_symbols[0]}
            return cache_query(SFP, columns=columns, **fund_filters)

        return pd.DataFrame()

    async def async_query(self, columns: list[str] | str | None = None, **filters) -> pd.DataFrame:
        """
        Async query for price data, routing to SEP or SFP based on symbol category.
        Uses asyncio.gather for true parallelism without thread conflicts.
        """
        ticker_filter = filters.get('ticker')
        if ticker_filter is None:
            return pd.DataFrame()

        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        else:
            tickers = list(ticker_filter)

        categories = await self.async_get_categories(tickers)
        equity_symbols = [s for s in tickers if categories.get(s) not in FUND_CATEGORIES]
        fund_symbols = [s for s in tickers if categories.get(s) in FUND_CATEGORIES]

        # Query SEP and SFP in parallel if we have both
        if equity_symbols and fund_symbols:
            equity_filters = {**filters, 'ticker': equity_symbols if len(equity_symbols) > 1 else equity_symbols[0]}
            fund_filters = {**filters, 'ticker': fund_symbols if len(fund_symbols) > 1 else fund_symbols[0]}

            sep_result, sfp_result = await asyncio.gather(
                async_query(SEP, columns=columns, **equity_filters),
                async_query(SFP, columns=columns, **fund_filters),
            )

            results = []
            if not sep_result.empty:
                results.append(sep_result)
            if not sfp_result.empty:
                results.append(sfp_result)

            if not results:
                return pd.DataFrame()
            return pd.concat(results)

        # Only one type of symbol
        if equity_symbols:
            equity_filters = {**filters, 'ticker': equity_symbols if len(equity_symbols) > 1 else equity_symbols[0]}
            return await async_query(SEP, columns=columns, **equity_filters)

        if fund_symbols:
            fund_filters = {**filters, 'ticker': fund_symbols if len(fund_symbols) > 1 else fund_symbols[0]}
            return await async_query(SFP, columns=columns, **fund_filters)

        return pd.DataFrame()
