"""
Price data multiplexer that routes queries to SEP or SFP based on ticker category.
"""
import pandas as pd
from .tables import SEPTable, SFPTable, TickersTable


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
        self.sep = SEPTable()
        self.sfp = SFPTable()
        self.tickers = TickersTable()
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
            df = self.tickers.query(ticker=unknown)
            if not df.empty:
                for ticker, row in df.iterrows():
                    if isinstance(ticker, tuple):
                        ticker = ticker[0]  # Handle multi-index
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

        Parameters match SEPTable/SFPTable.query():
            columns: List of columns to return (or None for all)
            ticker: Single ticker or list of tickers
            date_gte: Start date (inclusive)
            date_lte: End date (inclusive)

        Returns DataFrame indexed by (ticker, date) with requested columns.
        """
        ticker_filter = filters.get('ticker')
        if ticker_filter is None:
            return pd.DataFrame()

        if isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        else:
            tickers = list(ticker_filter)

        equity_symbols, fund_symbols = self.split_by_category(tickers)

        results = []

        if equity_symbols:
            equity_filters = {**filters, 'ticker': equity_symbols if len(equity_symbols) > 1 else equity_symbols[0]}
            df = self.sep.query(columns=columns, **equity_filters)
            if not df.empty:
                results.append(df)

        if fund_symbols:
            fund_filters = {**filters, 'ticker': fund_symbols if len(fund_symbols) > 1 else fund_symbols[0]}
            df = self.sfp.query(columns=columns, **fund_filters)
            if not df.empty:
                results.append(df)

        if not results:
            return pd.DataFrame()

        return pd.concat(results)
