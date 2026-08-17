"""
Tests for the price multiplexer.

Sharadar splits prices across two tables and neither one holds the other's
symbols, so routing a symbol to the wrong table returns nothing at all rather
than something wrong. These tests are about getting that choice right.
"""
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from ndl_cache import PriceData
from ndl_cache.async_client import AsyncNDLClient
from ndl_cache.testing import temp_db


# One listing per (table, symbol), as SHARADAR/TICKERS gives them. The fund
# categories here are deliberately wider than ETF/CEF/ETN: SFP also holds ETD,
# UNIT, ETMF, IDX and MF rows, 623 of them on the live table.
LISTINGS = [
    ('SEP', 'AAPL', 'Domestic Common Stock'),
    ('SF1', 'AAPL', 'Domestic Common Stock'),
    ('SF2', 'AAPL', 'Domestic Common Stock'),
    ('SEP', 'NRDS', 'Domestic Common Stock'),
    ('SFP', 'GOVT', 'ETF'),
    ('SFP', 'PDI', 'CEF'),
    ('SF2', 'PDI', 'CEF'),
    ('SFP', 'DTLA', 'ETD'),
    ('SFP', 'AFB', 'CEF Preferred'),
    ('SF3B', 'BRK', 'Institutional Investor'),
]

DATES = ['2024-01-02', '2024-01-03', '2024-01-04']


def listed_in(source: str) -> list[str]:
    return [ticker for table, ticker, _ in LISTINGS if table == source]


def tickers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{'table': table, 'ticker': ticker, 'category': category,
          'name': f'{ticker} inc', 'lastupdated': date(2024, 1, 4)}
         for table, ticker, category in LISTINGS])


def prices_frame(source: str, tickers: list[str], gte: str, lte: str,
                 columns: list[str] | None) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        if ticker not in listed_in(source):
            continue
        for i, day in enumerate(DATES):
            if not (gte <= day <= lte):
                continue
            rows.append({
                'ticker': ticker,
                'date': date.fromisoformat(day),
                'close': 100.0 + i,
                'closeadj': 100.0 + i,
                'lastupdated': date(2024, 1, 5),
            })
    frame = pd.DataFrame(rows, columns=['ticker', 'date', 'close', 'closeadj',
                                        'lastupdated'])
    return frame[[c for c in columns if c in frame.columns]] if columns else frame


@pytest.fixture(autouse=True)
def mock_ndl_client():
    """Serve the two price tables and the ticker listing from the fixtures."""
    async def get_table(self, table_name, columns=None, paginate=True, **filters):
        source = table_name.split('/')[-1]
        if source == 'TICKERS':
            return tickers_frame()
        wanted = filters.get('ticker')
        wanted = [wanted] if isinstance(wanted, str) else list(wanted or [])
        window = filters.get('date', {})
        return prices_frame(source, wanted, window.get('gte', '0000-01-01'),
                            window.get('lte', '9999-12-31'), columns)

    with patch.object(AsyncNDLClient, 'get_table', get_table):
        yield


@pytest.fixture
def prices():
    with temp_db():
        yield PriceData()


def closes(prices, symbols):
    """Which symbols came back with prices, in a form easy to assert on."""
    df = prices.query(columns=['close'], ticker=symbols,
                      date_gte=DATES[0], date_lte=DATES[-1])
    return set() if df.empty else set(df.index.get_level_values('ticker'))


class TestRoutingToTheRightPriceTable:

    def test_an_etf_gets_its_prices(self, prices):
        # The whole failure this file exists for: an ETF routed to the equity
        # table, which holds no ETF rows, comes back empty with no error.
        assert closes(prices, ['GOVT']) == {'GOVT'}

    def test_an_equity_gets_its_prices(self, prices):
        assert closes(prices, ['AAPL']) == {'AAPL'}

    def test_a_mixed_request_returns_both(self, prices):
        assert closes(prices, ['AAPL', 'GOVT', 'NRDS', 'PDI']) == {
            'AAPL', 'GOVT', 'NRDS', 'PDI'}

    def test_a_fund_outside_the_common_categories_still_routes_to_funds(self, prices):
        # ETD and CEF Preferred are as much SFP symbols as ETF is. Deciding by
        # a list of fund categories sent these to the equity table.
        assert closes(prices, ['DTLA', 'AFB']) == {'DTLA', 'AFB'}

    def test_a_symbol_with_no_listing_returns_nothing_rather_than_raising(self, prices):
        assert closes(prices, ['NOSUCH']) == set()

    def test_a_symbol_listed_only_outside_the_price_tables_returns_nothing(self, prices):
        assert closes(prices, ['BRK']) == set()


class TestCategories:

    def test_a_fund_reports_its_own_category(self, prices):
        # A symbol is listed once per source table, so reading the listing
        # positionally picked up the table name as the symbol and left every
        # category Unknown.
        assert prices.get_categories(['GOVT']) == {'GOVT': 'ETF'}

    def test_an_equity_reports_its_own_category(self, prices):
        assert prices.get_categories(['AAPL']) == {'AAPL': 'Domestic Common Stock'}

    def test_an_unlisted_symbol_is_unknown(self, prices):
        assert prices.get_categories(['NOSUCH']) == {'NOSUCH': 'Unknown'}

    def test_a_repeated_lookup_does_not_ask_again(self, prices):
        calls = []
        original = AsyncNDLClient.get_table

        async def counted(self, table_name, **kwargs):
            calls.append(table_name)
            return await original(self, table_name, **kwargs)

        with patch.object(AsyncNDLClient, 'get_table', counted):
            prices.get_categories(['GOVT', 'NOSUCH'])
            before = len(calls)
            prices.get_categories(['GOVT', 'NOSUCH'])

        # Including a symbol that has no listing at all, which otherwise looks
        # unresolved forever and is asked about on every call.
        assert len(calls) == before


class TestSplit:

    def test_funds_and_equities_are_separated(self, prices):
        assert prices.split_by_category(['AAPL', 'GOVT', 'PDI', 'NRDS']) == (
            ['AAPL', 'NRDS'], ['GOVT', 'PDI'])

    def test_an_unlisted_symbol_goes_to_equities(self, prices):
        # Neither table has it, so the choice does not change the answer, but
        # dropping it silently would.
        assert prices.split_by_category(['NOSUCH']) == (['NOSUCH'], [])


class TestAsync:

    async def test_async_routing_matches_sync(self, prices):
        df = await prices.async_query(columns=['close'],
                                      ticker=['AAPL', 'GOVT', 'DTLA'],
                                      date_gte=DATES[0], date_lte=DATES[-1])
        assert set(df.index.get_level_values('ticker')) == {'AAPL', 'GOVT', 'DTLA'}
