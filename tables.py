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
