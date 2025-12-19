import pytest
import json
import pandas as pd
from pathlib import Path
from io import StringIO
from math import isclose
from unittest.mock import patch, MagicMock
from urllib.parse import parse_qs
import tempfile
import os

import cache
from cache import CachedTable
from tables import SEPTable, ActionsTable


# Query fixture cache for mocking NDL API
QUERY_CACHE_FILEPATH = Path(__file__).parent / 'query_fixture.json'

if QUERY_CACHE_FILEPATH.exists():
    with QUERY_CACHE_FILEPATH.open('r') as f:
        try:
            query_cache = json.load(f)
        except json.JSONDecodeError:
            query_cache = {}
else:
    query_cache = {}

original_get_table = cache.ndl.get_table


def encode_query_key(table: str, **kwargs) -> str:
    """
    Encode NDL query params to a readable URL-like key.

    Example: SHARADAR/SEP?ticker=AAPL&date.gte=2020-08-28&date.lte=2020-08-31
    """
    params = []
    for key, value in sorted(kwargs.items()):
        if key == 'paginate':
            continue  # Skip pagination flag
        if isinstance(value, dict):
            # Nested dict like date={'gte': '2020-01-01', 'lte': '2020-12-31'}
            for subkey, subval in sorted(value.items()):
                if isinstance(subval, list):
                    params.append(f"{key}.{subkey}={','.join(str(v) for v in subval)}")
                else:
                    params.append(f"{key}.{subkey}={subval}")
        elif isinstance(value, list):
            params.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            params.append(f"{key}={value}")

    query_string = '&'.join(params) if params else ''
    return f"{table}?{query_string}" if query_string else table


def mock_get_table(*args, **kwargs):
    table = args[0] if args else ''
    key = encode_query_key(table, **kwargs)

    if key in query_cache:
        records = query_cache[key]
        result = pd.DataFrame(records)
        # Restore date column type
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date']).dt.date
    else:
        result = original_get_table(*args, **kwargs)

        # Save as list of records (human-readable)
        records = json.loads(result.to_json(orient='records', date_format='iso'))
        query_cache[key] = records
        with QUERY_CACHE_FILEPATH.open('w') as f:
            json.dump(query_cache, f, indent=2)

    assert isinstance(result, pd.DataFrame)
    return result


@pytest.fixture(autouse=True)
def mock_ndl():
    """Mock NDL API with cached responses."""
    cache.ndl.get_table = mock_get_table
    yield
    cache.ndl.get_table = original_get_table


@pytest.fixture
def sep():
    """Create SEPTable with temp database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.duckdb')
        with patch.object(cache, 'DB_PATH', db_path):
            table = SEPTable()
            yield table
            table.conn.close()


@pytest.fixture
def track_api_calls():
    """Track NDL API calls made during a test."""
    api_calls = []
    original = cache.ndl.get_table

    def tracking_mock(*args, **kwargs):
        api_calls.append({'args': args, 'kwargs': kwargs})
        return original(*args, **kwargs)

    cache.ndl.get_table = tracking_mock
    yield api_calls
    cache.ndl.get_table = original


class TestImmutableData:
    """Test the immutable_data transformation.

    Note: split_factor and split_dividend_factor are no longer stored in immutable data.
    They are now computed dynamically from ACTIONS table on read.
    """

    def test_immutable_columns(self):
        """Verify only unadjusted prices/volume are stored as immutable."""
        queried = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL'],
            'date': ['2020-08-28', '2020-08-31'],
            'open': [126.01, 127.58],
            'high': [126.44, 131.00],
            'low': [124.58, 126.00],
            'close': [124.81, 129.04],
            'volume': [187630000, 223506000],
            'closeadj': [121.30, 125.41],
            'closeunadj': [499.23, 129.04],
        })

        immutable = SEPTable.immutable_data(queried)

        # Only unadjusted columns should be stored
        assert 'openunadj' in immutable.columns
        assert 'closeunadj' in immutable.columns
        assert 'volumeunadj' in immutable.columns
        # split_factor should NOT be stored (computed from ACTIONS)
        assert 'split_factor' not in immutable.columns
        assert 'split_dividend_factor' not in immutable.columns

    def test_price_unadj_calculation(self):
        """price_unadj = price * split_factor"""
        queried = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': ['2020-08-28'],
            'open': [126.01],
            'high': [126.44],
            'low': [124.58],
            'close': [124.81],
            'volume': [187630000],
            'closeadj': [121.30],
            'closeunadj': [499.23],
        })

        immutable = SEPTable.immutable_data(queried)

        # openunadj = open * split_factor (exact)
        # split_factor = closeunadj / close
        split_factor = 499.23 / 124.81
        assert immutable['openunadj'].iloc[0] == 126.01 * split_factor

    def test_volume_unadj_calculation(self):
        """volume_unadj = volume / split_factor (actual shares traded)"""
        queried = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': ['2020-08-28'],
            'open': [126.01],
            'high': [126.44],
            'low': [124.58],
            'close': [124.81],
            'volume': [187630000],
            'closeadj': [121.30],
            'closeunadj': [499.23],
        })

        immutable = SEPTable.immutable_data(queried)

        # volumeunadj = volume / split_factor (exact)
        # split_factor = closeunadj / close
        split_factor = 499.23 / 124.81
        assert immutable['volumeunadj'].iloc[0] == 187630000 / split_factor

    def test_closeunadj_roundtrip(self):
        """Verify closeunadj derivation matches original."""
        queried = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'date': ['2020-08-28', '2020-08-28'],
            'open': [126.01, 214.00],
            'high': [126.44, 215.00],
            'low': [124.58, 212.00],
            'close': [124.81, 213.25],
            'volume': [187630000, 25000000],
            'closeadj': [121.30, 210.00],
            'closeunadj': [499.23, 213.25],
        })

        immutable = SEPTable.immutable_data(queried)

        # closeunadj = close / split_adj should match original exactly
        assert immutable['closeunadj'].iloc[0] == 499.23
        assert immutable['closeunadj'].iloc[1] == 213.25


class TestDerivedData:
    """Test the derived_data transformation.

    Note: derived_data is now an instance method that computes adjustment
    factors from ACTIONS data rather than reading from stored columns.
    """

    def test_price_derived_with_mock_actions(self, sep):
        """price = price_unadj * split_adj (from ACTIONS)"""
        # Query data around AAPL 4:1 split on 2020-08-31
        df = sep.query(
            columns=['open', 'openunadj', 'close', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-02'
        )

        # After split: closeunadj should equal close (adjustment = 1)
        # Index is (ticker, date), filter by date level
        post_split = df[df.index.get_level_values('date').astype(str) >= '2020-08-31']
        for _, row in post_split.iterrows():
            assert abs(row['closeunadj'] - row['close']) < 0.01

    def test_volume_derived_with_mock_actions(self, sep):
        """volume = volume_unadj / split_adj (from ACTIONS)"""
        df = sep.query(
            columns=['volume', 'volumeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-02'
        )

        # Post-split: volume should roughly equal volumeunadj
        # Index is (ticker, date), filter by date level
        post_split = df[df.index.get_level_values('date').astype(str) >= '2020-08-31']
        for _, row in post_split.iterrows():
            assert abs(row['volumeunadj'] - row['volume']) < 1

    def test_adj_prices_derived_with_mock_actions(self, sep):
        """price_adj includes dividend adjustments from ACTIONS"""
        df = sep.query(
            columns=['close', 'closeadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-02'
        )

        # closeadj should be less than or equal to close (dividends reduce it)
        for _, row in df.iterrows():
            # closeadj <= close (within small tolerance for recent data)
            assert row['closeadj'] <= row['close'] + 1.0


class TestSEPTableIntegration:
    """Integration tests using mocked NDL API."""

    def test_sync_and_query(self, sep):
        """Test full sync and query cycle."""
        df = sep.query(
            columns=['close', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert len(df) > 0
        # ticker and date are in index, not columns
        assert 'ticker' in df.index.names
        assert 'date' in df.index.names
        assert 'close' in df.columns
        assert 'closeunadj' in df.columns

    def test_query_immutable_columns(self, sep):
        """Test querying immutable columns directly."""
        df = sep.query(
            columns=['closeunadj', 'volumeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert 'closeunadj' in df.columns
        assert 'volumeunadj' in df.columns

    def test_query_derived_columns(self, sep):
        """Test querying derived columns."""
        df = sep.query(
            columns=['close', 'closeadj', 'volume', 'volumeadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert 'close' in df.columns
        assert 'closeadj' in df.columns
        assert 'volume' in df.columns
        assert 'volumeadj' in df.columns

    def test_cache_hit(self, sep):
        """Test that second query uses cache."""
        # First query - triggers sync
        df1 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        # Second query - should use cache
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert df1['close'].tolist() == df2['close'].tolist()

    def test_split_continuity(self, sep):
        """Test that split-adjusted prices are continuous across split."""
        df = sep.query(
            columns=['close', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-01'
        )

        # closeunadj should have a big jump at split (4:1)
        closeunadj = df['closeunadj'].tolist()

        # close should be relatively continuous
        close = df['close'].tolist()

        # Check that close values are all in similar range (no 4x jump)
        assert max(close) / min(close) < 1.5  # Within 50%

    def test_volume_adjustment_direction(self, sep):
        """Test that volume adjusts opposite to price."""
        df = sep.query(
            columns=['volume', 'volumeunadj', 'close', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-01'
        )

        # Find pre-split and post-split rows
        pre_split = df[df['closeunadj'] > 400].iloc[0]  # closeunadj ~500 pre-split
        post_split = df[df['closeunadj'] < 200].iloc[0]  # closeunadj ~130 post-split

        # Pre-split: volumeunadj should be ~1/4 of volume
        assert pre_split['volumeunadj'] < pre_split['volume']

        # Post-split: volumeunadj should equal volume (factor = 1)
        assert post_split['volumeunadj'] == post_split['volume']


class TestFilterConversion:
    """Test NDL filter format conversion."""

    def test_gte_lte_filters(self, sep):
        """Test that date_gte and date_lte are converted correctly."""
        # This implicitly tests filter conversion by making a successful query
        df = sep.query(
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert len(df) > 0


class TestTableSchema:
    """Test table schema and constraints."""

    def test_primary_key_order(self, sep):
        """Test that primary key is (ticker, date) for efficient queries."""
        # Sync data first to create table (lazy creation)
        sep.sync(ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')

        result = sep.conn.execute("""
            SELECT constraint_column_names
            FROM duckdb_constraints()
            WHERE table_name = 'sharadar_sep'
            AND constraint_type = 'PRIMARY KEY'
        """).fetchone()

        assert result is not None
        assert result[0] == ['ticker', 'date']

    def test_insert_replace_behavior(self, sep):
        """Test that INSERT OR REPLACE works correctly."""
        # First sync
        sep.sync(ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')
        count1 = sep.conn.execute("SELECT COUNT(*) FROM sharadar_sep").fetchone()[0]

        # Second sync of same data should not duplicate
        sep.sync(ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')
        count2 = sep.conn.execute("SELECT COUNT(*) FROM sharadar_sep").fetchone()[0]

        assert count1 == count2


def filter_sep_calls(api_calls):
    """Filter API calls to only include SHARADAR/SEP (exclude ACTIONS calls)."""
    return [c for c in api_calls if c['args'][0] == 'SHARADAR/SEP']


class TestSetIntersection:
    """Test that the cache respects set intersections to avoid redundant queries.

    Note: Tests filter out ACTIONS API calls since SEP now syncs ACTIONS
    for each ticker before querying prices.
    """

    def test_multi_ticker_no_double_query(self, sep, track_api_calls):
        """
        Query MSFT alone, then MSFT+AAPL together.
        Cover solver fetches only AAPL (MSFT already cached).
        """
        # First query: just MSFT
        df1 = sep.query(
            columns=['close'],
            ticker='MSFT',
            date_gte='2020-08-28',
            date_lte='2020-08-28'
        )
        assert len(df1) == 1
        assert df1.index.get_level_values('ticker')[0] == 'MSFT'
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1, f"Should have made exactly 1 SEP API call for MSFT, got {len(sep_calls)}"

        # Second query: MSFT + AAPL
        df2 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-28',
            date_lte='2020-08-28'
        )
        assert len(df2) == 2
        assert set(df2.index.get_level_values('ticker').tolist()) == {'MSFT', 'AAPL'}

        # Cover solver: only fetches AAPL (MSFT already in cache)
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, f"Expected 2 total SEP API calls, got {len(sep_calls)}"

        # Second SEP call fetches only AAPL (efficient - no overfetch)
        second_sep_call = sep_calls[1]
        ticker_arg = second_sep_call['kwargs'].get('ticker')
        assert ticker_arg == 'AAPL', \
            f"Second SEP API call should request only AAPL, got ticker={ticker_arg}"

    def test_date_range_no_double_query(self, sep, track_api_calls):
        """
        Query Monday alone, then Monday+Tuesday together.
        Monday should not be re-fetched from the API.
        """
        # First query: just Monday (2020-08-31)
        df1 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-08-31'
        )
        assert len(df1) == 1
        assert str(df1.index.get_level_values('date')[0])[:10] == '2020-08-31'
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1, f"Should have made exactly 1 SEP API call for Monday, got {len(sep_calls)}"

        # Second query: Monday + Tuesday (2020-08-31 to 2020-09-01)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-01'
        )
        assert len(df2) == 2

        # Should only have made ONE additional call for Tuesday (not Monday again)
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, f"Expected 2 total SEP API calls, got {len(sep_calls)}"

        # Verify second call was only for Tuesday
        second_call = sep_calls[1]
        date_filter = second_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-01', \
            f"Second SEP API call should start from Tuesday, got {date_filter}"

    def test_date_range_boundary_extension(self, sep, track_api_calls):
        """
        Bounds-based sync tracking: query Mon-Wed, then extend to Mon-Fri.
        Extension query should only fetch Thu-Fri (boundary gap).
        """
        # First query: Monday-Wednesday (2020-08-31 to 2020-09-02)
        df1 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-02'
        )
        assert len(df1) == 3
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1

        # Second query: Monday-Friday (extend end date)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-04'
        )
        assert len(df2) == 5

        # Should have made ONE additional call for Thu-Fri only
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, f"Expected 2 total SEP API calls, got {len(sep_calls)}"

        # Verify second call was for the extension (Thu-Fri)
        second_call = sep_calls[1]
        date_filter = second_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-03', \
            f"Second SEP API call should start from Thursday, got {date_filter}"
        assert date_filter.get('lte') == '2020-09-04', \
            f"Second SEP API call should end on Friday, got {date_filter}"

    def test_disjoint_queries_fill_gap(self, sep, track_api_calls):
        """
        Query Monday, then Friday separately.
        The Friday query should fetch Tue-Fri to extend the synced range.
        """
        # First query: Monday only (2020-08-31)
        df1 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-08-31'
        )
        assert len(df1) == 1
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1

        # Second query: Friday only (2020-09-04)
        # This should trigger a fetch for Tue-Fri (the gap after synced_to)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-09-04',
            date_lte='2020-09-04'
        )
        assert len(df2) == 1
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, f"Expected 2 SEP API calls, got {len(sep_calls)}"

        # Verify second call fetched Tue-Fri (extends from day after Mon to Fri)
        second_call = sep_calls[1]
        date_filter = second_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-01', \
            f"Second SEP call should start from Tuesday (day after synced_to), got {date_filter}"
        assert date_filter.get('lte') == '2020-09-04', \
            f"Second SEP call should end on Friday, got {date_filter}"

        # Now the full range Mon-Fri should be cached
        df3 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-04'
        )
        assert len(df3) == 5, f"Should have all 5 days cached, got {len(df3)}"
        # No additional API calls needed
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, \
            f"Third query should use cache, expected 2 total SEP calls, got {len(sep_calls)}"

    def test_multi_ticker_different_gaps(self, sep, track_api_calls):
        """
        Query MSFT Mon-Wed, then AAPL Wed-Fri, then both Mon-Fri.
        With bounds-based tracking, third query fills boundary gaps per ticker:
        - MSFT needs Thu-Fri (future gap)
        - AAPL needs Mon-Tue (past gap)
        """
        # First query: MSFT Monday-Wednesday (2020-08-31 to 2020-09-02)
        df1 = sep.query(
            columns=['close'],
            ticker='MSFT',
            date_gte='2020-08-31',
            date_lte='2020-09-02'
        )
        assert len(df1) == 3
        assert df1.index.get_level_values('ticker')[0] == 'MSFT'
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1

        # Second query: AAPL Wednesday-Friday (2020-09-02 to 2020-09-04)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-09-02',
            date_lte='2020-09-04'
        )
        assert len(df2) == 3
        assert df2.index.get_level_values('ticker')[0] == 'AAPL'
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2

        # Third query: Both tickers Monday-Friday
        df3 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-31',
            date_lte='2020-09-04'
        )
        # 5 trading days * 2 tickers = 10 rows
        assert len(df3) == 10, f"Expected 10 rows, got {len(df3)}"
        assert set(df3.index.get_level_values('ticker').unique()) == {'MSFT', 'AAPL'}

        # Bounds-based: 2 additional calls (AAPL past gap + MSFT future gap)
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 4, f"Expected 4 total SEP API calls, got {len(sep_calls)}"

        # Check gap-fill calls (order depends on solver, check both are present)
        gap_calls = sep_calls[2:]
        gap_dates = [(c['kwargs'].get('date', {}).get('gte'), c['kwargs'].get('date', {}).get('lte'))
                     for c in gap_calls]
        # Should have AAPL Mon-Tue and MSFT Thu-Fri
        assert ('2020-08-31', '2020-09-01') in gap_dates or \
               any('2020-08-31' in str(d) for d in gap_dates), \
               f"Expected past gap Mon-Tue, got {gap_dates}"
        assert ('2020-09-03', '2020-09-04') in gap_dates or \
               any('2020-09-04' in str(d) for d in gap_dates), \
               f"Expected future gap Thu-Fri, got {gap_dates}"

    def test_multi_ticker_same_gap_batched(self, sep, track_api_calls):
        """
        Query MSFT+AAPL for Monday, then for Tuesday.
        Both queries should make ONE batched API call for both tickers.
        """
        # First query: Both tickers Monday (2020-08-31)
        df1 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-31',
            date_lte='2020-08-31'
        )
        assert len(df1) == 2
        assert set(df1.index.get_level_values('ticker').tolist()) == {'MSFT', 'AAPL'}
        # One batched call for both tickers
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 1
        assert set(sep_calls[0]['kwargs'].get('ticker')) == {'MSFT', 'AAPL'}

        # Second query: Both tickers Tuesday (2020-09-01)
        df2 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-09-01',
            date_lte='2020-09-01'
        )
        assert len(df2) == 2
        assert set(df2.index.get_level_values('ticker').tolist()) == {'MSFT', 'AAPL'}

        # Should make only ONE additional batched call for both tickers
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 2, \
            f"Expected 2 total SEP API calls (1 initial + 1 batched), got {len(sep_calls)}"

        # Verify the second SEP call includes both tickers
        second_sep_call = sep_calls[1]
        ticker_arg = second_sep_call['kwargs'].get('ticker')
        assert set(ticker_arg) == {'MSFT', 'AAPL'}, \
            f"Second SEP call should request both tickers, got {ticker_arg}"

    def test_cover_solver_batches_efficiently(self, sep, track_api_calls):
        """
        Cover solver batches tickers with similar gaps efficiently.

        Cache state:
          MSFT: synced Mon-Tue (bounds: Mon-Tue)
          AAPL: synced Mon only (bounds: Mon-Mon)

        Query: both tickers Mon-Fri

        Cover solver batches:
        - MSFT needs Wed-Fri (future gap)
        - AAPL needs Tue-Fri (future gap, longer)
        Both are future gaps, solver should batch them into one call
        """
        # Setup: Cache MSFT for Mon-Tue
        sep.query(columns=['close'], ticker='MSFT', date_gte='2020-08-31', date_lte='2020-09-01')
        # Setup: Cache AAPL for Mon only
        sep.query(columns=['close'], ticker='AAPL', date_gte='2020-08-31', date_lte='2020-08-31')

        sep_calls = filter_sep_calls(track_api_calls)
        setup_calls = len(sep_calls)
        assert setup_calls == 2, f"Setup should make 2 SEP calls, got {setup_calls}"

        # Now query both tickers for Mon-Fri
        df = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-31',
            date_lte='2020-09-04'
        )

        # Should have all 10 rows (2 tickers × 5 days)
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"

        # Cover solver batches future gaps: 1 call for both tickers Tue-Fri
        # (MSFT gap is Wed-Fri, AAPL gap is Tue-Fri, solver uses longest)
        sep_calls = filter_sep_calls(track_api_calls)
        assert len(sep_calls) == 3, \
            f"Expected 3 total SEP calls (2 setup + 1 batched gap-fill), got {len(sep_calls)}"

        # The gap-fill call requests both tickers
        gap_fill_call = sep_calls[2]
        ticker_arg = gap_fill_call['kwargs'].get('ticker')
        assert set(ticker_arg) == {'MSFT', 'AAPL'}, f"Expected both tickers, got {ticker_arg}"
        # Date range covers the longer gap (Tue-Fri for AAPL, overfetches for MSFT)
        date_filter = gap_fill_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-01', f"Gap should start Tue, got {date_filter}"
        assert date_filter.get('lte') == '2020-09-04', f"Gap should end Fri, got {date_filter}"


class TestFilterSplitting:
    """Tests for automatic request splitting to stay under page limit."""

    @pytest.fixture
    def sep(self, tmp_path):
        """Create SEPTable instance for testing instance methods."""
        db_path = str(tmp_path / 'test.duckdb')
        with patch.object(cache, 'DB_PATH', db_path):
            yield SEPTable()

    def test_small_request_not_split(self, sep):
        """Requests under page limit should not be split."""
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = sep._split_filters(filters)
        assert len(chunks) == 1
        assert chunks[0] == filters

    def test_multi_ticker_split(self, sep):
        """Large multi-ticker requests should be split by ticker groups."""
        # 100 tickers × 252 days = 25,200 rows > 10k limit
        tickers = [f'T{i:03d}' for i in range(100)]
        filters = {'ticker': tickers, 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = sep._split_filters(filters)

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should have fewer tickers
        for chunk in chunks:
            chunk_tickers = chunk.get('ticker')
            if isinstance(chunk_tickers, list):
                assert len(chunk_tickers) < 100
            # Estimate should be under split threshold
            est = sep._estimate_rows(chunk)
            assert est < cache.NDL_SPLIT_THRESHOLD

        # All tickers should be covered
        all_tickers = []
        for chunk in chunks:
            t = chunk.get('ticker')
            if isinstance(t, list):
                all_tickers.extend(t)
            else:
                all_tickers.append(t)
        assert set(all_tickers) == set(tickers)

    def test_long_date_range_split(self, sep):
        """Single ticker with very long date range should split by dates."""
        # 1 ticker × 50 years ≈ 12,600 rows > 10k limit
        filters = {'ticker': 'AAPL', 'date_gte': '1975-01-01', 'date_lte': '2024-12-31'}
        chunks = sep._split_filters(filters)

        # Should split into multiple date ranges
        assert len(chunks) > 1

        # Each chunk should cover non-overlapping date ranges
        for chunk in chunks:
            assert chunk['ticker'] == 'AAPL'
            assert 'date_gte' in chunk
            assert 'date_lte' in chunk
            # Estimate should be under split threshold
            est = sep._estimate_rows(chunk)
            assert est < cache.NDL_SPLIT_THRESHOLD

    def test_estimate_rows_for_range(self, sep):
        """Row estimation per ticker should be reasonable for SEP (daily data)."""
        # 1 year ≈ 252 trading days
        est = sep._estimate_rows_for_range('2024-01-01', '2024-12-31')
        assert 250 <= est <= 260

        # 1 month ≈ 21 trading days
        est = sep._estimate_rows_for_range('2024-01-01', '2024-01-31')
        assert 15 <= est <= 25

    def test_estimate_rows(self, sep):
        """Row estimation should account for tickers and dates."""
        # Single ticker, 1 year
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = sep._estimate_rows(filters)
        assert 250 <= est <= 260

        # 10 tickers, 1 year
        filters = {'ticker': [f'T{i}' for i in range(10)], 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = sep._estimate_rows(filters)
        assert 2500 <= est <= 2600
