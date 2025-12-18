import pytest
import json
import pandas as pd
from pathlib import Path
from io import StringIO
from math import isclose
from unittest.mock import patch
from urllib.parse import parse_qs

import cache
from cache import CachedTable, SEPTable


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
    """Create SEPTable with in-memory database."""
    with patch.object(SEPTable, '__init__', lambda self, db_path=':memory:': CachedTable.__init__(self, ':memory:')):
        table = SEPTable(':memory:')
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
    """Test the immutable_data transformation."""

    def test_split_factor_calculation(self):
        """split_factor = closeunadj / close"""
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

        # split_factor = closeunadj / close (exact)
        assert immutable['split_factor'].iloc[0] == 499.23 / 124.81
        assert immutable['split_factor'].iloc[1] == 1.0

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
        split_factor = 499.23 / 124.81
        assert immutable['openunadj'].iloc[0] == 126.01 * split_factor

    def test_volume_unadj_calculation(self):
        """volume_unadj = volume / split_factor (inverse of price)"""
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

        # closeunadj = close * split_factor should match original exactly
        assert immutable['closeunadj'].iloc[0] == 499.23
        assert immutable['closeunadj'].iloc[1] == 213.25


class TestDerivedData:
    """Test the derived_data transformation."""

    def test_price_derived_from_unadj(self):
        """price = price_unadj / split_factor"""
        immutable = pd.DataFrame({
            'split_factor': [4.0, 1.0],
            'split_dividend_factor': [4.1, 1.03],
            'openunadj': [504.0, 127.58],
            'highunadj': [506.0, 131.00],
            'lowunadj': [498.0, 126.00],
            'closeunadj': [499.23, 129.04],
            'volumeunadj': [46907500, 223506000],
        })

        derived = SEPTable.derived_data(immutable)

        # open = openunadj / split_factor (exact)
        assert derived['open'].iloc[0] == 504.0 / 4.0
        assert derived['open'].iloc[1] == 127.58

    def test_volume_derived_from_unadj(self):
        """volume = volume_unadj * split_factor (inverse of price)"""
        immutable = pd.DataFrame({
            'split_factor': [4.0, 1.0],
            'split_dividend_factor': [4.1, 1.03],
            'openunadj': [504.0, 127.58],
            'highunadj': [506.0, 131.00],
            'lowunadj': [498.0, 126.00],
            'closeunadj': [499.23, 129.04],
            'volumeunadj': [46907500, 223506000],
        })

        derived = SEPTable.derived_data(immutable)

        # volume = volumeunadj * split_factor (exact)
        assert derived['volume'].iloc[0] == 46907500 * 4.0
        assert derived['volume'].iloc[1] == 223506000

    def test_adj_prices_derived(self):
        """price_adj = price_unadj / split_dividend_factor"""
        immutable = pd.DataFrame({
            'split_factor': [4.0],
            'split_dividend_factor': [4.1],
            'openunadj': [504.0],
            'highunadj': [506.0],
            'lowunadj': [498.0],
            'closeunadj': [499.23],
            'volumeunadj': [46907500],
        })

        derived = SEPTable.derived_data(immutable)

        # openadj = openunadj / split_dividend_factor (exact)
        assert derived['openadj'].iloc[0] == 504.0 / 4.1


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
        assert 'ticker' in df.columns
        assert 'date' in df.columns
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


class TestSetIntersection:
    """Test that the cache respects set intersections to avoid redundant queries."""

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
        assert df1['ticker'].iloc[0] == 'MSFT'
        assert len(track_api_calls) == 1, "Should have made exactly 1 API call for MSFT"

        # Second query: MSFT + AAPL
        df2 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-28',
            date_lte='2020-08-28'
        )
        assert len(df2) == 2
        assert set(df2['ticker'].tolist()) == {'MSFT', 'AAPL'}

        # Cover solver: only fetches AAPL (MSFT already in cache)
        assert len(track_api_calls) == 2, f"Expected 2 total API calls, got {len(track_api_calls)}"

        # Second call fetches only AAPL (efficient - no overfetch)
        second_call = track_api_calls[1]
        ticker_arg = second_call['kwargs'].get('ticker')
        assert ticker_arg == 'AAPL', \
            f"Second API call should request only AAPL, got ticker={ticker_arg}"

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
        assert str(df1['date'].iloc[0])[:10] == '2020-08-31'
        assert len(track_api_calls) == 1, "Should have made exactly 1 API call for Monday"

        # Second query: Monday + Tuesday (2020-08-31 to 2020-09-01)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-01'
        )
        assert len(df2) == 2

        # Should only have made ONE additional call for Tuesday (not Monday again)
        assert len(track_api_calls) == 2, f"Expected 2 total API calls, got {len(track_api_calls)}"

        # Verify second call was only for Tuesday
        second_call = track_api_calls[1]
        date_filter = second_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-01', \
            f"Second API call should start from Tuesday, got {date_filter}"

    def test_date_range_gap_fill(self, sep, track_api_calls):
        """
        Query Monday, then Wednesday, then Monday-Wednesday.
        The third query should only fetch Tuesday (the gap).
        """
        # First query: Monday (2020-08-31)
        df1 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-08-31'
        )
        assert len(df1) == 1
        assert len(track_api_calls) == 1

        # Second query: Wednesday (2020-09-02)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-09-02',
            date_lte='2020-09-02'
        )
        assert len(df2) == 1
        assert len(track_api_calls) == 2

        # Third query: Monday through Wednesday
        df3 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-31',
            date_lte='2020-09-02'
        )
        assert len(df3) == 3

        # Should have made exactly ONE more call for Tuesday only
        assert len(track_api_calls) == 3, f"Expected 3 total API calls, got {len(track_api_calls)}"

        # Verify third call was only for Tuesday
        third_call = track_api_calls[2]
        date_filter = third_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-09-01', \
            f"Third API call should start from Tuesday, got {date_filter}"
        assert date_filter.get('lte') == '2020-09-01', \
            f"Third API call should end on Tuesday, got {date_filter}"

    def test_multi_ticker_different_gaps(self, sep, track_api_calls):
        """
        Query MSFT Mon-Wed, then AAPL Wed-Fri, then both Mon-Fri.
        With overfetch strategy, third query makes ONE call for full rectangle.
        """
        # First query: MSFT Monday-Wednesday (2020-08-31 to 2020-09-02)
        df1 = sep.query(
            columns=['close'],
            ticker='MSFT',
            date_gte='2020-08-31',
            date_lte='2020-09-02'
        )
        assert len(df1) == 3
        assert df1['ticker'].iloc[0] == 'MSFT'
        assert len(track_api_calls) == 1

        # Second query: AAPL Wednesday-Friday (2020-09-02 to 2020-09-04)
        df2 = sep.query(
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-09-02',
            date_lte='2020-09-04'
        )
        assert len(df2) == 3
        assert df2['ticker'].iloc[0] == 'AAPL'
        assert len(track_api_calls) == 2

        # Third query: Both tickers Monday-Friday
        df3 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-31',
            date_lte='2020-09-04'
        )
        # 5 trading days * 2 tickers = 10 rows
        assert len(df3) == 10, f"Expected 10 rows, got {len(df3)}"
        assert set(df3['ticker'].unique()) == {'MSFT', 'AAPL'}

        # With overfetch: ONE call for full rectangle (both tickers × Mon-Fri)
        assert len(track_api_calls) == 3, f"Expected 3 total API calls, got {len(track_api_calls)}"

        # Verify overfetch call covers full rectangle
        overfetch_call = track_api_calls[2]
        assert set(overfetch_call['kwargs'].get('ticker')) == {'MSFT', 'AAPL'}
        date_filter = overfetch_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-08-31'
        assert date_filter.get('lte') == '2020-09-04'

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
        assert set(df1['ticker'].tolist()) == {'MSFT', 'AAPL'}
        # One batched call for both tickers
        assert len(track_api_calls) == 1
        assert set(track_api_calls[0]['kwargs'].get('ticker')) == {'MSFT', 'AAPL'}

        # Second query: Both tickers Tuesday (2020-09-01)
        df2 = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-09-01',
            date_lte='2020-09-01'
        )
        assert len(df2) == 2
        assert set(df2['ticker'].tolist()) == {'MSFT', 'AAPL'}

        # Should make only ONE additional batched call for both tickers
        assert len(track_api_calls) == 2, \
            f"Expected 2 total API calls (1 initial + 1 batched), got {len(track_api_calls)}"

        # Verify the second call includes both tickers
        second_call = track_api_calls[1]
        ticker_arg = second_call['kwargs'].get('ticker')
        assert set(ticker_arg) == {'MSFT', 'AAPL'}, \
            f"Second call should request both tickers, got {ticker_arg}"

    def test_cover_solver_batches_efficiently(self, sep, track_api_calls):
        """
        Cover solver batches tickers efficiently, fetching only missing gaps.

        Cache state:
          MSFT: has Mon, Wed (missing Tue)
          AAPL: has Wed only (missing Mon, Tue)

        Query: both tickers Mon-Wed

        Cover solver finds optimal solution: one call for both tickers Mon-Tue
        (doesn't overfetch Wed since both already have it cached)
        """
        # Setup: Cache MSFT for Mon and Wed
        sep.query(columns=['close'], ticker='MSFT', date_gte='2020-08-31', date_lte='2020-08-31')
        sep.query(columns=['close'], ticker='MSFT', date_gte='2020-09-02', date_lte='2020-09-02')
        # Setup: Cache AAPL for Wed only
        sep.query(columns=['close'], ticker='AAPL', date_gte='2020-09-02', date_lte='2020-09-02')

        setup_calls = len(track_api_calls)
        assert setup_calls == 3, f"Setup should make 3 calls, got {setup_calls}"

        # Now query both tickers for Mon-Wed
        df = sep.query(
            columns=['close'],
            ticker=['MSFT', 'AAPL'],
            date_gte='2020-08-31',
            date_lte='2020-09-02'
        )

        # Should have all 6 rows (2 tickers × 3 days)
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

        # Cover solver makes only 1 additional call (batched for gaps only)
        assert len(track_api_calls) == 4, \
            f"Expected 4 total calls (3 setup + 1 gap-fill), got {len(track_api_calls)}"

        # The gap-fill call requests both tickers for Mon-Tue (not Wed, already cached)
        gap_fill_call = track_api_calls[3]
        ticker_arg = gap_fill_call['kwargs'].get('ticker')
        assert set(ticker_arg) == {'MSFT', 'AAPL'}, f"Expected both tickers, got {ticker_arg}"
        date_filter = gap_fill_call['kwargs'].get('date', {})
        assert date_filter.get('gte') == '2020-08-31'
        assert date_filter.get('lte') == '2020-09-01'  # Mon-Tue, not Wed


class TestFilterSplitting:
    """Tests for automatic request splitting to stay under page limit."""

    def test_small_request_not_split(self):
        """Requests under page limit should not be split."""
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = CachedTable._split_filters(filters)
        assert len(chunks) == 1
        assert chunks[0] == filters

    def test_multi_ticker_split(self):
        """Large multi-ticker requests should be split by ticker groups."""
        # 100 tickers × 252 days = 25,200 rows > 10k limit
        tickers = [f'T{i:03d}' for i in range(100)]
        filters = {'ticker': tickers, 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = CachedTable._split_filters(filters)

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should have fewer tickers
        for chunk in chunks:
            chunk_tickers = chunk.get('ticker')
            if isinstance(chunk_tickers, list):
                assert len(chunk_tickers) < 100
            # Estimate should be under split threshold
            est = CachedTable._estimate_rows(chunk)
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

    def test_long_date_range_split(self):
        """Single ticker with very long date range should split by dates."""
        # 1 ticker × 50 years ≈ 12,600 rows > 10k limit
        filters = {'ticker': 'AAPL', 'date_gte': '1975-01-01', 'date_lte': '2024-12-31'}
        chunks = CachedTable._split_filters(filters)

        # Should split into multiple date ranges
        assert len(chunks) > 1

        # Each chunk should cover non-overlapping date ranges
        for chunk in chunks:
            assert chunk['ticker'] == 'AAPL'
            assert 'date_gte' in chunk
            assert 'date_lte' in chunk
            # Estimate should be under split threshold
            est = CachedTable._estimate_rows(chunk)
            assert est < cache.NDL_SPLIT_THRESHOLD

    def test_estimate_trading_days(self):
        """Trading day estimation should be reasonable."""
        # 1 year ≈ 252 trading days
        est = CachedTable._estimate_trading_days('2024-01-01', '2024-12-31')
        assert 250 <= est <= 260

        # 1 month ≈ 21 trading days
        est = CachedTable._estimate_trading_days('2024-01-01', '2024-01-31')
        assert 15 <= est <= 25

    def test_estimate_rows(self):
        """Row estimation should account for tickers and dates."""
        # Single ticker, 1 year
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = CachedTable._estimate_rows(filters)
        assert 250 <= est <= 260

        # 10 tickers, 1 year
        filters = {'ticker': [f'T{i}' for i in range(10)], 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = CachedTable._estimate_rows(filters)
        assert 2500 <= est <= 2600
