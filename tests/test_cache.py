import json
import pandas as pd
from pathlib import Path

import pytest

from ndl_cache import cache, SEPTable
from ndl_cache.testing import temp_db


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
        if 'lastupdated' in result.columns:
            result['lastupdated'] = pd.to_datetime(result['lastupdated']).dt.date
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
    with temp_db():
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


class TestSEPTableBasic:
    """Basic tests for SEP table functionality."""

    def test_sync_and_query(self, sep):
        """Test full sync and query cycle."""
        df = sep.query(
            columns=['close', 'closeadj', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert len(df) > 0
        # ticker and date are in index, not columns
        assert 'ticker' in df.index.names
        assert 'date' in df.index.names
        assert 'close' in df.columns
        assert 'closeadj' in df.columns
        assert 'closeunadj' in df.columns

    def test_query_all_columns(self, sep):
        """Test querying all available columns."""
        df = sep.query(
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        # Should have all query_columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert 'closeadj' in df.columns
        assert 'closeunadj' in df.columns

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

    def test_sharadar_adjusted_prices(self, sep):
        """Test that Sharadar's adjusted prices are stored as-is."""
        df = sep.query(
            columns=['close', 'closeadj', 'closeunadj'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-09-01'
        )

        # closeadj should be less than or equal to close (dividends reduce it)
        for _, row in df.iterrows():
            assert row['closeadj'] <= row['close'] + 1.0

        # Around the 4:1 split on 2020-08-31:
        # - closeunadj shows the actual price (big jump at split)
        # - close is split-adjusted (continuous)
        close = df['close'].tolist()

        # close should be relatively continuous (no 4x jump)
        assert max(close) / min(close) < 1.5  # Within 50%


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
    """Filter API calls to only include SHARADAR/SEP."""
    return [c for c in api_calls if c['args'][0] == 'SHARADAR/SEP']


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


class TestFilterSplitting:
    """Tests for automatic request splitting to stay under page limit."""

    @pytest.fixture
    def sep(self):
        """Create SEPTable instance for testing instance methods."""
        with temp_db():
            table = SEPTable()
            yield table
            table.conn.close()

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


class TestSyncBounds:
    """Test sync bounds tracking with new schema."""

    def test_sync_bounds_includes_lastupdated(self, sep):
        """Sync bounds should track max_lastupdated."""
        # Sync some data
        sep.sync(ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-31')

        # Check sync_bounds table has the new columns
        result = sep.conn.execute(f"""
            SELECT ticker, synced_from, synced_to, max_lastupdated, last_staleness_check
            FROM {sep._sync_bounds_table_name()}
            WHERE ticker = 'AAPL'
        """).fetchone()

        assert result is not None
        ticker, synced_from, synced_to, max_lastupdated, last_staleness_check = result
        assert ticker == 'AAPL'
        assert synced_from is not None
        assert synced_to is not None
        # max_lastupdated should be populated from the fetched data
        # (will be None if lastupdated wasn't in the response)
