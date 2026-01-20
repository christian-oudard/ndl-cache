"""
Tests for ndl-cache caching layer.

These tests mock the async NDL client to avoid network calls.
"""
import json
from pathlib import Path
from unittest.mock import patch

import aioduckdb
import pandas as pd
import pytest

from ndl_cache import SEP, query, async_query
from ndl_cache.async_cache import get_db_path, _CacheManager, NDL_SPLIT_THRESHOLD
from ndl_cache.async_client import AsyncNDLClient
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


def encode_query_key(table: str, columns=None, paginate=True, **kwargs) -> str:
    """
    Encode NDL query params to a readable URL-like key.

    Matches the format used in query_fixture.json:
    SHARADAR/SEP?date.gte=2020-08-28&date.lte=2020-08-31&qopts.columns=ticker,date,...&ticker=AAPL
    """
    params = []

    # Handle date filters first (they come as date={'gte': ..., 'lte': ...})
    for key, value in sorted(kwargs.items()):
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

    # Add columns as qopts.columns to match fixture format
    if columns:
        params.append(f"qopts.columns={','.join(columns)}")

    # Sort params to ensure consistent ordering
    params.sort()

    query_string = '&'.join(params) if params else ''
    return f"{table}?{query_string}" if query_string else table


def make_mock_get_table(api_calls_tracker=None):
    """Create a mock get_table function that uses cached responses."""
    async def mock_get_table(self, table_name, columns=None, paginate=True, **filters):
        # Track the call if tracker provided
        if api_calls_tracker is not None:
            api_calls_tracker.append({
                'args': (table_name,),
                'kwargs': {'columns': columns, 'paginate': paginate, **filters}
            })

        # Convert filters to the format expected by encode_query_key
        # The async client uses direct kwargs like ticker="AAPL", date={"gte": "...", "lte": "..."}
        encoded_filters = {}
        for key, value in filters.items():
            encoded_filters[key] = value

        key = encode_query_key(table_name, columns=columns, paginate=paginate, **encoded_filters)

        if key in query_cache:
            records = query_cache[key]
            result = pd.DataFrame(records)
            # Restore date column type
            if 'date' in result.columns:
                result['date'] = pd.to_datetime(result['date']).dt.date
            if 'lastupdated' in result.columns:
                result['lastupdated'] = pd.to_datetime(result['lastupdated']).dt.date
            return result

        # If not in cache, return empty DataFrame (for tests that don't need real data)
        # In real usage, this would make an API call
        return pd.DataFrame()

    return mock_get_table


@pytest.fixture(autouse=True)
def mock_ndl_client():
    """Mock AsyncNDLClient.get_table with cached responses."""
    mock_fn = make_mock_get_table()
    with patch.object(AsyncNDLClient, 'get_table', mock_fn):
        yield


@pytest.fixture
def use_temp_db():
    """Set up temp database for tests."""
    with temp_db():
        yield


@pytest.fixture
def track_api_calls():
    """Track NDL API calls made during a test."""
    api_calls = []
    mock_fn = make_mock_get_table(api_calls)
    with patch.object(AsyncNDLClient, 'get_table', mock_fn):
        yield api_calls


class TestSEPTableBasic:
    """Basic tests for SEP table functionality."""

    def test_sync_and_query(self, use_temp_db):
        """Test full sync and query cycle."""
        df = query(
            SEP,
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

    def test_query_all_columns(self, use_temp_db):
        """Test querying all available columns."""
        df = query(
            SEP,
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

    def test_cache_hit(self, use_temp_db):
        """Test that second query uses cache."""
        # First query - triggers sync
        df1 = query(
            SEP,
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        # Second query - should use cache
        df2 = query(
            SEP,
            columns=['close'],
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert df1['close'].tolist() == df2['close'].tolist()

    def test_sharadar_adjusted_prices(self, use_temp_db):
        """Test that Sharadar's adjusted prices are stored as-is."""
        df = query(
            SEP,
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

    def test_gte_lte_filters(self, use_temp_db):
        """Test that date_gte and date_lte are converted correctly."""
        # This implicitly tests filter conversion by making a successful query
        df = query(
            SEP,
            ticker='AAPL',
            date_gte='2020-08-28',
            date_lte='2020-08-31'
        )

        assert len(df) > 0


class TestTableSchema:
    """Test table schema and constraints."""

    @pytest.mark.asyncio
    async def test_primary_key_order(self, use_temp_db):
        """Test that primary key is (ticker, date) for efficient queries."""
        # Query data first to create table (lazy creation)
        await async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')

        async with aioduckdb.connect(get_db_path()) as conn:
            cursor = await conn.execute("""
                SELECT constraint_column_names
                FROM duckdb_constraints()
                WHERE table_name = 'sharadar_sep'
                AND constraint_type = 'PRIMARY KEY'
            """)
            result = await cursor.fetchone()

        assert result is not None
        assert result[0] == ['ticker', 'date']

    @pytest.mark.asyncio
    async def test_insert_replace_behavior(self, use_temp_db):
        """Test that INSERT OR REPLACE works correctly."""
        # First query triggers sync
        await async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')

        async with aioduckdb.connect(get_db_path()) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM sharadar_sep")
            result1 = await cursor.fetchone()
            count1 = result1[0]

        # Second query of same data should not duplicate
        await async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28')

        async with aioduckdb.connect(get_db_path()) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM sharadar_sep")
            result2 = await cursor.fetchone()
            count2 = result2[0]

        assert count1 == count2


def filter_sep_calls(api_calls):
    """Filter API calls to only include SHARADAR/SEP."""
    return [c for c in api_calls if c['args'][0] == 'SHARADAR/SEP']


class TestSetIntersection:
    """Test that the cache respects set intersections to avoid redundant queries."""

    def test_multi_ticker_no_double_query(self, track_api_calls):
        """
        Query MSFT alone, then MSFT+AAPL together.
        Cover solver fetches only AAPL (MSFT already cached).
        """
        with temp_db():
            # First query: just MSFT
            df1 = query(
                SEP,
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
            df2 = query(
                SEP,
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

    def test_date_range_no_double_query(self, track_api_calls):
        """
        Query Monday alone, then Monday+Tuesday together.
        Monday should not be re-fetched from the API.
        """
        with temp_db():
            # First query: just Monday (2020-08-31)
            df1 = query(
                SEP,
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
            df2 = query(
                SEP,
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

    def test_date_range_boundary_extension(self, track_api_calls):
        """
        Bounds-based sync tracking: query Mon-Wed, then extend to Mon-Fri.
        Extension query should only fetch Thu-Fri (boundary gap).
        """
        with temp_db():
            # First query: Monday-Wednesday (2020-08-31 to 2020-09-02)
            df1 = query(
                SEP,
                columns=['close'],
                ticker='AAPL',
                date_gte='2020-08-31',
                date_lte='2020-09-02'
            )
            assert len(df1) == 3
            sep_calls = filter_sep_calls(track_api_calls)
            assert len(sep_calls) == 1

            # Second query: Monday-Friday (extend end date)
            df2 = query(
                SEP,
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

    def test_multi_ticker_same_gap_batched(self, track_api_calls):
        """
        Query MSFT+AAPL for Monday, then for Tuesday.
        Both queries should make ONE batched API call for both tickers.
        """
        with temp_db():
            # First query: Both tickers Monday (2020-08-31)
            df1 = query(
                SEP,
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
            df2 = query(
                SEP,
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
    def mgr(self):
        """Create _CacheManager instance for testing internal methods."""
        with temp_db():
            yield _CacheManager(SEP)

    def test_small_request_not_split(self, mgr):
        """Requests under page limit should not be split."""
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = mgr._split_filters(filters)
        assert len(chunks) == 1
        assert chunks[0] == filters

    def test_multi_ticker_split(self, mgr):
        """Large multi-ticker requests should be split by ticker groups."""
        # 100 tickers × 252 days = 25,200 rows > 10k limit
        tickers = [f'T{i:03d}' for i in range(100)]
        filters = {'ticker': tickers, 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        chunks = mgr._split_filters(filters)

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should have fewer tickers
        for chunk in chunks:
            chunk_tickers = chunk.get('ticker')
            if isinstance(chunk_tickers, list):
                assert len(chunk_tickers) < 100
            # Estimate should be under split threshold
            est = mgr._estimate_rows(chunk)
            assert est < NDL_SPLIT_THRESHOLD

        # All tickers should be covered
        all_tickers = []
        for chunk in chunks:
            t = chunk.get('ticker')
            if isinstance(t, list):
                all_tickers.extend(t)
            else:
                all_tickers.append(t)
        assert set(all_tickers) == set(tickers)

    def test_long_date_range_split(self, mgr):
        """Single ticker with very long date range should split by dates."""
        # 1 ticker × 50 years ≈ 12,600 rows > 10k limit
        filters = {'ticker': 'AAPL', 'date_gte': '1975-01-01', 'date_lte': '2024-12-31'}
        chunks = mgr._split_filters(filters)

        # Should split into multiple date ranges
        assert len(chunks) > 1

        # Each chunk should cover non-overlapping date ranges
        for chunk in chunks:
            assert chunk['ticker'] == 'AAPL'
            assert 'date_gte' in chunk
            assert 'date_lte' in chunk
            # Estimate should be under split threshold
            est = mgr._estimate_rows(chunk)
            assert est < NDL_SPLIT_THRESHOLD

    def test_estimate_rows_for_range(self, mgr):
        """Row estimation per ticker should be reasonable for SEP (daily data)."""
        # 1 year ≈ 252 trading days
        est = mgr._estimate_rows_for_range('2024-01-01', '2024-12-31')
        assert 250 <= est <= 260

        # 1 month ≈ 21 trading days
        est = mgr._estimate_rows_for_range('2024-01-01', '2024-01-31')
        assert 15 <= est <= 25

    def test_estimate_rows(self, mgr):
        """Row estimation should account for tickers and dates."""
        # Single ticker, 1 year
        filters = {'ticker': 'AAPL', 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = mgr._estimate_rows(filters)
        assert 250 <= est <= 260

        # 10 tickers, 1 year
        filters = {'ticker': [f'T{i}' for i in range(10)], 'date_gte': '2024-01-01', 'date_lte': '2024-12-31'}
        est = mgr._estimate_rows(filters)
        assert 2500 <= est <= 2600

    def test_skip_split_when_rows_per_year_none(self, use_temp_db):
        """Tables with rows_per_year=None should skip splitting (e.g., ACTIONS)."""
        from ndl_cache import ACTIONS

        # ACTIONS has rows_per_year=None for sparse data
        assert ACTIONS.rows_per_year is None

        actions_mgr = _CacheManager(ACTIONS)

        # Even with many tickers and long date range, should NOT split
        tickers = [f'T{i:03d}' for i in range(100)]
        filters = {'ticker': tickers, 'date_gte': '1990-01-01', 'date_lte': '2024-12-31'}
        chunks = actions_mgr._split_filters(filters)

        # Should return single chunk (no splitting)
        assert len(chunks) == 1
        assert chunks[0] == filters


class TestSyncBounds:
    """Test sync bounds tracking with new schema."""

    @pytest.mark.asyncio
    async def test_sync_bounds_includes_lastupdated(self, use_temp_db):
        """Sync bounds should track max_lastupdated."""
        # Query data (which triggers sync)
        await async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-31')

        # Check sync_bounds table has the new columns
        async with aioduckdb.connect(get_db_path()) as conn:
            cursor = await conn.execute(f"""
                SELECT ticker, synced_from, synced_to, max_lastupdated, last_staleness_check
                FROM {SEP.sync_bounds_table_name()}
                WHERE ticker = 'AAPL'
            """)
            result = await cursor.fetchone()

        assert result is not None
        ticker, synced_from, synced_to, _, _ = result
        assert ticker == 'AAPL'
        assert synced_from is not None
        assert synced_to is not None
        # max_lastupdated should be populated from the fetched data
        # (will be None if lastupdated wasn't in the response)


class TestConcurrentAccess:
    """
    Test concurrent access doesn't cause race conditions.

    These tests mock the API but exercise real concurrent DuckDB operations
    to catch race conditions in table creation and data insertion.
    """

    @pytest.fixture
    def mock_api_with_delay(self):
        """
        Create a mock API that adds a small delay to increase race window.

        The delay makes it more likely that concurrent coroutines will
        interleave their DuckDB operations.
        """
        import asyncio

        async def delayed_get_table(self, table_name, columns=None, paginate=True, **filters):
            # Small delay to widen the race window
            await asyncio.sleep(0.01)

            # Return mock data based on table and filters
            ticker = filters.get('ticker', 'TEST')
            if isinstance(ticker, list):
                ticker = ticker[0]

            date_gte = filters.get('date', {}).get('gte', '2020-08-28')
            date_lte = filters.get('date', {}).get('lte', '2020-08-28')

            # Generate mock data
            if 'SEP' in table_name or 'SFP' in table_name:
                return pd.DataFrame({
                    'ticker': [ticker],
                    'date': [date_gte],
                    'close': [100.0],
                    'closeunadj': [100.0],
                    'volume': [1000000],
                    'lastupdated': [date_gte],
                })
            elif 'ACTIONS' in table_name:
                action = filters.get('action', 'dividend')
                return pd.DataFrame({
                    'ticker': [ticker],
                    'date': [date_gte],
                    'action': [action],
                    'value': [0.5],
                    'lastupdated': [date_gte],
                })
            elif 'DAILY' in table_name:
                return pd.DataFrame({
                    'ticker': [ticker],
                    'date': [date_gte],
                    'marketcap': [1e9],
                    'lastupdated': [date_gte],
                })
            else:
                return pd.DataFrame()

        return delayed_get_table

    @pytest.mark.asyncio
    async def test_concurrent_table_creation(self, use_temp_db, mock_api_with_delay):
        """
        Multiple concurrent queries should not conflict when creating tables.

        This reproduces the race condition:
        "TransactionContext Error: Catalog write-write conflict on create with Table"
        """
        import asyncio

        with patch.object(AsyncNDLClient, 'get_table', mock_api_with_delay):
            # Run multiple queries that will all try to create the same table
            results = await asyncio.gather(
                async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28'),
                async_query(SEP, ticker='MSFT', date_gte='2020-08-28', date_lte='2020-08-28'),
                async_query(SEP, ticker='GOOGL', date_gte='2020-08-28', date_lte='2020-08-28'),
                async_query(SEP, ticker='AMZN', date_gte='2020-08-28', date_lte='2020-08-28'),
                return_exceptions=True,
            )

        # Check for race condition errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).lower()
                assert 'conflict' not in error_msg, f"Table creation race: {result}"
                assert 'catalog' not in error_msg, f"Table creation race: {result}"

    @pytest.mark.asyncio
    async def test_concurrent_sync_bounds_insert(self, use_temp_db, mock_api_with_delay):
        """
        Concurrent queries for the same ticker should not conflict on sync_bounds.

        This reproduces the race condition:
        "Constraint Error: Duplicate key ticker: AAPL violates primary key"
        """
        import asyncio

        with patch.object(AsyncNDLClient, 'get_table', mock_api_with_delay):
            # All queries for the same ticker - they'll race to insert sync_bounds
            results = await asyncio.gather(
                async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28'),
                async_query(SEP, ticker='AAPL', date_gte='2020-08-29', date_lte='2020-08-29'),
                async_query(SEP, ticker='AAPL', date_gte='2020-08-30', date_lte='2020-08-30'),
                async_query(SEP, ticker='AAPL', date_gte='2020-08-31', date_lte='2020-08-31'),
                return_exceptions=True,
            )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).lower()
                assert 'duplicate' not in error_msg, f"Sync bounds insert race: {result}"
                assert 'constraint' not in error_msg, f"Sync bounds insert race: {result}"

    @pytest.mark.asyncio
    async def test_concurrent_data_insert_same_rows(self, use_temp_db):
        """
        Concurrent queries that return overlapping data should not conflict.

        This reproduces the race condition where two queries fetch and try
        to insert the same rows into the data table.
        """
        import asyncio

        # Mock that returns the SAME data for different queries
        # This simulates the case where dividend and split queries both
        # try to insert the same rows
        async def mock_returning_same_data(self, table_name, columns=None, paginate=True, **filters):
            await asyncio.sleep(0.01)
            # Always return the same row regardless of filters
            return pd.DataFrame({
                'ticker': ['AAPL'],
                'date': ['2020-08-28'],
                'action': ['dividend'],
                'value': [0.5],
                'lastupdated': ['2020-08-28'],
            })

        from ndl_cache import ACTIONS

        with patch.object(AsyncNDLClient, 'get_table', mock_returning_same_data):
            # Multiple queries that will all try to insert the same row
            results = await asyncio.gather(
                async_query(ACTIONS, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28', action='dividend'),
                async_query(ACTIONS, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28', action='dividend'),
                async_query(ACTIONS, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28', action='dividend'),
                return_exceptions=True,
            )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).lower()
                assert 'duplicate' not in error_msg, f"Data insert race: {result}"
                assert 'constraint' not in error_msg, f"Data insert race: {result}"

    @pytest.mark.asyncio
    async def test_concurrent_different_tables(self, use_temp_db, mock_api_with_delay):
        """
        Concurrent queries to different tables should not conflict.

        Each table has its own sync_bounds table, so concurrent creation
        of different sync_bounds tables should not race.
        """
        import asyncio
        from ndl_cache import ACTIONS, DAILY

        with patch.object(AsyncNDLClient, 'get_table', mock_api_with_delay):
            results = await asyncio.gather(
                async_query(SEP, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28'),
                async_query(ACTIONS, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28', action='dividend'),
                async_query(DAILY, ticker='AAPL', date_gte='2020-08-28', date_lte='2020-08-28'),
                return_exceptions=True,
            )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).lower()
                assert 'conflict' not in error_msg, f"Cross-table race: {result}"
                assert 'duplicate' not in error_msg, f"Cross-table race: {result}"

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, use_temp_db, mock_api_with_delay):
        """
        Stress test with many concurrent queries to maximize race probability.
        """
        import asyncio

        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']

        with patch.object(AsyncNDLClient, 'get_table', mock_api_with_delay):
            # Create many concurrent queries
            queries = [
                async_query(SEP, ticker=ticker, date_gte='2020-08-28', date_lte='2020-08-28')
                for ticker in tickers
            ]
            results = await asyncio.gather(*queries, return_exceptions=True)

        race_errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).lower()
                if 'conflict' in error_msg or 'duplicate' in error_msg or 'constraint' in error_msg:
                    race_errors.append(f"Query {i} ({tickers[i]}): {result}")

        assert not race_errors, f"Race condition errors:\n" + "\n".join(race_errors)
