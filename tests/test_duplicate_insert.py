"""
Test that demonstrates the duplicate row insertion bug.

When the same data is fetched twice (e.g., parallel queries, retries, overlapping
date ranges), the INSERT statement fails because it doesn't handle conflicts.
"""
import pandas as pd
import pytest

from ndl_cache import SEP
from ndl_cache.async_cache import _CacheManager
from ndl_cache.testing import temp_db


@pytest.mark.asyncio
async def test_duplicate_insert_causes_constraint_error():
    """
    Test that inserting the same data twice causes a constraint violation.

    This demonstrates the bug: when parallel queries or retries fetch overlapping
    data, the second INSERT fails because there's no ON CONFLICT handling.
    """
    with temp_db() as db_path:
        mgr = _CacheManager(SEP)

        # Sample data that might be returned by the API
        sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'date': ['2024-01-02', '2024-01-03', '2024-01-02', '2024-01-03'],
            'close': [185.0, 186.0, 375.0, 376.0],
            'volume': [1000000, 1100000, 800000, 850000],
        })

        # First insert should succeed
        conn = await mgr._get_conn()
        await mgr._ensure_data_table(['close', 'volume'])

        cols = ['ticker', 'date', 'close', 'volume']
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        rows = [tuple(row) for row in sample_data.itertuples(index=False, name=None)]

        await conn.executemany(f"""
            INSERT INTO {SEP.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, rows)

        # Verify data was inserted
        result = await conn.execute(f"SELECT COUNT(*) FROM {SEP.safe_table_name()}")
        count = (await result.fetchone())[0]
        assert count == 4, f"Expected 4 rows, got {count}"

        # Second insert of the SAME data should fail with constraint error
        # This simulates what happens when:
        # - Parallel queries fetch overlapping data
        # - A retry re-fetches data that was partially inserted
        # - Two queries request overlapping date ranges
        with pytest.raises(Exception) as exc_info:
            await conn.executemany(f"""
                INSERT INTO {SEP.safe_table_name()} ({col_names})
                VALUES ({placeholders})
            """, rows)

        # The error should be a constraint violation (duplicate primary key)
        assert "Duplicate key" in str(exc_info.value) or "UNIQUE constraint" in str(exc_info.value), \
            f"Expected constraint error, got: {exc_info.value}"


@pytest.mark.asyncio
async def test_insert_or_replace_handles_duplicates():
    """
    Test that INSERT OR REPLACE correctly handles duplicate rows.

    This is the fix: using INSERT OR REPLACE instead of plain INSERT.
    """
    with temp_db() as db_path:
        mgr = _CacheManager(SEP)

        sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL'],
            'date': ['2024-01-02', '2024-01-03'],
            'close': [185.0, 186.0],
            'volume': [1000000, 1100000],
        })

        conn = await mgr._get_conn()
        await mgr._ensure_data_table(['close', 'volume'])

        cols = ['ticker', 'date', 'close', 'volume']
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        rows = [tuple(row) for row in sample_data.itertuples(index=False, name=None)]

        # First insert
        await conn.executemany(f"""
            INSERT OR REPLACE INTO {SEP.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, rows)

        # Second insert of same data - should succeed with INSERT OR REPLACE
        await conn.executemany(f"""
            INSERT OR REPLACE INTO {SEP.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, rows)

        # Should still have only 2 rows (not 4)
        result = await conn.execute(f"SELECT COUNT(*) FROM {SEP.safe_table_name()}")
        count = (await result.fetchone())[0]
        assert count == 2, f"Expected 2 rows after INSERT OR REPLACE, got {count}"


@pytest.mark.asyncio
async def test_insert_with_updated_values():
    """
    Test that INSERT OR REPLACE updates values when re-inserting.

    This ensures that if the API returns updated data (e.g., corrected prices),
    the cache is updated rather than failing.
    """
    with temp_db() as db_path:
        mgr = _CacheManager(SEP)

        conn = await mgr._get_conn()
        await mgr._ensure_data_table(['close', 'volume'])

        cols = ['ticker', 'date', 'close', 'volume']
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)

        # Insert original data
        original_rows = [('AAPL', '2024-01-02', 185.0, 1000000)]
        await conn.executemany(f"""
            INSERT OR REPLACE INTO {SEP.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, original_rows)

        # Insert updated data (price corrected)
        updated_rows = [('AAPL', '2024-01-02', 185.50, 1000000)]  # price changed
        await conn.executemany(f"""
            INSERT OR REPLACE INTO {SEP.safe_table_name()} ({col_names})
            VALUES ({placeholders})
        """, updated_rows)

        # Verify the updated value is stored
        result = await conn.execute(
            f"SELECT close FROM {SEP.safe_table_name()} WHERE ticker='AAPL' AND date='2024-01-02'"
        )
        close_price = (await result.fetchone())[0]
        assert close_price == 185.50, f"Expected updated price 185.50, got {close_price}"
