"""
Database migrations for ndl-cache.

These migrations fix data issues in existing caches.
"""
import logging

import duckdb

from .async_cache import get_db_path
from .tables import SEP, SFP, ACTIONS, SF1, DAILY, TICKERS

log = logging.getLogger(__name__)


def remove_duplicate_rows(db_path: str = None) -> dict[str, int]:
    """
    Remove duplicate rows from all tables in the cache database.

    This fixes the bug where parallel queries or retries could insert
    duplicate rows, causing constraint errors on subsequent queries.

    Args:
        db_path: Path to the DuckDB database. If None, uses the default path.

    Returns:
        dict mapping table name to number of duplicates removed
    """
    if db_path is None:
        db_path = get_db_path()

    log.info(f"Checking for duplicate rows in {db_path}")

    conn = duckdb.connect(db_path)
    results = {}

    # All tables and their primary key columns
    tables = [
        (SEP, ['ticker', 'date']),
        (SFP, ['ticker', 'date']),
        (ACTIONS, ['ticker', 'date', 'action']),
        (SF1, ['ticker', 'dimension', 'calendardate', 'datekey']),
        (DAILY, ['ticker', 'date']),
        (TICKERS, ['ticker']),
    ]

    for table_def, pk_cols in tables:
        table_name = table_def.safe_table_name()

        # Check if table exists
        check_query = f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{table_name.strip('"')}'
        """
        if conn.execute(check_query).fetchone()[0] == 0:
            log.debug(f"Table {table_name} does not exist, skipping")
            continue

        # Count duplicates
        pk_cols_str = ', '.join(pk_cols)
        count_query = f"""
            SELECT COUNT(*) - COUNT(DISTINCT ({pk_cols_str}))
            FROM {table_name}
        """
        try:
            duplicate_count = conn.execute(count_query).fetchone()[0]
        except duckdb.CatalogException:
            log.debug(f"Table {table_name} does not exist, skipping")
            continue

        if duplicate_count == 0:
            log.debug(f"No duplicates in {table_name}")
            results[table_name] = 0
            continue

        log.info(f"Found {duplicate_count} duplicate rows in {table_name}")

        # Remove duplicates by keeping only one row per primary key
        # DuckDB doesn't have DELETE with rowid easily, so we recreate the table
        # First, get all columns
        cols_query = f"SELECT * FROM {table_name} LIMIT 0"
        col_names = [desc[0] for desc in conn.execute(cols_query).description]
        col_names_str = ', '.join(col_names)

        # Create temp table with deduplicated data
        dedup_query = f"""
            CREATE TEMPORARY TABLE {table_name}_dedup AS
            SELECT DISTINCT ON ({pk_cols_str}) {col_names_str}
            FROM {table_name}
        """
        conn.execute(dedup_query)

        # Delete all from original and reinsert
        conn.execute(f"DELETE FROM {table_name}")
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_dedup")
        conn.execute(f"DROP TABLE {table_name}_dedup")

        # Verify
        new_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        log.info(f"Removed {duplicate_count} duplicates from {table_name}, {new_count} rows remaining")

        results[table_name] = duplicate_count

    conn.close()
    return results


def migrate(db_path: str = None) -> dict[str, int]:
    """
    Run all migrations on the cache database.

    This is safe to run multiple times (idempotent).

    Args:
        db_path: Path to the DuckDB database. If None, uses the default path.

    Returns:
        dict with migration results
    """
    results = {}

    # Migration 1: Remove duplicate rows
    duplicates_removed = remove_duplicate_rows(db_path)
    results['duplicates_removed'] = duplicates_removed

    return results


if __name__ == '__main__':
    # Run migrations on default database
    import sys

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = migrate(db_path)

    total_duplicates = sum(results['duplicates_removed'].values())
    if total_duplicates > 0:
        print(f"\nRemoved {total_duplicates} total duplicate rows")
    else:
        print("\nNo duplicates found")
