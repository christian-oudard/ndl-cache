"""Testing utilities for ndl-cache."""

import os
import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def temp_db(path: str = None):
    """Context manager for using a temporary database path.

    If path is None, creates a temporary directory with a test database.
    Useful for testing.

    Usage:
        with temp_db('/path/to/test.duckdb'):
            table = SEPTable()
            ...

    Or with auto-cleanup:
        with temp_db() as db_path:
            table = SEPTable()
            ...
    """
    old_env = os.environ.get('NDL_CACHE_DB_PATH')

    if path is None:
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'test.duckdb')
        cleanup_dir = tmpdir
    else:
        cleanup_dir = None

    os.environ['NDL_CACHE_DB_PATH'] = path
    try:
        yield path
    finally:
        if old_env is None:
            del os.environ['NDL_CACHE_DB_PATH']
        else:
            os.environ['NDL_CACHE_DB_PATH'] = old_env

        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
