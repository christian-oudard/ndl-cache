"""
Several processes sharing one cache file, each free to read and to sync.

A backtest runs its arms in separate processes, and several backtests run at
once against the same cache.
"""
import multiprocessing
import time
from unittest.mock import patch

import pandas as pd

from ndl_cache import DAILY, query
from ndl_cache.async_client import AsyncNDLClient
from ndl_cache.testing import temp_db

FETCH_SECONDS = 3.0
DAYS = len(pd.bdate_range('2024-01-02', '2024-01-31'))


def rows(ticker, start, end):
    return pd.DataFrame([
        {'ticker': ticker, 'date': d.date(), 'marketcap': 1.0,
         'lastupdated': pd.Timestamp('2024-05-01').date()}
        for d in pd.bdate_range(start, end)])


async def slow_fetch(client, table_name, columns=None, paginate=True, **filters):
    """A provider call that takes as long as a real one can."""
    import asyncio
    await asyncio.sleep(FETCH_SECONDS)
    tickers = filters['ticker']
    tickers = tickers if isinstance(tickers, list) else [tickers]
    date = filters.get('date', {})
    return pd.concat([rows(t, date.get('gte', '2024-01-02'),
                           date.get('lte', '2024-01-31')) for t in tickers])


def sync(db_path, ticker, start, end, results):
    """In a child process: query through the slow provider."""
    with temp_db(db_path), patch.object(AsyncNDLClient, 'get_table',
                                        slow_fetch):
        started = time.monotonic()
        try:
            df = query(DAILY, ticker=ticker, date_gte=start, date_lte=end)
            results.put((ticker, len(df), time.monotonic() - started, None))
        except Exception as e:  # reported to the parent, which asserts
            results.put((ticker, 0, time.monotonic() - started, repr(e)))


def run_children(db_path, jobs):
    context = multiprocessing.get_context('spawn')
    results = context.Queue()
    children = [context.Process(target=sync, args=(db_path, *job, results))
                for job in jobs]
    for child in children:
        child.start()
    outcomes = [results.get(timeout=60) for _ in children]
    for child in children:
        child.join(timeout=60)
    return {ticker: (n, seconds, error) for ticker, n, seconds, error
            in outcomes}


def fill(db_path, ticker):
    async def fast(client, table_name, columns=None, paginate=True, **f):
        return rows(ticker, '2024-01-02', '2024-01-31')
    with temp_db(db_path), patch.object(AsyncNDLClient, 'get_table', fast):
        query(DAILY, ticker=ticker, date_gte='2024-01-02',
              date_lte='2024-01-31')


def test_a_reader_is_not_refused_while_another_process_syncs(tmp_path):
    db_path = str(tmp_path / 'cache.duckdb')
    fill(db_path, 'MSFT')
    context = multiprocessing.get_context('spawn')
    results = context.Queue()
    writer = context.Process(
        target=sync,
        args=(db_path, 'AAPL', '2024-01-02', '2024-01-31', results))
    writer.start()
    # The writer is inside its provider call.
    time.sleep(FETCH_SECONDS / 3)

    started = time.monotonic()
    with temp_db(db_path):
        df = query(DAILY, ticker='MSFT', date_gte='2024-01-02',
                   date_lte='2024-01-31')
    waited = time.monotonic() - started

    ticker, n, _, error = results.get(timeout=60)
    writer.join(timeout=60)
    assert error is None, error
    assert len(df) == DAYS
    # Reading cached data does not wait out another process's network call.
    assert waited < FETCH_SECONDS / 2, waited


def test_processes_syncing_at_once_all_succeed_in_parallel(tmp_path):
    db_path = str(tmp_path / 'cache.duckdb')
    jobs = [(t, '2024-01-02', '2024-01-31') for t in ('AAPL', 'MSFT', 'IBM')]

    started = time.monotonic()
    outcomes = run_children(db_path, jobs)
    elapsed = time.monotonic() - started

    for ticker, (n, _, error) in outcomes.items():
        assert error is None, (ticker, error)
        assert n == DAYS, (ticker, n)
    # The provider calls overlap rather than queueing behind one another.
    assert elapsed < 2 * FETCH_SECONDS + 5, elapsed

    with temp_db(db_path):
        for ticker in ('AAPL', 'MSFT', 'IBM'):
            assert len(query(DAILY, ticker=ticker, date_gte='2024-01-02',
                             date_lte='2024-01-31')) == DAYS


def test_processes_syncing_the_same_range_leave_it_whole(tmp_path):
    db_path = str(tmp_path / 'cache.duckdb')
    jobs = [('AAPL', '2024-01-02', '2024-01-31')] * 1
    outcomes = run_children(db_path, jobs * 3)
    for _, (n, _, error) in outcomes.items():
        assert error is None, error

    calls = []

    async def counted(client, table_name, columns=None, paginate=True, **f):
        calls.append(f)
        return pd.DataFrame()
    with temp_db(db_path), patch.object(AsyncNDLClient, 'get_table', counted):
        df = query(DAILY, ticker='AAPL', date_gte='2024-01-02',
                   date_lte='2024-01-31')
    assert len(df) == DAYS
    assert calls == []
