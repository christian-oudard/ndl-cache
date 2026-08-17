"""
Shared test setup.

The rate limiter keeps its call history in a file shared by every process
using the same credential, so tests have to be pointed somewhere disposable.
Otherwise a test run would pace itself against the developer's real history,
and worse, record calls it never made into it.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated_rate_limit_state(tmp_path, monkeypatch):
    monkeypatch.setenv('NDL_RATE_LIMIT_STATE',
                       str(tmp_path / 'rate_limit.sqlite'))
