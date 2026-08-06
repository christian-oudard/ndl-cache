"""
Async client for Nasdaq Data Link API.

Provides async HTTP requests using aiohttp, with retry logic and error handling.
This is a focused implementation for datatables only (not full nasdaqdatalink replacement).
"""
import asyncio
import os
import warnings
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from ndl_cache.rate_limit import RateLimiter, parse_retry_after


class NDLError(Exception):
    """Base exception for Nasdaq Data Link errors."""

    def __init__(
        self,
        message: str,
        http_status: int | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.code = code


class AuthenticationError(NDLError):
    """API key is missing or invalid."""
    pass


class RateLimitError(NDLError):
    """Rate limit exceeded."""
    pass


class NotFoundError(NDLError):
    """Requested resource not found."""
    pass


def _get_api_key() -> str | None:
    """Get API key from environment or config file."""
    # Check environment variable
    env_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if env_key:
        return env_key

    # Check config file
    config_file = Path.home() / ".nasdaq" / "data_link_apikey"
    if config_file.exists():
        key = config_file.read_text().strip()
        if key:
            return key

    return None


def _is_rate_limit(message: str) -> bool:
    """
    Whether an error body describes a rate limit rather than a bad credential.

    Nasdaq reports the speed limit with more than one HTTP status, and the
    403 form is indistinguishable from a rejected key by status alone.
    """
    lowered = message.lower()
    return 'speed limit' in lowered or 'rate limit' in lowered or (
        'exceeded' in lowered and 'limit' in lowered)


def _is_authentication(message: str) -> bool:
    """Whether a 401/403 body is really about the credential."""
    lowered = message.lower()
    return 'api key' in lowered or 'unauthor' in lowered or 'subscription' in lowered


def _raise_for_error(status: int, data: dict | None = None,
                     body: str | None = None):
    """
    Raise the appropriate exception for an HTTP status and response.

    ``body`` is the raw response text, used when the server answers with
    something other than the usual JSON error envelope. A 414 comes back as an
    HTML error page, and without this the only report was "API request failed"
    with no status, which says nothing about what to fix.
    """
    message = None
    code = None

    if data and "quandl_error" in data:
        error_info = data["quandl_error"]
        message = error_info.get("message")
        code = error_info.get("code")

    if not message:
        detail = (body or '').strip()
        if len(detail) > 200:
            detail = detail[:200] + '...'
        message = f'HTTP {status}'
        if status == 414:
            message += (' URI too long: the request URL exceeded the server '
                        'limit, so the query needs splitting into more chunks')
        if detail:
            message += f': {detail}'

    if status == 429 or _is_rate_limit(message):
        raise RateLimitError(message, status, code)
    elif status == 404:
        raise NotFoundError(message, status, code)
    elif (status == 401 or status == 403) and _is_authentication(message):
        raise AuthenticationError(message, status, code)
    else:
        # Nasdaq answers some request errors with 403 even when the credential
        # is fine, notably a query naming a column the table no longer has.
        # Reporting those as an auth failure sends the reader off checking keys
        # and proxies for what is really a stale schema in this package.
        raise NDLError(message, status, code)


class AsyncNDLClient:
    """
    Async client for Nasdaq Data Link (NDL) API.

    Provides async HTTP requests with retry logic for rate limits and transient errors.

    Usage:
        async with AsyncNDLClient() as client:
            df = await client.get_table("SHARADAR/SEP", ticker="AAPL")
    """

    BASE_URL = "https://data.nasdaq.com/api/v3/datatables"

    # Request settings
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.5
    PAGE_LIMIT = 100

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        rate_limiter: "RateLimiter | None" = None,
    ):
        """
        Initialize the async client.

        Args:
            api_key: API key (defaults to env var or config file)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts for failed requests
        """
        self.api_key = api_key or _get_api_key()
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        # Shared across this client's concurrent fetchers, so one rejection
        # stands the whole client down instead of only the worker that saw it.
        self.rate_limiter = rate_limiter or RateLimiter()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "accept": "application/json",
                "request-source": "ndl-cache",
            }
            if self.api_key:
                headers["x-api-token"] = self.api_key

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            # trust_env makes aiohttp honour HTTP_PROXY/HTTPS_PROXY/NO_PROXY
            # and .netrc, which requests does by default and aiohttp does not.
            # Without it, every call bypasses any configured proxy. Where a
            # proxy supplies the credential, the request then goes out with
            # whatever placeholder is on disk, and Nasdaq answers as if the
            # caller were anonymous: the anonymous pool is 20 calls per 10
            # minutes shared across all anonymous users, so it is permanently
            # exhausted and the reply is a rate-limit error. That error is
            # deeply misleading, since the rate is fine and the credential is
            # the problem.
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=timeout, trust_env=True)

        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _request(self, url: str, params: dict | None = None) -> dict:
        """Make an HTTP GET request with retry logic."""
        session = await self._get_session()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # Pace before every attempt, including retries, so a burst of
            # parallel fetchers cannot collectively outrun the limit.
            await self.rate_limiter.acquire()
            try:
                # The slot is held for the whole request, not just its start.
                # Nasdaq's concurrency limit counts calls that are open, so
                # releasing at send time would let several overlap.
                async with self.rate_limiter.in_flight():
                    async with session.get(url, params=params) as resp:
                        if resp.status >= 400:
                            body = None
                            try:
                                data = await resp.json()
                            except Exception:
                                data = None
                                try:
                                    body = await resp.text()
                                except Exception:
                                    body = None
                            try:
                                _raise_for_error(resp.status, data, body)
                            except RateLimitError:
                                # Honour Retry-After when sent, and stand the
                                # whole client down either way. Nasdaq suspends
                                # the account rather than throttling, so the old
                                # sub-second backoff just extended the ban.
                                self.rate_limiter.penalize(
                                    parse_retry_after(resp.headers))
                                raise

                        return await resp.json()

            except RateLimitError:
                if attempt < self.max_retries:
                    continue
                raise

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.RETRY_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise NDLError(f"Request failed: {e}") from e

        raise NDLError(f"Request failed after {self.max_retries} retries") from last_error

    async def get_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        paginate: bool = True,
        **filters,
    ) -> pd.DataFrame:
        """
        Fetch data from an NDL datatable.

        Args:
            table_name: Table identifier (e.g., "SHARADAR/SEP")
            columns: List of columns to fetch (None for all)
            paginate: Whether to follow pagination cursors
            **filters: Query filters like ticker="AAPL", date={"gte": "2020-01-01"}

        Returns:
            DataFrame with the results
        """
        url = f"{self.BASE_URL}/{table_name}.json"
        params = self._build_params(columns, filters)

        all_data: list[list] = []
        all_columns: list[str] | None = None
        page_count = 0

        while True:
            data = await self._request(url, params=params)

            datatable = data.get("datatable", {})
            rows = datatable.get("data", [])
            columns_meta = datatable.get("columns", [])

            if all_columns is None:
                all_columns = [c["name"] for c in columns_meta]

            all_data.extend(rows)

            # Check for next page
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor_id")

            if not paginate or next_cursor is None:
                break

            page_count += 1
            if page_count >= self.PAGE_LIMIT:
                warnings.warn(
                    f"Reached page limit ({self.PAGE_LIMIT}). "
                    "Some data may be missing. Consider narrowing your query.",
                    UserWarning,
                )
                break

            params["qopts.cursor_id"] = next_cursor

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data, columns=all_columns)

    def _build_params(
        self,
        columns: list[str] | None,
        filters: dict[str, Any],
    ) -> dict[str, str]:
        """Build query parameters from columns and filters."""
        params: dict[str, str] = {}

        if columns:
            params["qopts.columns"] = ",".join(columns)

        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filters like date={"gte": "2020-01-01", "lte": "2020-12-31"}
                for op, val in value.items():
                    params[f"{key}.{op}"] = str(val)
            elif isinstance(value, (list, tuple)):
                # List filters like ticker=["AAPL", "MSFT"]
                params[key] = ",".join(str(v) for v in value)
            else:
                params[key] = str(value)

        return params


async def gather_tables(
    *requests: tuple[str, dict],
    client: AsyncNDLClient | None = None,
) -> list[pd.DataFrame]:
    """
    Fetch multiple tables concurrently.

    Args:
        *requests: Tuples of (table_name, filters_dict)
        client: Optional client to reuse (creates temporary if None)

    Returns:
        List of DataFrames in the same order as requests

    Example:
        results = await gather_tables(
            ("SHARADAR/SEP", {"ticker": "AAPL", "date": {"gte": "2020-01-01"}}),
            ("SHARADAR/ACTIONS", {"ticker": "AAPL", "action": "dividend"}),
        )
    """
    if client is None:
        async with AsyncNDLClient() as temp_client:
            return await gather_tables(*requests, client=temp_client)

    async def fetch_one(table: str, opts: dict) -> pd.DataFrame:
        columns = opts.pop("columns", None)
        paginate = opts.pop("paginate", True)
        return await client.get_table(table, columns=columns, paginate=paginate, **opts)

    tasks = [fetch_one(table, dict(opts)) for table, opts in requests]
    return list(await asyncio.gather(*tasks))
