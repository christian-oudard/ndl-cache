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


def _raise_for_error(status: int, data: dict | None = None):
    """Raise appropriate exception based on HTTP status and response."""
    message = "API request failed"
    code = None

    if data and "quandl_error" in data:
        error_info = data["quandl_error"]
        message = error_info.get("message", message)
        code = error_info.get("code")

    if status == 401 or status == 403:
        raise AuthenticationError(message, status, code)
    elif status == 404:
        raise NotFoundError(message, status, code)
    elif status == 429:
        raise RateLimitError(message, status, code)
    else:
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
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)

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
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status >= 400:
                        try:
                            data = await resp.json()
                        except Exception:
                            data = None
                        _raise_for_error(resp.status, data)

                    return await resp.json()

            except RateLimitError:
                if attempt < self.max_retries:
                    wait_time = self.RETRY_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait_time)
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
