"""
Tests that the client honours the limiter, and that a rejection produces a
long stand-down rather than an immediate retry.

Also covers the error classification, since a misclassified 403 is what makes
a stale column look like a credential problem.
"""
import pytest

from ndl_cache.async_client import (
    AsyncNDLClient, AuthenticationError, NDLError, RateLimitError,
    _is_rate_limit, _raise_for_error,
)
from ndl_cache.rate_limit import RateLimiter, Window


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def time(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    """Returns queued responses and records how many calls were made."""

    closed = False

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(200, {'datatable': {'data': []}})


RATE_LIMIT_BODY = {
    'quandl_error': {
        'code': 'QELx01',
        'message': ('You have exceeded the API speed limit and your account '
                    'has temporarily been disabled.'),
    }
}


class TestErrorClassification:

    def test_speed_limit_message_is_a_rate_limit(self):
        assert _is_rate_limit(
            'You have exceeded the API speed limit and your account has '
            'temporarily been disabled.')

    def test_plain_auth_message_is_not(self):
        assert not _is_rate_limit(
            'A valid API key is required to retrieve data.')

    def test_speed_limit_on_403_is_not_mistaken_for_bad_credentials(self):
        # Status alone cannot tell these apart.
        with pytest.raises(RateLimitError):
            _raise_for_error(403, RATE_LIMIT_BODY)

    def test_genuine_403_still_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            _raise_for_error(403, {'quandl_error': {
                'code': 'QEPx04',
                'message': 'A valid API key is required to retrieve data.'}})

    def test_missing_column_is_not_an_authentication_error(self):
        # Nasdaq answers this with 403. Reporting it as an auth failure sends
        # the reader off checking keys and proxies for a stale schema here.
        with pytest.raises(NDLError) as caught:
            _raise_for_error(403, {'quandl_error': {
                'code': 'QEPx06',
                'message': '["famasector"] column does not exist.'}})
        assert not isinstance(caught.value, AuthenticationError)


class TestClientPacing:

    async def _client(self, responses, clock, windows):
        limiter = RateLimiter(windows=windows, clock=clock.time,
                              sleep=clock.sleep, penalty_seconds=60.0)
        client = AsyncNDLClient(api_key='x', rate_limiter=limiter)
        session = FakeSession(responses)
        client._session = session
        return client, session

    async def test_requests_are_paced_by_the_limiter(self):
        clock = FakeClock()
        client, session = await self._client(
            [], clock, [Window(calls=2, seconds=10)])
        for _ in range(4):
            await client._request('http://example/x')
        assert session.calls == 4
        # Two calls fit immediately; the third waits out the window, which
        # then frees both slots at once, so the fourth is free again.
        assert clock.slept == [pytest.approx(10.0)]
        # Four calls in a 2-per-10s window cannot take less than 10 seconds.
        assert clock.now - 1000.0 >= 10.0

    async def test_rate_limit_triggers_a_long_stand_down(self):
        clock = FakeClock()
        client, session = await self._client(
            [FakeResponse(429, RATE_LIMIT_BODY)], clock,
            [Window(calls=100, seconds=10)])
        await client._request('http://example/x')
        assert session.calls == 2               # one rejection, then a retry
        assert clock.slept[-1] == pytest.approx(60.0)

    async def test_retry_after_header_is_honoured(self):
        clock = FakeClock()
        client, _ = await self._client(
            [FakeResponse(429, RATE_LIMIT_BODY, {'Retry-After': '300'})],
            clock, [Window(calls=100, seconds=10)])
        await client._request('http://example/x')
        assert clock.slept[-1] == pytest.approx(300.0)

    async def test_persistent_rate_limit_eventually_raises(self):
        clock = FakeClock()
        client, session = await self._client(
            [FakeResponse(429, RATE_LIMIT_BODY) for _ in range(10)],
            clock, [Window(calls=100, seconds=10)])
        with pytest.raises(RateLimitError):
            await client._request('http://example/x')
        # Bounded by max_retries rather than hammering indefinitely.
        assert session.calls == client.max_retries + 1


class TestProxySupport:

    async def test_session_trusts_the_environment(self):
        # aiohttp ignores HTTP_PROXY/HTTPS_PROXY unless trust_env is set,
        # while requests honours them by default. Without this, calls bypass
        # any configured proxy; where a proxy supplies the credential, the
        # request goes out with whatever placeholder is on disk and Nasdaq
        # answers as an anonymous caller, whose pool is permanently exhausted.
        client = AsyncNDLClient(api_key='x')
        session = await client._get_session()
        try:
            assert session._trust_env is True
        finally:
            await client.close()


class TestErrorDetail:
    """
    An error the caller cannot act on is nearly as bad as no error. A 414
    comes back as an HTML page rather than the usual JSON envelope, and used
    to surface as "API request failed" with no status at all.
    """

    def test_non_json_body_still_reports_the_status(self):
        with pytest.raises(NDLError) as caught:
            _raise_for_error(414, None, '<html>Request-URI Too Long</html>')
        assert '414' in str(caught.value)
        assert caught.value.http_status == 414

    def test_414_explains_what_to_do(self):
        with pytest.raises(NDLError) as caught:
            _raise_for_error(414, None, 'Request-URI Too Long')
        assert 'URI too long' in str(caught.value)

    def test_long_bodies_are_truncated(self):
        with pytest.raises(NDLError) as caught:
            _raise_for_error(500, None, 'x' * 5000)
        assert len(str(caught.value)) < 400

    def test_json_error_message_still_preferred(self):
        with pytest.raises(NDLError) as caught:
            _raise_for_error(400, {'quandl_error': {
                'code': 'QEMx01', 'message': 'Something specific'}}, 'ignored')
        assert 'Something specific' in str(caught.value)


class TestServerErrorRetry:
    """
    Nasdaq intermittently answers a valid request with a 5xx and a generic
    "unexpected error in the system". Treating that as fatal loses the rest
    of a long sync, which is how a cache warm-up died partway through.
    """

    SERVER_ERROR = {
        'quandl_error': {
            'code': 'QEMx01',
            'message': ("We're sorry, but there's been an unexpected error "
                        "in the system."),
        }
    }

    BAD_REQUEST = {
        'quandl_error': {
            'code': 'QEPx06',
            'message': '["famasector"] column does not exist.',
        }
    }

    async def _client(self, responses, clock):
        limiter = RateLimiter(windows=[Window(calls=100, seconds=10)],
                              clock=clock.time, sleep=clock.sleep)
        client = AsyncNDLClient(api_key='x', rate_limiter=limiter)
        client._session = FakeSession(responses)
        return client, client._session

    async def test_a_transient_server_error_is_retried(self):
        clock = FakeClock()
        client, session = await self._client(
            [FakeResponse(500, self.SERVER_ERROR)], clock)
        await client._request('http://example/x')
        assert session.calls == 2

    async def test_a_persistent_server_error_eventually_raises(self):
        clock = FakeClock()
        client, session = await self._client(
            [FakeResponse(500, self.SERVER_ERROR) for _ in range(10)], clock)
        with pytest.raises(NDLError):
            await client._request('http://example/x')
        assert session.calls == client.max_retries + 1

    async def test_a_bad_request_is_not_retried(self):
        # A malformed query fails the same way however often it is sent, so
        # retrying only multiplies the load and delays the real error.
        clock = FakeClock()
        client, session = await self._client(
            [FakeResponse(403, self.BAD_REQUEST) for _ in range(10)], clock)
        with pytest.raises(NDLError):
            await client._request('http://example/x')
        assert session.calls == 1

    async def test_retries_back_off(self):
        clock = FakeClock()
        client, _ = await self._client(
            [FakeResponse(503, self.SERVER_ERROR) for _ in range(2)], clock)
        await client._request('http://example/x')
        assert clock.slept == [pytest.approx(client.RETRY_BACKOFF),
                               pytest.approx(client.RETRY_BACKOFF * 2)]


class TestConnectionFailureRetry:
    """
    Two multi-hour runs died on network faults: one on a dropped connection
    whose retries all failed within two seconds, and one on a request timeout
    that was never retried at all because asyncio.TimeoutError is not an
    aiohttp error.
    """

    class FailingSession(FakeSession):
        """Raises a given exception for the first n calls."""

        def __init__(self, error, failures):
            super().__init__([])
            self._error = error
            self._failures = failures
            self.closed = False

        def get(self, url, params=None):
            self.calls += 1
            if self.calls <= self._failures:
                raise self._error
            return FakeResponse(200, {'datatable': {'data': []}})

        async def close(self):
            self.closed = True

    async def _client(self, session, clock):
        limiter = RateLimiter(windows=[Window(calls=100, seconds=10)],
                              clock=clock.time, sleep=clock.sleep)
        client = AsyncNDLClient(api_key='x', rate_limiter=limiter)
        client._session = session
        return client

    async def test_a_request_timeout_is_retried(self):
        import asyncio
        clock = FakeClock()
        session = self.FailingSession(asyncio.TimeoutError(), 1)
        client = await self._client(session, clock)
        # A fresh session is created after the failure, so keep serving the
        # same fake rather than letting aiohttp build a real one.
        client._get_session = lambda: _identity(session)
        await client._request('http://example/x')
        assert session.calls == 2

    async def test_a_dropped_connection_is_retried(self):
        import aiohttp
        clock = FakeClock()
        session = self.FailingSession(
            aiohttp.ServerDisconnectedError(), 2)
        client = await self._client(session, clock)
        client._get_session = lambda: _identity(session)
        await client._request('http://example/x')
        assert session.calls == 3

    async def test_the_stale_session_is_discarded(self):
        # Reusing a pool holding a dead socket fails identically every time,
        # which spends all the retries without ever trying a live connection.
        import aiohttp
        clock = FakeClock()
        session = self.FailingSession(aiohttp.ServerDisconnectedError(), 1)
        client = await self._client(session, clock)
        client._get_session = lambda: _identity(session)
        await client._request('http://example/x')
        assert session.closed

    async def test_connection_retries_wait_longer_than_server_errors(self):
        import aiohttp
        clock = FakeClock()
        session = self.FailingSession(aiohttp.ServerDisconnectedError(), 1)
        client = await self._client(session, clock)
        client._get_session = lambda: _identity(session)
        await client._request('http://example/x')
        assert clock.slept == [pytest.approx(client.CONNECTION_BACKOFF)]
        assert client.CONNECTION_BACKOFF > client.RETRY_BACKOFF


async def _identity(value):
    return value
