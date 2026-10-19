"""All HTTP exchanges are scripted; no authenticated API is contacted."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import format_datetime
import io
import json
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import BaseHandler, HTTPRedirectHandler, Request
from urllib.response import addinfourl
from unittest.mock import patch

import pytest

from trading_agent.data import DataValidationError, MarketDataProvider
from trading_agent.data.calendar import KOLKATA, SessionCalendar
from trading_agent.data.persistence import load_snapshot, write_snapshot
from trading_agent.data.smartapi import (
    BASE_URL, CANDLE_PATH, QUOTE_PATH, HTTPResponse, RetryPolicy,
    SmartAPIAuthenticationError, SmartAPIClient, SmartAPIHistoricalProvider,
    SmartAPIQuoteProvider, SmartAPIRateLimitError, SmartAPIResponseError,
    SmartAPIUnavailableError, StaleQuoteError, UrllibTransport,
)

TOKEN = "mock-session-not-a-real-credential"
API_KEY = "mock-api-key-not-a-real-credential"
SYMBOL = "ABC-EQ"
INSTRUMENT = "123456"


def local(time="09:30:00", day="2026-01-05"):
    return datetime.fromisoformat(f"{day}T{time}").replace(tzinfo=KOLKATA)


class Clock:
    def __init__(self, now=None):
        self.start = now or local()
        self.elapsed = 0.0
        self.sleeps = []

    def now(self):
        return self.start + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.elapsed += seconds


class Transport:
    def __init__(self, clock, *script):
        self.clock, self.script = clock, list(script)
        self.requests, self.starts = [], []

    def __call__(self, request, *, timeout):
        assert request.full_url in {BASE_URL + CANDLE_PATH, BASE_URL + QUOTE_PATH}
        assert request.get_method() == "POST" and timeout == 7
        self.requests.append(request)
        self.starts.append(self.clock.elapsed)
        assert self.script, "unexpected extra transport request"
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item() if callable(item) else item


def ok(data):
    return HTTPResponse(200, {}, json.dumps({"status": True, "data": data}).encode())


def candle(time, *, day="2026-01-05", **changes):
    values = {"timestamp": local(time, day).isoformat(), "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000}
    values.update(changes)
    return list(values.values())


def quote(clock, *, feed=None, **changes):
    row = {"exchange": "NSE", "symbolToken": INSTRUMENT, "tradingSymbol": SYMBOL, "ltp": "100.1234", "exchFeedTime": (feed or clock.now()).isoformat(), "exchTradeTime": (feed or clock.now()).isoformat()}
    row.update(changes)
    return ok({"fetched": [row], "unfetched": []})


def client(clock, transport, **kwargs):
    return SmartAPIClient(token=TOKEN, api_key=API_KEY, symbol_tokens={SYMBOL: INSTRUMENT}, transport=transport, clock=clock.now, monotonic=clock.monotonic, sleep=clock.sleep, **kwargs)


def test_explicit_mapping_readonly_requests_and_completed_candle_filter():
    clock = Clock(local("09:27"))
    transport = Transport(clock, ok([candle("09:15"), candle("09:20"), candle("09:25", close="forming-not-a-price"), candle("09:30")]))
    provider = SmartAPIHistoricalProvider(client(clock, transport))
    assert isinstance(provider, MarketDataProvider)
    bars = provider.get_bars(SYMBOL, local("09:15"), local("09:35"), as_of=clock.now())
    assert len(bars) == 2
    assert bars[0].timestamp.isoformat() == "2026-01-05T03:45:00+00:00"
    assert all(bar.source == "HISTORICAL" and bar.complete for bar in bars)
    request, = transport.requests
    assert request.full_url == BASE_URL + CANDLE_PATH
    assert json.loads(request.data) == {"exchange": "NSE", "symboltoken": INSTRUMENT, "interval": "FIVE_MINUTE", "fromdate": "2026-01-05 09:15", "todate": "2026-01-05 09:27"}
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["authorization"] == "Bearer " + TOKEN
    assert headers["x-privatekey"] == API_KEY
    with pytest.raises(ValueError, match="configured"):
        provider.get_bars("UNKNOWN", local("09:15"), local("09:30"))
    assert len(transport.requests) == 1


@pytest.mark.parametrize("indices", [(), (1, 2), (0, 2), (0, 1)])
def test_historical_missing_intraday_candles_fail_closed(indices):
    clock = Clock()
    rows = [candle(value) for value in ("09:15", "09:20", "09:25")]
    transport = Transport(clock, ok([rows[index] for index in indices]))
    provider = SmartAPIHistoricalProvider(client(clock, transport))
    with pytest.raises(DataValidationError, match="missing intraday"):
        provider.get_bars(SYMBOL, local("09:15"), local("09:30"))


def test_holidays_weekends_and_precompletion_require_no_api_or_fallback():
    clock = Clock(local("09:19:59"))
    transport = Transport(clock)
    provider = SmartAPIHistoricalProvider(client(clock, transport), calendar=SessionCalendar(holidays=["2026-01-26"]))
    assert provider.get_bars(SYMBOL, local("09:15"), local("10:00")) == ()
    assert provider.get_bars(SYMBOL, local("09:15", "2026-01-04"), local("15:30", "2026-01-04")) == ()
    assert provider.get_bars(SYMBOL, local("09:15", "2026-01-26"), local("15:30", "2026-01-26"), as_of=local("16:00", "2026-01-26")) == ()
    assert not transport.requests


def test_chunking_filters_inclusive_api_boundaries_without_deduplicating_bad_rows():
    clock = Clock(local("09:40"))
    transport = Transport(clock, ok([candle(value) for value in ("09:15", "09:20", "09:25")]), ok([candle(value) for value in ("09:25", "09:30", "09:35")]))
    provider = SmartAPIHistoricalProvider(client(clock, transport), max_request_span=timedelta(minutes=10))
    bars = provider.get_bars(SYMBOL, local("09:15"), local("09:35"))
    assert [bar.timestamp for bar in bars] == [local(value) for value in ("09:15", "09:20", "09:25", "09:30")]
    assert transport.starts == [0, 1]
    too_large = SmartAPIHistoricalProvider(client(clock, Transport(clock)), max_request_span=timedelta(minutes=10), max_chunks=1)
    with pytest.raises(ValueError, match="chunk bound"):
        too_large.get_bars(SYMBOL, local("09:15"), local("09:35"))


@pytest.mark.parametrize("rows,error", [
    ([candle("09:15"), candle("09:15")], DataValidationError),
    ([candle("09:15", timestamp="2026-01-05T09:15:00")], SmartAPIResponseError),
    ([candle("09:15", close=200)], SmartAPIResponseError),
    ([["2026-01-05T09:15:00+05:30", 100]], SmartAPIResponseError),
])
def test_malformed_or_duplicate_historical_payload_is_never_repaired(rows, error):
    clock = Clock()
    provider = SmartAPIHistoricalProvider(client(clock, Transport(clock, ok(rows))))
    with pytest.raises(error):
        provider.get_bars(SYMBOL, local("09:15"), local("09:20"))


def test_provider_snapshot_manifest_contains_source_facts_and_no_session_details(tmp_path):
    clock = Clock()
    provider = SmartAPIHistoricalProvider(client(clock, Transport(clock, ok([candle("09:15")]))))
    bars = provider.get_bars(SYMBOL, local("09:15"), local("09:20"))
    path = tmp_path / "authorised-history.csv"
    manifest = write_snapshot(path, bars, source=provider.source, provenance=provider.provenance, source_metadata=provider.source_metadata, captured_at=clock.now())
    assert load_snapshot(path, expected_manifest_sha256=manifest.digest).bars == bars
    text = manifest.to_json()
    assert TOKEN not in text and API_KEY not in text and INSTRUMENT not in text
    assert manifest.source_metadata["endpoint"] == CANDLE_PATH
    assert manifest.provenance == "HISTORICAL"


def test_429_retry_after_and_local_one_request_per_second_apply_to_all_calls():
    clock = Clock()
    transport = Transport(clock, HTTPResponse(429, {"Retry-After": "2"}, b"not echoed"), lambda: quote(clock), lambda: quote(clock))
    provider = SmartAPIQuoteProvider(client(clock, transport))
    first, second = provider.get_quote(SYMBOL), provider.get_quote(SYMBOL)
    assert first.price == Decimal("100.1234") and second.timestamp > first.timestamp
    assert transport.starts == [0, 2, 3]
    assert clock.sleeps == [2, 1]
    assert first.provenance == "READ_ONLY_POLL"
    assert json.loads(transport.requests[-1].data) == {"mode": "FULL", "exchangeTokens": {"NSE": [INSTRUMENT]}}


def test_http_date_retry_after_and_conservative_configurable_local_cap():
    clock = Clock()
    retry_at = format_datetime((clock.now() + timedelta(seconds=3)).astimezone(timezone.utc), usegmt=True)
    transport = Transport(clock, HTTPResponse(429, {"retry-after": retry_at}), lambda: quote(clock), lambda: quote(clock))
    provider = SmartAPIQuoteProvider(client(clock, transport, rate_limit_per_second=0.5))
    provider.get_quote(SYMBOL)
    provider.get_quote(SYMBOL)
    assert transport.starts == [0, 3, 5]


def test_excessive_retry_after_does_not_sleep_unbounded_or_retry_early():
    clock = Clock()
    transport = Transport(clock, HTTPResponse(429, {"Retry-After": "3600"}), lambda: quote(clock))
    provider = SmartAPIQuoteProvider(client(clock, transport))
    for _ in range(2):
        with pytest.raises(SmartAPIRateLimitError, match="wait bound"):
            provider.get_quote(SYMBOL)
    assert len(transport.requests) == 1 and not clock.sleeps
    clock.elapsed = 3600
    assert provider.get_quote(SYMBOL).timestamp == clock.now()
    assert len(transport.requests) == 2


def test_transient_disconnect_reconnect_uses_bounded_retry_and_fresh_response():
    clock = Clock()
    transport = Transport(clock, HTTPResponse(503), URLError(TOKEN + API_KEY), lambda: quote(clock))
    result = SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)
    assert result.timestamp == clock.now()
    assert transport.starts == [0, 1, 2]


@pytest.mark.parametrize("failure,error", [(HTTPResponse(429), SmartAPIRateLimitError), (HTTPResponse(503), SmartAPIUnavailableError), (URLError("mock disconnected"), SmartAPIUnavailableError)])
def test_retry_exhaustion_is_bounded_and_does_not_make_up_data(failure, error):
    clock = Clock()
    transport = Transport(clock, failure, failure, failure)
    with pytest.raises(error):
        SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)
    assert len(transport.requests) == 3 and len(clock.sleeps) == 2


@pytest.mark.parametrize("failure", [HTTPResponse(401), HTTPResponse(403), HTTPResponse(200, {}, json.dumps({"status": False, "errorcode": "AG8002", "message": TOKEN}).encode()), HTTPResponse(200, {}, json.dumps({"status": False, "error_type": "TokenException", "message": API_KEY}).encode())])
def test_auth_failure_is_not_retried_or_refreshed_and_latches(failure):
    clock = Clock()
    transport = Transport(clock, failure)
    provider = SmartAPIQuoteProvider(client(clock, transport))
    for _ in range(2):
        with pytest.raises(SmartAPIAuthenticationError) as error:
            provider.get_quote(SYMBOL)
        assert TOKEN not in str(error.value) and API_KEY not in str(error.value)
    assert len(transport.requests) == 1 and not clock.sleeps


def test_locally_expired_session_never_sends_and_expiry_during_wait_stops_retry():
    clock = Clock()
    transport = Transport(clock)
    with pytest.raises(SmartAPIAuthenticationError):
        SmartAPIQuoteProvider(client(clock, transport, expires_at=clock.now())).get_quote(SYMBOL)
    assert not transport.requests
    transport = Transport(clock, HTTPResponse(429, {"Retry-After": "2"}))
    provider = SmartAPIQuoteProvider(client(clock, transport, expires_at=clock.now() + timedelta(seconds=1)))
    with pytest.raises(SmartAPIAuthenticationError):
        provider.get_quote(SYMBOL)
    assert len(transport.requests) == 1


@pytest.mark.parametrize("age", [timedelta(seconds=16), timedelta(seconds=-1)])
def test_stale_or_future_exchange_timestamp_is_rejected_not_replaced_with_receipt(age):
    clock = Clock()
    transport = Transport(clock, quote(clock, feed=clock.now() - age))
    with pytest.raises(StaleQuoteError):
        SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)


@pytest.mark.parametrize("trade_time", [None, "2026-01-05T09:00:00+05:30"])
def test_fresh_heartbeat_does_not_make_old_or_undated_last_trade_live(trade_time):
    clock = Clock()
    transport = Transport(clock, quote(clock, exchTradeTime=trade_time))
    with pytest.raises(StaleQuoteError, match="heartbeat"):
        SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)


def test_offset_free_exchange_clock_is_explicit_ist_and_old_quotes_do_not_return_cache():
    clock = Clock()
    transport = Transport(clock, quote(clock, exchFeedTime="05-Jan-2026 09:30:00"), quote(clock, feed=clock.now() - timedelta(minutes=1)))
    provider = SmartAPIQuoteProvider(client(clock, transport))
    result = provider.get_quote(SYMBOL)
    assert result.timestamp == local() and result.received_at == clock.now()
    assert result.timestamp.isoformat() == "2026-01-05T04:00:00+00:00"
    with pytest.raises(StaleQuoteError):
        provider.get_quote(SYMBOL)


def test_missing_feed_timestamp_wrong_instrument_and_bad_price_fail_closed():
    clock = Clock()
    for changes in ({"exchFeedTime": None}, {"symbolToken": "999999"}, {"exchange": "BSE"}, {"ltp": "NaN"}, {"ltp": 0}, {"ltp": "1e10000"}, {"ltp": "1e-10000"}):
        transport = Transport(clock, quote(clock, **changes))
        with pytest.raises(SmartAPIResponseError):
            SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)


def test_poll_is_bounded_and_a_new_poll_can_reconnect_after_retry_exhaustion():
    clock = Clock()
    transport = Transport(clock, lambda: quote(clock), URLError("disconnect"), URLError("disconnect"), URLError("disconnect"), lambda: quote(clock))
    provider = SmartAPIQuoteProvider(client(clock, transport))
    poll = provider.poll(SYMBOL, count=3, poll_interval=0)
    assert next(poll).price > 0
    with pytest.raises(SmartAPIUnavailableError):
        next(poll)
    assert len(list(provider.poll(SYMBOL, count=1))) == 1
    assert len(transport.requests) == 5


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_refused_without_forwarding_credentials(status):
    clock = Clock()
    transport = UrllibTransport()
    handlers = [handler for handler in transport._opener.handlers if isinstance(handler, HTTPRedirectHandler)]
    assert len(handlers) == 1
    request = Request(BASE_URL + QUOTE_PATH, data=b"{}", headers={"Authorization": "Bearer " + TOKEN, "X-PrivateKey": API_KEY}, method="POST")
    assert handlers[0].redirect_request(request, None, status, "redirect", {}, "https://untrusted.example/collect") is None
    response = HTTPError(request.full_url, status, "redirect", {"Location": "https://untrusted.example/collect"}, io.BytesIO(TOKEN.encode()))
    with patch.object(transport._opener, "open", side_effect=response) as open_:
        with pytest.raises(SmartAPIResponseError, match="redirect"):
            SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)
    assert open_.call_count == 1


def test_arbitrary_origin_route_and_client_header_override_are_forbidden():
    clock = Clock()
    transport = Transport(clock)
    connection = client(clock, transport)
    with pytest.raises(SmartAPIResponseError, match="allowlisted"):
        connection._request_readonly("order.place", {})
    with pytest.raises(SmartAPIResponseError, match="allowlisted"):
        UrllibTransport()(Request("https://untrusted.example", data=b"{}"), timeout=7)
    with pytest.raises(ValueError, match="forbidden"):
        client(clock, transport, client_metadata={"Authorization": "override"})
    assert not transport.requests
    assert not any(hasattr(connection, name) for name in ("login", "refresh_token", "place_order", "cancel_order"))


def test_real_urllib_redirect_processor_is_stopped_before_any_second_request():
    clock = Clock()
    transport = UrllibTransport()
    requested = []

    class ScriptedHTTPS(BaseHandler):
        handler_order = 1

        def https_open(self, request):
            requested.append(request.full_url)
            response = addinfourl(io.BytesIO(b"redirect body"), {"Location": "https://untrusted.example/collect"}, request.full_url, 302)
            response.msg = "Found"
            return response

    transport._opener.add_handler(ScriptedHTTPS())
    # Make a mistaken escape to the actual networking handler a test failure.
    with patch("urllib.request.HTTPSHandler.https_open", side_effect=AssertionError("network forbidden")):
        with pytest.raises(SmartAPIResponseError, match="redirect"):
            SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)
    assert requested == [BASE_URL + QUOTE_PATH]


def test_malformed_error_marker_is_sanitized_and_latency_cannot_make_old_quote_fresh():
    clock = Clock()
    transport = Transport(clock, HTTPResponse(200, {}, json.dumps({"status": True, "errorcode": [TOKEN], "data": {}}).encode()))
    with pytest.raises(SmartAPIResponseError, match="schema") as error:
        SmartAPIQuoteProvider(client(clock, transport)).get_quote(SYMBOL)
    assert TOKEN not in str(error.value)
    original = clock.now()

    def slow_quote():
        clock.sleep(16)
        return quote(clock, feed=original)

    with pytest.raises(StaleQuoteError):
        SmartAPIQuoteProvider(client(clock, Transport(clock, slow_quote))).get_quote(SYMBOL)


@pytest.mark.parametrize("failure", [
    HTTPResponse(200, {}, (TOKEN + API_KEY).encode()),
    HTTPResponse(400, {}, (TOKEN + API_KEY).encode()),
    HTTPResponse(200, {}, json.dumps({"status": False, "message": TOKEN + API_KEY}).encode()),
    RuntimeError(TOKEN + API_KEY),
    SmartAPIResponseError(TOKEN + API_KEY),
])
def test_errors_and_repr_do_not_echo_credentials_response_bodies_or_transport_details(failure, caplog):
    clock = Clock()
    transport = Transport(clock, failure)
    connection = client(clock, transport)
    with pytest.raises(SmartAPIResponseError) as error:
        SmartAPIQuoteProvider(connection).get_quote(SYMBOL)
    output = str(error.value) + repr(connection) + caplog.text + "".join(traceback.format_exception(error.value))
    assert TOKEN not in output and API_KEY not in output
    assert TOKEN not in repr(HTTPResponse(400, {"Authorization": TOKEN}, API_KEY.encode()))
    assert len(transport.requests) == 1
