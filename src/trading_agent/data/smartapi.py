"""Narrow, read-only SmartAPI HTTP adapters with caller-owned session tokens.

Only the official SDK's historical-candle and market-quote POST paths are
allowlisted. There is no login, token refresh, SDK import, credential-file
handling, arbitrary URL, or order interface. Credentials exist in memory only;
requests, response bodies and transport exceptions are never logged or echoed.

The public SDK confirms origin/path/request conventions, not current terms,
costs, rate limits, entitlements or availability. The default 1 request/second
is a conservative LOCAL cap, not a verified broker limit. FULL quote payload
timestamp formats are checked locally and need verification against authorised
account responses. Poll results are labelled READ_ONLY_POLL, not verified live.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import ipaddress
import json
import math
import re
import ssl
import threading
import time
from types import MappingProxyType
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from .bars import DataProvenance, DataValidationError, MarketBar, decimal_value, utc_timestamp
from .calendar import FIVE_MINUTES, KOLKATA, SessionCalendar

BASE_URL = "https://apiconnect.angelone.in"
CANDLE_PATH = "/rest/secure/angelbroking/historical/v1/getCandleData"
QUOTE_PATH = "/rest/secure/angelbroking/market/v1/quote"
_ROUTES = MappingProxyType({"candles": CANDLE_PATH, "quote": QUOTE_PATH})
_URLS = frozenset(BASE_URL + path for path in _ROUTES.values())
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_AUTH_CODES = frozenset({"AG8001", "AG8002", "AG8003", "AB1010", "AB1007"})
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class SmartAPIError(RuntimeError):
    """Sanitized adapter failure; no raw headers, response or credentials."""


class SmartAPIAuthenticationError(SmartAPIError):
    """A pre-obtained session is unavailable/expired; automatic retry stops."""


class SmartAPIRateLimitError(SmartAPIError):
    """Rate limiting exhausted retries or requested a wait beyond local bounds."""


class SmartAPIUnavailableError(SmartAPIError):
    """Transport/transient failures exhausted the bounded retry budget."""


class SmartAPIResponseError(SmartAPIError):
    """Unusable response, forbidden redirect, or unexpected payload schema."""


class StaleQuoteError(SmartAPIError):
    """A quote is stale, unavailable, future-dated or out of sequence/session."""


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    body: bytes = field(default=b"", repr=False)


class HTTPTransport(Protocol):
    def __call__(self, request: Request, *, timeout: float) -> HTTPResponse:
        ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even same-origin redirects are refused: forwarding Authorization or
        # X-PrivateKey is never delegated to urllib's default redirect handler.
        return None


def _read_http(response) -> HTTPResponse:
    with response:
        status = int(response.status)
        headers = dict(response.headers or {})
        if not 200 <= status < 300:
            # HTTP failure bodies are unnecessary for classification and can
            # contain reflected auth material. Do not even read them.
            return HTTPResponse(status, headers)
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise SmartAPIResponseError("SmartAPI response exceeds the local size bound")
        return HTTPResponse(status, headers, body)


class UrllibTransport:
    """TLS-verified urllib transport; redirects and environment proxies disabled."""

    def __init__(self) -> None:
        self._opener = build_opener(ProxyHandler({}), _NoRedirect(), HTTPSHandler(context=ssl.create_default_context()))

    def __call__(self, request: Request, *, timeout: float) -> HTTPResponse:
        if request.full_url not in _URLS or request.get_method() != "POST":
            raise SmartAPIResponseError("SmartAPI transport accepts only allowlisted read-only POST requests")
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        return _read_http(response)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    timeout_seconds: float = 7.0
    backoff_seconds: float = 0.5
    max_retry_delay: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be an integer in [1, 10]")
        values = (self.timeout_seconds, self.backoff_seconds, self.max_retry_delay)
        if not all(math.isfinite(value) for value in values) or self.timeout_seconds <= 0 or self.max_retry_delay <= 0 or self.backoff_seconds < 0:
            raise ValueError("retry time bounds must be finite and valid")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SmartAPIClient:
    """Shared auth/rate/transport boundary for the two read-only adapters.

    ``symbol_tokens`` is an explicit symbol -> NSE instrument-token mapping;
    no token master, ticker inference, or order/search endpoint is used.
    Optional ``client_metadata`` accepts only locally supplied IP/MAC headers
    used by the official SDK. It is never discovered or persisted here.
    Reconstruct the client with a new pre-obtained session after auth expiry.
    """

    def __init__(
        self,
        *,
        token: str,
        api_key: str,
        symbol_tokens: Mapping[str, str],
        transport: HTTPTransport | None = None,
        retry_policy: RetryPolicy = RetryPolicy(),
        rate_limit_per_second: float = 1.0,
        expires_at: datetime | None = None,
        client_metadata: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = _now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(token, str) or not isinstance(api_key, str):
            raise ValueError("a pre-obtained token and API key are required")
        token = token.removeprefix("Bearer ")
        if any(not value or len(value) > 16_384 or any(ord(char) < 33 or ord(char) > 126 for char in value) for value in (token, api_key)):
            raise ValueError("session credentials must be non-empty printable header values")
        if not math.isfinite(rate_limit_per_second) or rate_limit_per_second <= 0:
            raise ValueError("rate_limit_per_second must be finite and positive")
        mapping = dict(symbol_tokens)
        if not mapping or any(not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip() or not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) for symbol, value in mapping.items()):
            raise ValueError("explicit symbol -> numeric-string NSE token mappings are required")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("instrument tokens cannot ambiguously map to multiple symbols")
        metadata = dict(client_metadata or {})
        if set(metadata) - {"X-ClientLocalIP", "X-ClientPublicIP", "X-MACAddress"}:
            raise ValueError("client metadata contains a forbidden header")
        try:
            for name, value in metadata.items():
                if not isinstance(value, str):
                    raise ValueError
                if name == "X-MACAddress":
                    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}", value):
                        raise ValueError
                else:
                    ipaddress.ip_address(value)
        except (ValueError, TypeError):
            raise ValueError("client IP/MAC metadata is invalid") from None
        self._token, self._api_key = token, api_key
        self._tokens = MappingProxyType(mapping)
        self._client_metadata = metadata
        self._transport = transport if transport is not None else UrllibTransport()
        self.retry_policy = retry_policy
        self._interval = 1.0 / rate_limit_per_second
        self._expires_at = utc_timestamp(expires_at) if expires_at is not None else None
        self._clock, self._monotonic, self._sleep = clock, monotonic, sleep
        self._lock = threading.Lock()
        self._last_start: float | None = None
        self._not_before = 0.0
        self._auth_failed = False

    def __repr__(self) -> str:
        return "<SmartAPIClient read_only=True>"

    def now(self) -> datetime:
        return utc_timestamp(self._clock())

    def token_for(self, symbol: str) -> str:
        try:
            return self._tokens[symbol]
        except KeyError:
            raise ValueError("instrument has no explicitly configured NSE token") from None

    def _check_auth(self) -> None:
        if self._auth_failed or (self._expires_at is not None and self.now() >= self._expires_at):
            self._auth_failed = True
            raise SmartAPIAuthenticationError("SmartAPI session is unavailable; supply a new pre-obtained session")

    def _reserve_slot(self) -> None:
        with self._lock:
            now = self._monotonic()
            if self._not_before - now > self.retry_policy.max_retry_delay:
                raise SmartAPIRateLimitError("SmartAPI retry-after exceeds the local wait bound; retry later")
            earliest = max(now, self._not_before, self._last_start + self._interval if self._last_start is not None else now)
            if earliest > now:
                self._sleep(earliest - now)
            self._last_start = self._monotonic()

    def _cooldown(self, delay: float) -> None:
        with self._lock:
            self._not_before = max(self._not_before, self._monotonic() + delay)

    def _retry_delay(self, response: HTTPResponse | None, attempt: int) -> float:
        delay = min(self.retry_policy.max_retry_delay, self.retry_policy.backoff_seconds * 2**attempt)
        if response is None:
            return delay
        retry_after = next((value for key, value in response.headers.items() if key.lower() == "retry-after"), None)
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except (ValueError, TypeError):
                try:
                    seconds = (utc_timestamp(parsedate_to_datetime(retry_after)) - self.now()).total_seconds()
                except (ValueError, TypeError, OverflowError, DataValidationError):
                    seconds = 0.0
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        return delay

    def _request_readonly(self, route: str, payload: Mapping) -> object:
        if route not in _ROUTES:
            raise SmartAPIResponseError("SmartAPI route is not allowlisted")
        for attempt in range(self.retry_policy.max_attempts):
            self._check_auth()
            self._reserve_slot()
            self._check_auth()  # The session may have expired during a wait.
            headers = {
                "Authorization": "Bearer " + self._token, "X-PrivateKey": self._api_key,
                "Content-Type": "application/json", "Accept": "application/json",
                "X-UserType": "USER", "X-SourceID": "WEB", **self._client_metadata,
            }
            request = Request(BASE_URL + _ROUTES[route], data=json.dumps(dict(payload), allow_nan=False).encode("utf-8"), headers=headers, method="POST")
            response = None
            try:
                try:
                    response = self._transport(request, timeout=self.retry_policy.timeout_seconds)
                except HTTPError as error:
                    response = _read_http(error)
            except (URLError, OSError, TimeoutError, ConnectionError):
                if attempt + 1 == self.retry_policy.max_attempts:
                    raise SmartAPIUnavailableError("SmartAPI transport unavailable after bounded retries") from None
                self._cooldown(self._retry_delay(None, attempt))
                continue
            except SmartAPIError:
                raise SmartAPIResponseError("SmartAPI transport rejected the read-only request") from None
            except Exception:
                raise SmartAPIResponseError("SmartAPI transport failed") from None
            if not isinstance(response, HTTPResponse) or isinstance(response.status, bool) or not isinstance(response.status, int) or not 100 <= response.status <= 599:
                raise SmartAPIResponseError("SmartAPI transport returned an invalid response")
            if response.status in (401, 403):
                self._auth_failed = True
                raise SmartAPIAuthenticationError("SmartAPI session was rejected; supply a new pre-obtained session")
            if 300 <= response.status < 400:
                raise SmartAPIResponseError("SmartAPI redirects are refused")
            if response.status in _RETRY_STATUSES:
                self._cooldown(self._retry_delay(response, attempt))
                if attempt + 1 < self.retry_policy.max_attempts:
                    continue
                if response.status == 429:
                    raise SmartAPIRateLimitError("SmartAPI rate limit exhausted bounded retries")
                raise SmartAPIUnavailableError("SmartAPI transient response exhausted bounded retries")
            if not 200 <= response.status < 300:
                raise SmartAPIResponseError(f"SmartAPI read-only request failed (HTTP {response.status})")
            try:
                if not isinstance(response.body, bytes) or len(response.body) > _MAX_RESPONSE_BYTES:
                    raise ValueError
                data = json.loads(response.body)
                if not isinstance(data, dict):
                    raise ValueError
            except (ValueError, TypeError, UnicodeError):
                raise SmartAPIResponseError("SmartAPI response was not valid bounded JSON") from None
            error_code = data.get("errorcode")
            if error_code is not None and not isinstance(error_code, str):
                raise SmartAPIResponseError("SmartAPI response error-code schema is invalid")
            if data.get("error_type") == "TokenException" or error_code in _AUTH_CODES:
                self._auth_failed = True
                raise SmartAPIAuthenticationError("SmartAPI session expired or was rejected")
            if data.get("status") is not True or data.get("error_type"):
                raise SmartAPIResponseError("SmartAPI read-only request was not successful")
            if "data" not in data:
                raise SmartAPIResponseError("SmartAPI response omitted its data payload")
            return data["data"]
        raise AssertionError("unreachable retry state")


def _candle_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise SmartAPIResponseError("SmartAPI candle timestamp is invalid")
    try:
        return utc_timestamp(datetime.fromisoformat(value))
    except (ValueError, TypeError):
        raise SmartAPIResponseError("SmartAPI candle timestamp must carry a UTC offset") from None


class SmartAPIHistoricalProvider:
    """Completed FIVE_MINUTE NSE candles in [start, end), with gap rejection."""

    source = "ANGEL_ONE_SMARTAPI"
    provenance = DataProvenance.HISTORICAL.value

    def __init__(self, client: SmartAPIClient, *, calendar: SessionCalendar | None = None, max_request_span: timedelta = timedelta(days=30), max_chunks: int = 100) -> None:
        if max_request_span < FIVE_MINUTES or max_request_span % timedelta(minutes=1):
            raise ValueError("max_request_span must contain whole minutes and be at least five minutes")
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer")
        self._client = client
        self.calendar = calendar if calendar is not None else SessionCalendar()
        self.max_request_span, self.max_chunks = max_request_span, max_chunks

    @property
    def source_metadata(self) -> Mapping[str, str]:
        return MappingProxyType({"provider": self.source, "exchange": "NSE", "interval": "FIVE_MINUTE", "api_origin": BASE_URL, "endpoint": CANDLE_PATH, "timestamp_convention": "BAR_START", "calendar_version": self.calendar.version})

    def get_bars(self, symbol: str, start: datetime, end: datetime, *, as_of: datetime | None = None) -> tuple[MarketBar, ...]:
        token = self._client.token_for(symbol)
        start, end = utc_timestamp(start), utc_timestamp(end)
        reference = utc_timestamp(as_of) if as_of is not None else self._client.now()
        expected = self.calendar.expected_bar_starts(start, end, as_of=reference)
        if not expected:
            return ()
        stop = min(end, reference)
        if math.ceil((stop - start) / self.max_request_span) > self.max_chunks:
            raise ValueError("historical range exceeds the configured local chunk bound")
        bars = []
        cursor = start
        while cursor < stop:
            boundary = min(cursor + self.max_request_span, stop)
            if self.calendar.expected_bar_starts(cursor, boundary, as_of=reference):
                payload = {"exchange": "NSE", "symboltoken": token, "interval": "FIVE_MINUTE", "fromdate": cursor.astimezone(KOLKATA).strftime("%Y-%m-%d %H:%M"), "todate": boundary.astimezone(KOLKATA).strftime("%Y-%m-%d %H:%M")}
                rows = self._client._request_readonly("candles", payload)
                if not isinstance(rows, list):
                    raise SmartAPIResponseError("SmartAPI candle payload must be a list")
                for row in rows:
                    if not isinstance(row, list) or len(row) != 6:
                        raise SmartAPIResponseError("SmartAPI candle row schema is invalid")
                    timestamp = _candle_timestamp(row[0])
                    if not cursor <= timestamp < boundary or timestamp + FIVE_MINUTES > reference:
                        continue  # Inclusive API endpoints and forming bars are not observations.
                    try:
                        bars.append(MarketBar(symbol, timestamp, *row[1:], source=self.provenance))
                    except (DataValidationError, TypeError):
                        raise SmartAPIResponseError("SmartAPI candle values failed validation") from None
            cursor = boundary
        return self.calendar.validate_intraday_bars(bars, symbol, start=start, end=end, as_of=reference)


def _quote_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise SmartAPIResponseError("SmartAPI quote omitted a valid exchange-feed timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        # Explicit English month decoding avoids process-locale dependence.
        match = re.fullmatch(r"(\d{2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2})", value)
        months = {name: index for index, name in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
        try:
            if match is None:
                raise ValueError
            day, month, year, hour, minute, second = match.groups()
            timestamp = datetime(int(year), months[month.lower()], int(day), int(hour), int(minute), int(second))
        except (ValueError, KeyError):
            raise SmartAPIResponseError("SmartAPI quote exchange-feed timestamp format is unsupported") from None
    # The provider's offset-free exchange-feed clock is explicitly interpreted
    # as Asia/Kolkata, never the host's local timezone or receipt timestamp.
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=KOLKATA)
    return utc_timestamp(timestamp)


@dataclass(frozen=True, slots=True)
class MarketQuote:
    symbol: str
    timestamp: datetime
    received_at: datetime
    price: Decimal
    trade_timestamp: datetime | None = None
    source: str = "ANGEL_ONE_SMARTAPI"
    provenance: str = "READ_ONLY_POLL"


class SmartAPIQuoteProvider:
    """Fresh FULL-mode quote polls, without claims of verified live entitlement."""

    source = "ANGEL_ONE_SMARTAPI"
    provenance = "READ_ONLY_POLL"

    def __init__(self, client: SmartAPIClient, *, max_age: timedelta = timedelta(seconds=15), future_tolerance: timedelta = timedelta(0), calendar: SessionCalendar | None = None) -> None:
        if max_age <= timedelta(0) or future_tolerance < timedelta(0):
            raise ValueError("quote age must be positive and future tolerance non-negative")
        self._client, self.max_age, self.future_tolerance = client, max_age, future_tolerance
        self.calendar = calendar if calendar is not None else SessionCalendar()
        self._last_timestamp: dict[str, datetime] = {}

    def get_quote(self, symbol: str) -> MarketQuote:
        token = self._client.token_for(symbol)
        if not self.calendar.is_open(self._client.now()):
            raise StaleQuoteError("SmartAPI quote polling is outside the configured session")
        data = self._client._request_readonly("quote", {"mode": "FULL", "exchangeTokens": {"NSE": [token]}})
        received = self._client.now()
        if not isinstance(data, dict) or not isinstance(data.get("fetched"), list) or len(data["fetched"]) != 1 or data.get("unfetched"):
            raise SmartAPIResponseError("SmartAPI quote payload is missing or ambiguous")
        row = data["fetched"][0]
        if not isinstance(row, dict) or row.get("exchange") != "NSE" or row.get("symbolToken") != token:
            raise SmartAPIResponseError("SmartAPI quote instrument does not match the configured mapping")
        timestamp = _quote_timestamp(row.get("exchFeedTime"))
        trade_time = _quote_timestamp(row["exchTradeTime"]) if row.get("exchTradeTime") else None
        if received - timestamp > self.max_age or timestamp - received > self.future_tolerance:
            raise StaleQuoteError("SmartAPI quote is stale or future-dated")
        if trade_time is not None and trade_time - received > self.future_tolerance:
            raise StaleQuoteError("SmartAPI quote trade timestamp is future-dated")
        if not self.calendar.is_open(timestamp) or not self.calendar.is_open(received):
            raise StaleQuoteError("SmartAPI quote is outside the configured session")
        if symbol in self._last_timestamp and timestamp < self._last_timestamp[symbol]:
            raise StaleQuoteError("SmartAPI quote exchange-feed timestamp moved backwards")
        try:
            price = decimal_value(row.get("ltp"), "quote price")
            if price <= 0 or not math.isfinite(float(price)) or float(price) == 0:
                raise ValueError
        except (ValueError, TypeError):
            raise SmartAPIResponseError("SmartAPI quote price is invalid") from None
        self._last_timestamp[symbol] = timestamp
        return MarketQuote(symbol, timestamp, received, price, trade_time)

    def poll(self, symbol: str, *, count: int = 1, poll_interval: float = 1.0) -> Iterator[MarketQuote]:
        """Bounded polling; exhausted retries/errors propagate, never serve cache."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or not math.isfinite(poll_interval) or poll_interval < 0:
            raise ValueError("poll count/interval must be finite and valid")
        for index in range(count):
            if index:
                self._client._sleep(poll_interval)
            yield self.get_quote(symbol)


__all__ = [
    "BASE_URL", "CANDLE_PATH", "QUOTE_PATH", "HTTPResponse", "HTTPTransport", "MarketQuote",
    "RetryPolicy", "SmartAPIAuthenticationError", "SmartAPIClient", "SmartAPIError",
    "SmartAPIHistoricalProvider", "SmartAPIQuoteProvider", "SmartAPIRateLimitError",
    "SmartAPIResponseError", "SmartAPIUnavailableError", "StaleQuoteError", "UrllibTransport",
]
