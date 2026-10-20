"""Provider-independent completed-bar feeds for continuous paper replay.

The finite :class:`MarketDataProvider` protocol remains the source boundary for
historical range queries.  This module adds a narrow incremental envelope for a
runner without claiming that any provider is live or entitled.  Every event
has an explicit availability time, local receipt time, provider identity and
deterministic event ID.  No credentials or raw provider payloads are stored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Protocol

from .bars import DataProvenance, MarketBar, utc_timestamp
from .providers import MarketDataProvider


class FeedError(RuntimeError):
    """Base class for incremental-feed failures."""


class FeedUnavailableError(FeedError):
    """A bounded retry may safely retry this provider failure."""


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Non-secret identity and contract metadata for one feed epoch."""

    provider_id: str
    provenance: str
    interval: timedelta = timedelta(minutes=5)
    timestamp_convention: str = "BAR_START"
    exchange: str = "NSE"
    calendar_version: str = "UNSPECIFIED"
    instrument_mapping_version: str | None = None
    source_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be non-empty")
        if self.interval <= timedelta(0):
            raise ValueError("feed interval must be positive")
        if self.timestamp_convention != "BAR_START":
            raise ValueError("continuous feeds require BAR_START timestamps")
        if not isinstance(self.exchange, str) or not self.exchange.strip():
            raise ValueError("exchange must be non-empty")
        if not isinstance(self.calendar_version, str) or not self.calendar_version.strip():
            raise ValueError("calendar_version must be non-empty")
        metadata = dict(self.source_metadata)
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key.strip()
            or not value.strip()
            or any(ord(char) < 32 for char in value)
            for key, value in metadata.items()
        ):
            raise ValueError("source_metadata must contain printable string pairs")
        object.__setattr__(self, "provider_id", self.provider_id.strip())
        object.__setattr__(self, "provenance", self.provenance.strip().upper())
        object.__setattr__(self, "exchange", self.exchange.strip().upper())
        object.__setattr__(self, "calendar_version", self.calendar_version.strip())
        object.__setattr__(self, "source_metadata", MappingProxyType(metadata))

    def to_mapping(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provenance": self.provenance,
            "interval_seconds": int(self.interval.total_seconds()),
            "timestamp_convention": self.timestamp_convention,
            "exchange": self.exchange,
            "calendar_version": self.calendar_version,
            "instrument_mapping_version": self.instrument_mapping_version,
            "source_metadata": dict(self.source_metadata),
        }


@dataclass(frozen=True, slots=True)
class CompletedBarEvent:
    """One completed bar with provider and local-clock evidence."""

    bar: MarketBar
    available_at: datetime
    received_at: datetime
    provider_event_id: str
    provider: ProviderIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.bar, MarketBar):
            raise TypeError("bar must be a MarketBar")
        if not isinstance(self.provider_event_id, str) or not self.provider_event_id.strip():
            raise ValueError("provider_event_id must be non-empty")
        available = utc_timestamp(self.available_at)
        received = utc_timestamp(self.received_at)
        expected = self.bar.timestamp + self.provider.interval
        if available != expected:
            raise ValueError("available_at must equal bar timestamp plus feed interval")
        if received < available:
            raise ValueError("received_at cannot precede completed-bar availability")
        if self.bar.source != self.provider.provenance:
            raise ValueError("bar provenance does not match provider identity")
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "received_at", received)
        object.__setattr__(self, "provider_event_id", self.provider_event_id.strip())

    def to_mapping(self) -> dict[str, object]:
        return {
            "bar": self.bar.to_mapping(),
            "available_at": self.available_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "provider_event_id": self.provider_event_id,
            "provider": self.provider.to_mapping(),
        }


class CompletedBarFeed(Protocol):
    """Incremental provider boundary consumed by the continuous runner."""

    identity: ProviderIdentity

    def fetch(
        self,
        symbol: str,
        after: datetime | None,
        until: datetime,
        *,
        as_of: datetime,
    ) -> Sequence[CompletedBarEvent]:
        """Return completed events after ``after`` and before ``until``."""


def _event_id(provider: ProviderIdentity, bar: MarketBar) -> str:
    payload = json.dumps(
        {"provider": provider.to_mapping(), "bar": bar.to_mapping()},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ReplayBarFeed:
    """Deterministic completed-bar feed over an immutable bar sequence."""

    def __init__(self, bars: Sequence[MarketBar], identity: ProviderIdentity) -> None:
        if not bars:
            raise ValueError("replay feed requires at least one bar")
        if any(bar.source != identity.provenance for bar in bars):
            raise ValueError("replay bars do not match provider provenance")
        previous: datetime | None = None
        for bar in bars:
            if previous is not None and bar.timestamp <= previous:
                raise ValueError("replay bars must be strictly chronological")
            previous = bar.timestamp
        self.identity = identity
        self._bars = tuple(bars)

    def events(self) -> tuple[CompletedBarEvent, ...]:
        """Return all events with ideal deterministic receipt times."""

        return tuple(
            CompletedBarEvent(
                bar,
                bar.timestamp + self.identity.interval,
                bar.timestamp + self.identity.interval,
                _event_id(self.identity, bar),
                self.identity,
            )
            for bar in self._bars
        )

    def fetch(
        self,
        symbol: str,
        after: datetime | None,
        until: datetime,
        *,
        as_of: datetime,
    ) -> tuple[CompletedBarEvent, ...]:
        if symbol not in {bar.symbol for bar in self._bars}:
            return ()
        boundary = utc_timestamp(until)
        clock = utc_timestamp(as_of)
        after_utc = utc_timestamp(after) if after is not None else None
        return tuple(
            event
            for event in self.events()
            if event.bar.symbol == symbol
            and (after_utc is None or event.bar.timestamp > after_utc)
            and event.bar.timestamp < boundary
            and event.available_at <= clock
        )


class RangeProviderFeed:
    """Adapt the existing finite range provider to completed-bar events."""

    def __init__(self, provider: MarketDataProvider, identity: ProviderIdentity) -> None:
        self.provider = provider
        self.identity = identity

    def fetch(
        self,
        symbol: str,
        after: datetime | None,
        until: datetime,
        *,
        as_of: datetime,
    ) -> tuple[CompletedBarEvent, ...]:
        end = utc_timestamp(until)
        start = (
            utc_timestamp(after) + self.identity.interval
            if after is not None
            else datetime.min.replace(tzinfo=timezone.utc)
        )
        bars = self.provider.get_bars(symbol, start, end)
        receipt = utc_timestamp(as_of)
        return tuple(
            CompletedBarEvent(
                bar,
                bar.timestamp + self.identity.interval,
                max(receipt, bar.timestamp + self.identity.interval),
                _event_id(self.identity, bar),
                self.identity,
            )
            for bar in bars
            if bar.timestamp + self.identity.interval <= receipt
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Small bounded retry policy for provider polling failures."""

    max_attempts: int = 3
    backoff_seconds: float = 0.5
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if not all(math.isfinite(value) for value in (self.backoff_seconds, self.max_delay_seconds)):
            raise ValueError("retry delays must be finite")
        if self.backoff_seconds < 0 or self.max_delay_seconds <= 0:
            raise ValueError("retry delays are invalid")


def fetch_with_retry(
    feed: CompletedBarFeed,
    symbol: str,
    after: datetime | None,
    until: datetime,
    *,
    as_of: datetime,
    policy: RetryPolicy = RetryPolicy(),
    sleep: Callable[[float], None] = time.sleep,
) -> Sequence[CompletedBarEvent]:
    """Fetch with bounded retries; never substitutes cached/synthetic data."""

    for attempt in range(policy.max_attempts):
        try:
            return feed.fetch(symbol, after, until, as_of=as_of)
        except FeedUnavailableError:
            if attempt + 1 == policy.max_attempts:
                raise
            sleep(min(policy.max_delay_seconds, policy.backoff_seconds * 2**attempt))
    raise AssertionError("unreachable retry state")


__all__ = [
    "CompletedBarEvent", "CompletedBarFeed", "FeedError", "FeedUnavailableError",
    "ProviderIdentity", "RangeProviderFeed", "ReplayBarFeed", "RetryPolicy",
    "fetch_with_retry",
]
