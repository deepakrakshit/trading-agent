"""Cross-record market-data quality checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from .bars import DataValidationError, MarketBar, utc_timestamp


def validate_bars(
    bars: Iterable[MarketBar],
    *,
    stale_after: timedelta | None = None,
    as_of: datetime | None = None,
    require_positive_volume: bool = False,
    expected_interval: timedelta | None = None,
) -> tuple[MarketBar, ...]:
    """Validate ordering, duplicates, freshness, and per-bar schema.

    Bars must be chronological for each symbol.  This function does not sort
    input: silently sorting a provider response would hide a feed defect and
    make replay order ambiguous.  ``as_of`` is injectable so stale-feed tests
    and replays remain deterministic.
    """

    materialised = tuple(bars)
    issues: list[str] = []
    seen: set[tuple[str, datetime]] = set()
    previous: dict[str, datetime] = {}
    latest: dict[str, datetime] = {}
    reference = utc_timestamp(as_of) if as_of is not None else None
    if expected_interval is not None and expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")

    for index, bar in enumerate(materialised):
        if not isinstance(bar, MarketBar):
            issues.append(f"row {index} is not a MarketBar")
            continue
        key = (bar.symbol, bar.timestamp)
        if key in seen:
            issues.append(f"duplicate bar for {bar.symbol} at {bar.timestamp.isoformat()}")
        seen.add(key)
        prior = previous.get(bar.symbol)
        if prior is not None and bar.timestamp <= prior:
            issues.append(
                f"out-of-order bar for {bar.symbol}: {bar.timestamp.isoformat()} "
                f"after {prior.isoformat()}"
            )
        elif prior is not None and expected_interval is not None:
            if bar.timestamp - prior != expected_interval:
                issues.append(f"missing/irregular bar interval for {bar.symbol}")
        previous[bar.symbol] = bar.timestamp
        latest[bar.symbol] = max(latest.get(bar.symbol, bar.timestamp), bar.timestamp)
        if require_positive_volume and bar.volume <= 0:
            issues.append(f"non-positive volume for {bar.symbol} at {bar.timestamp.isoformat()}")
        if reference is not None and bar.timestamp > reference:
            issues.append(f"future/unavailable bar for {bar.symbol} at {bar.timestamp.isoformat()}")

    if stale_after is not None:
        if stale_after < timedelta(0):
            raise ValueError("stale_after must be non-negative")
        reference = reference or datetime.now(timezone.utc)
        if not latest:
            issues.append("missing market feed: no completed bars")
        for symbol, timestamp in latest.items():
            age = reference - timestamp
            if age > stale_after:
                issues.append(
                    f"stale feed for {symbol}: latest bar is {age.total_seconds():.3f}s old"
                )

    if issues:
        raise DataValidationError("market-bar validation failed", issues=tuple(issues))
    return materialised
