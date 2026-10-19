"""Read-only market-data provider protocols and deterministic fixtures."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .bars import DataProvenance, MarketBar, utc_timestamp
from .validation import validate_bars


@runtime_checkable
class MarketDataProvider(Protocol):
    """Read-only provider boundary consumed by features and paper mode."""

    provenance: str

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Return completed bars in ``[start, end)`` for one symbol."""


class InMemoryBarProvider:
    """Validated in-memory provider for offline tests and replay fixtures."""

    def __init__(
        self,
        bars: Iterable[MarketBar],
        *,
        provenance: str | None = None,
        stale_after: timedelta | None = None,
    ) -> None:
        materialised = validate_bars(bars)
        inferred = {bar.source for bar in materialised}
        if provenance is not None:
            provenance = provenance.strip().upper()
            if not provenance:
                raise ValueError("provenance must be non-empty")
            if any(bar.source != provenance for bar in materialised):
                raise ValueError("bar source does not match provider provenance")
        elif len(inferred) > 1:
            raise ValueError("a provider cannot mix provenance labels")
        self.provenance = provenance or next(iter(inferred), DataProvenance.HISTORICAL.value)
        self.stale_after = stale_after
        self._bars = tuple(materialised)

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> tuple[MarketBar, ...]:
        """Return a fresh tuple in source order without mutating provider data."""

        start_utc = utc_timestamp(start)
        end_utc = utc_timestamp(end)
        if end_utc <= start_utc:
            raise ValueError("end must be after start")
        result = tuple(
            bar
            for bar in self._bars
            if bar.symbol == symbol and start_utc <= bar.timestamp < end_utc
        )
        validate_bars(result, stale_after=self.stale_after, as_of=end_utc)
        return result

    def all_bars(self) -> tuple[MarketBar, ...]:
        """Return the immutable source snapshot."""

        return self._bars


class SyntheticBarProvider(InMemoryBarProvider):
    """Provider whose records are explicitly labelled ``SYNTHETIC``."""

    def __init__(self, bars: Iterable[MarketBar]) -> None:
        checked = tuple(bars)
        if any(bar.source != DataProvenance.SYNTHETIC.value for bar in checked):
            raise ValueError("SyntheticBarProvider accepts only SYNTHETIC bars")
        super().__init__(checked, provenance=DataProvenance.SYNTHETIC.value)


class CSVBarProvider(InMemoryBarProvider):
    """Load authorised offline CSV rows; no network access is performed.

    Required columns are ``symbol,timestamp,open,high,low,close,volume``.  ISO
    timestamps must carry a UTC offset.  Source rows are validated without
    sorting or removing duplicates, and cannot override declared provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        provenance: str = DataProvenance.HISTORICAL.value,
    ) -> None:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = []
            for row in csv.DictReader(handle):
                if "source" not in row:
                    row["source"] = provenance
                rows.append(MarketBar.from_mapping(row))
        super().__init__(rows, provenance=provenance)


def synthetic_bars(
    symbol: str,
    start: datetime,
    closes: Iterable[Decimal | int | float | str],
    *,
    interval: timedelta = timedelta(minutes=5),
    volume: Decimal | int | float | str = 100,
) -> tuple[MarketBar, ...]:
    """Create deterministic completed candles for offline tests.

    This helper is intentionally explicit about provenance.  A caller cannot
    accidentally report these fixtures as historical or live market data.
    """

    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    current_time = utc_timestamp(start)
    previous_close: Decimal | None = None
    result: list[MarketBar] = []
    for raw_close in closes:
        close = Decimal(str(raw_close))
        if not close.is_finite() or close <= 0:
            raise ValueError("synthetic close must be positive")
        open_price = previous_close or close
        high = max(open_price, close) * Decimal("1.001")
        low = min(open_price, close) * Decimal("0.999")
        result.append(
            MarketBar(
                symbol=symbol,
                timestamp=current_time,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source=DataProvenance.SYNTHETIC.value,
            )
        )
        previous_close = close
        current_time += interval
    return tuple(result)
