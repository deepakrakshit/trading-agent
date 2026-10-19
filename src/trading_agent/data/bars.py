"""Immutable market-bar schema and basic value validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping


class DataValidationError(ValueError):
    """Raised when market data cannot safely enter the paper pipeline."""

    def __init__(self, message: str, *, issues: tuple[str, ...] = ()) -> None:
        super().__init__(message + (": " + "; ".join(issues) if issues else ""))
        self.issues = issues or (message,)


class DataProvenance(str, Enum):
    """Known provenance labels used in persisted data records."""

    SYNTHETIC = "SYNTHETIC"
    HISTORICAL = "HISTORICAL"
    LIVE_READ_ONLY = "LIVE_READ_ONLY"


def utc_timestamp(value: datetime) -> datetime:
    """Return an aware UTC timestamp or reject an ambiguous naive timestamp."""

    if not isinstance(value, datetime):
        raise DataValidationError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def decimal_value(value: Any, field: str) -> Decimal:
    """Convert a numeric value without accepting NaN, infinity, or booleans."""

    if isinstance(value, bool):
        raise DataValidationError(f"{field} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DataValidationError(f"{field} must be numeric") from exc
    if not number.is_finite():
        raise DataValidationError(f"{field} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class MarketBar:
    """A completed OHLCV candle with explicit source provenance.

    Timestamps are normalised to UTC.  Prices are stored as ``Decimal`` so
    accounting code does not inherit binary floating-point drift.  A bar is
    accepted only when its OHLC ordering is physically possible.
    """

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: str = DataProvenance.SYNTHETIC.value
    complete: bool = True

    def __post_init__(self) -> None:
        symbol = self.symbol.strip() if isinstance(self.symbol, str) else ""
        if not symbol:
            raise DataValidationError("symbol must be a non-empty string")
        timestamp = utc_timestamp(self.timestamp)
        open_price = decimal_value(self.open, "open")
        high_price = decimal_value(self.high, "high")
        low_price = decimal_value(self.low, "low")
        close_price = decimal_value(self.close, "close")
        volume = decimal_value(self.volume, "volume")
        if min(open_price, high_price, low_price, close_price) <= 0:
            raise DataValidationError("OHLC prices must be positive")
        if volume < 0:
            raise DataValidationError("volume must be non-negative")
        if high_price < max(open_price, close_price):
            raise DataValidationError("high must be at least open and close")
        if low_price > min(open_price, close_price):
            raise DataValidationError("low must be at most open and close")
        if high_price < low_price:
            raise DataValidationError("high must be at least low")
        if not isinstance(self.source, str) or not self.source.strip():
            raise DataValidationError("source provenance must be a non-empty string")
        if self.complete is not True:
            raise DataValidationError("forming/incomplete bars are not accepted")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "open", open_price)
        object.__setattr__(self, "high", high_price)
        object.__setattr__(self, "low", low_price)
        object.__setattr__(self, "close", close_price)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "source", self.source.strip().upper())

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MarketBar":
        """Build a bar from a provider row with the canonical field names."""

        required = ("symbol", "timestamp", "open", "high", "low", "close", "volume")
        missing = tuple(field for field in required if field not in row)
        if missing:
            raise DataValidationError(f"missing market-bar fields: {', '.join(missing)}")
        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise DataValidationError("timestamp must be an ISO-8601 timestamp") from exc
        return cls(
            symbol=row["symbol"],
            timestamp=timestamp,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            source=row.get("source", DataProvenance.HISTORICAL.value),
            complete=row.get("complete", True),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for event records."""

        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "source": self.source,
            "complete": self.complete,
        }
