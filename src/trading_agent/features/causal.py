"""Features that use only the current and preceding completed bars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, log, sqrt
from typing import Any, Iterable

from ..data import DataValidationError, MarketBar, validate_bars


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Causal feature values available immediately after a bar completes."""

    symbol: str
    timestamp: datetime
    close: float
    log_return: float | None
    range_pct: float
    volume_ratio: float | None
    rolling_volatility: float | None

    def to_mapping(self) -> dict[str, Any]:
        """Return a flat mapping suitable for persistence or model input."""

        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "close": self.close,
            "log_return": self.log_return,
            "range_pct": self.range_pct,
            "volume_ratio": self.volume_ratio,
            "rolling_volatility": self.rolling_volatility,
        }


class CausalFeatureBuilder:
    """Build deterministic rolling features without look-ahead.

    The rolling windows contain values through the current completed bar only.
    No scaler or statistic is fit across future rows; callers that need a
    train/test transform should fit that transform on the training output.
    """

    def __init__(self, lookback: int = 5) -> None:
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be at least one")
        self.lookback = lookback

    def transform(self, bars: Iterable[MarketBar]) -> tuple[FeatureRow, ...]:
        """Return one feature row per input bar in chronological order."""

        checked = validate_bars(bars)
        previous_closes: dict[str, float] = {}
        symbol_returns: dict[str, deque[float]] = {}
        symbol_volumes: dict[str, deque[float]] = {}
        rows: list[FeatureRow] = []

        for bar in checked:
            previous_close = previous_closes.get(bar.symbol)
            returns = symbol_returns.setdefault(bar.symbol, deque(maxlen=self.lookback))
            volumes = symbol_volumes.setdefault(bar.symbol, deque(maxlen=self.lookback))
            close = float(bar.close)
            current_return: float | None = None
            if previous_close is not None:
                current_return = log(close) - log(previous_close)
                returns.append(current_return)
            prior_volume_mean = sum(item / len(volumes) for item in volumes) if volumes else None
            volume_ratio = (
                float(bar.volume) / prior_volume_mean
                if prior_volume_mean is not None and prior_volume_mean > 0
                else None
            )
            volumes.append(float(bar.volume))
            volatility: float | None = None
            if len(returns) >= 2:
                mean = sum(returns) / len(returns)
                variance = sum((item - mean) ** 2 for item in returns) / len(returns)
                volatility = sqrt(variance)
            row = FeatureRow(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                close=close,
                log_return=current_return,
                range_pct=float((bar.high - bar.low) / bar.close),
                volume_ratio=volume_ratio,
                rolling_volatility=volatility,
            )
            values = (row.close, row.log_return, row.range_pct, row.volume_ratio, row.rolling_volatility)
            if any(value is not None and not isfinite(value) for value in values):
                raise DataValidationError(
                    f"non-finite feature for {bar.symbol} at {bar.timestamp.isoformat()}"
                )
            rows.append(row)
            previous_closes[bar.symbol] = close
        return tuple(rows)
