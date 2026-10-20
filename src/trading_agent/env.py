"""A causal, single-instrument, paper-trading Gymnasium environment.

The environment makes one deliberately small contract explicit:

* an action is a discrete *target exposure* rather than an order;
* the decision is made after the current bar has completed;
* a rebalance is filled at the following bar's open;
* reward is the change in log equity after the simulated fill and costs.

The default execution model is local and deterministic.  A later ledger or
paper-execution implementation can be injected through :class:`PaperExecution`
without changing the observation/action contract.  This module has no broker
credentials or live-order integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import warnings
from typing import Any, Protocol, runtime_checkable

try:  # Gymnasium is an optional runtime dependency for source-only installs.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Keep evaluation/registry importable without Gymnasium.
    gym = None
    spaces = None

try:
    import numpy as _np
except ImportError:  # pragma: no cover - local source-only fallback.
    _np = None

_EnvBase = gym.Env if gym is not None else object


@dataclass(frozen=True, slots=True)
class MarketBar:
    """One completed OHLCV bar.

    ``timestamp`` is intentionally kept as an opaque value.  A provider can
    use a timezone-aware ``datetime``, an integer epoch, or another immutable
    timestamp type.  The environment only preserves it in ``info``.
    """

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    completed: bool = True

    def __post_init__(self) -> None:
        if self.timestamp is None or not self.completed:
            raise ValueError("bars must have timestamps and be completed")
        prices = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in prices):
            raise ValueError("bar prices and volume must be finite")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("bar prices must be positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar OHLC values are inconsistent")
        if self.volume < 0:
            raise ValueError("bar volume cannot be negative")


Candle = MarketBar


def _coerce_bar(value: MarketBar | Mapping[str, Any] | Sequence[Any]) -> MarketBar:
    if isinstance(value, MarketBar):
        return value
    if isinstance(value, Mapping):
        timestamp = value.get("timestamp", value.get("time", value.get("datetime")))
        try:
            return MarketBar(
                timestamp=timestamp,
                open=float(value["open"]),
                high=float(value["high"]),
                low=float(value["low"]),
                close=float(value["close"]),
                volume=float(value.get("volume", 0.0)),
                completed=bool(value.get("completed", value.get("complete", True))),
            )
        except KeyError as exc:
            raise ValueError(f"bar is missing required field {exc.args[0]!r}") from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) not in (5, 6):
            raise ValueError("sequence bars must be [timestamp, open, high, low, close, volume]")
        timestamp, open_, high, low, close = value[:5]
        volume = value[5] if len(value) == 6 else 0.0
        return MarketBar(timestamp, float(open_), float(high), float(low), float(close), float(volume))
    raise TypeError(f"unsupported bar type: {type(value).__name__}")


@runtime_checkable
class CausalFeatureProvider(Protocol):
    """Build features from bars available at a completed-bar decision time.

    The environment passes a prefix ending at the current completed bar.  A
    provider must not infer or access values after ``index``.
    """

    def __call__(self, completed_bars: Sequence[MarketBar], index: int) -> Sequence[float]:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """State returned by an injected paper execution implementation."""

    cash: float
    shares: int
    fees: float
    turnover: float
    buy_price: float | None = None
    sell_price: float | None = None


@dataclass(frozen=True, slots=True)
class FillPoint:
    """An eligible next-open event, containing no future bar high/low/close."""

    timestamp: Any
    price: float


@runtime_checkable
class PaperExecution(Protocol):
    """Minimal dependency-injection boundary for a local paper fill model."""

    def rebalance(
        self,
        *,
        cash: float,
        shares: int,
        target_weight: float,
        equity: float,
        reference: FillPoint,
    ) -> ExecutionResult:
        """Return post-fill cash/holdings for the next completed-bar interval."""


@dataclass(frozen=True, slots=True)
class SimplePaperExecution:
    """Deterministic whole-share, cash-only next-open fill model.

    ``slippage_bps`` is applied once to the fill price.  ``fee_rate`` is a
    separate proportional fee on notional and is also applied once.
    """

    fee_rate: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.fee_rate < 1 or not 0 <= self.slippage_bps < 10_000:
            raise ValueError("fees/slippage must be finite, non-negative, and below 100%")

    @property
    def _slippage_fraction(self) -> float:
        return self.slippage_bps / 10_000.0

    def rebalance(
        self,
        *,
        cash: float,
        shares: int,
        target_weight: float,
        equity: float,
        reference: FillPoint,
    ) -> ExecutionResult:
        if cash < -1e-8 or shares < 0 or equity <= 0:
            raise ValueError("execution state must be cash-only, long-only, and solvent")
        if not 0.0 <= target_weight <= 1.0:
            raise ValueError("target_weight must be between zero and one")

        buy_price = reference.price * (1.0 + self._slippage_fraction)
        sell_price = reference.price * (1.0 - self._slippage_fraction)
        target_shares = math.floor((target_weight * equity) / buy_price)
        next_cash = float(cash)
        next_shares = int(shares)
        fees = 0.0
        turnover = 0.0
        last_buy_price: float | None = None
        last_sell_price: float | None = None

        sell_quantity = max(0, next_shares - target_shares)
        if sell_quantity:
            notional = sell_quantity * sell_price
            fee = notional * self.fee_rate
            next_cash += notional - fee
            next_shares -= sell_quantity
            fees += fee
            turnover += notional
            last_sell_price = sell_price

        buy_quantity = max(0, target_shares - next_shares)
        if buy_quantity:
            affordable = math.floor(next_cash / (buy_price * (1.0 + self.fee_rate)))
            buy_quantity = min(buy_quantity, max(0, affordable))
        if buy_quantity:
            notional = buy_quantity * buy_price
            fee = notional * self.fee_rate
            next_cash -= notional + fee
            next_shares += buy_quantity
            fees += fee
            turnover += notional
            last_buy_price = buy_price

        if next_cash < -1e-7:
            raise AssertionError("paper execution created negative cash")
        return ExecutionResult(
            cash=max(0.0, next_cash),
            shares=next_shares,
            fees=fees,
            turnover=turnover,
            buy_price=last_buy_price,
            sell_price=last_sell_price,
        )


def default_causal_features(bars: Sequence[MarketBar], index: int) -> tuple[float, ...]:
    """Return a small feature vector using only the supplied completed prefix."""

    if not bars or not 0 <= index < len(bars):
        raise IndexError("feature index is outside the completed-bar prefix")
    current = bars[index]
    previous = bars[index - 1] if index else current
    window = bars[max(0, index - 4) : index + 1]
    average_volume = sum(bar.volume for bar in window) / max(1, len(window))
    previous_close = previous.close
    return (
        current.open / current.close - 1.0,
        current.high / current.close - 1.0,
        current.low / current.close - 1.0,
        current.close / previous_close - 1.0,
        current.volume / average_volume - 1.0 if average_volume else 0.0,
    )


def _feature_array(values: Sequence[float]) -> Any:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("features must be finite")
    if _np is None:
        return result
    return _np.asarray(result, dtype=_np.float32)


class SingleInstrumentTradingEnv(_EnvBase):
    """Gymnasium-compatible cash-only, long-only trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: Sequence[MarketBar | Mapping[str, Any] | Sequence[Any]],
        *,
        initial_cash: float = 10_000.0,
        target_exposures: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        feature_provider: CausalFeatureProvider = default_causal_features,
        execution: PaperExecution | None = None,
        start_index: int = 0,
        terminal_liquidation: bool = False,
    ) -> None:
        if gym is None or _np is None:
            raise ImportError("the trading environment requires gymnasium and numpy")
        if initial_cash <= 0 or not math.isfinite(initial_cash):
            raise ValueError("initial_cash must be finite and positive")
        if len(bars) < 2:
            raise ValueError("at least two completed bars are required for next-bar fills")
        self.bars = tuple(_coerce_bar(bar) for bar in bars)
        for previous, current in zip(self.bars, self.bars[1:]):
            if previous.timestamp >= current.timestamp:
                raise ValueError("bar timestamps must be strictly chronological")
        if not 0 <= start_index < len(self.bars) - 1:
            raise ValueError("start_index must leave at least one fill bar")
        exposures = tuple(float(value) for value in target_exposures)
        if not exposures or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in exposures):
            raise ValueError("target_exposures must contain values in [0, 1]")
        if tuple(sorted(set(exposures))) != exposures:
            raise ValueError("target_exposures must be sorted and unique")
        self.initial_cash = float(initial_cash)
        self.target_exposures = exposures
        self.feature_provider = feature_provider
        self.execution = execution or SimplePaperExecution()
        self.start_index = int(start_index)
        self.end_index = len(self.bars) - 1
        self.terminal_liquidation = bool(terminal_liquidation)

        initial_features = tuple(float(value) for value in feature_provider(self.bars[:1], 0))
        if not initial_features:
            raise ValueError("feature_provider must return at least one feature")
        if not all(math.isfinite(value) for value in initial_features):
            raise ValueError("feature_provider returned non-finite features")
        self.feature_dim = len(initial_features)
        self.action_space = spaces.Discrete(len(self.target_exposures))
        self.observation_space = spaces.Box(
            low=-_np.inf,
            high=_np.inf,
            shape=(self.feature_dim + 4,),
            dtype=_np.float32,
        )

        self._index = self.start_index
        self._cash = self.initial_cash
        self._shares = 0
        self._equity = self.initial_cash
        self._high_water_mark = self.initial_cash
        self._done = False
        self._last_info: dict[str, Any] = {}

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def shares(self) -> int:
        return self._shares

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def drawdown(self) -> float:
        return 1.0 - self._equity / self._high_water_mark if self._high_water_mark else 0.0

    def _observation(self) -> Any:
        # Passing a prefix, rather than the full dataset, makes accidental
        # future-bar access impossible for ordinary feature providers.
        completed = self.bars[: self._index + 1]
        features = tuple(float(value) for value in self.feature_provider(completed, self._index))
        if len(features) != self.feature_dim:
            raise ValueError("feature_provider changed its output dimension")
        price = self.bars[self._index].close
        marked_equity = self._cash + self._shares * price
        exposure = (self._shares * price / marked_equity) if marked_equity > 0 else 0.0
        cash_fraction = self._cash / marked_equity if marked_equity > 0 else 0.0
        progress_denominator = max(1, self.end_index - self.start_index)
        progress = (self._index - self.start_index) / progress_denominator
        return _feature_array(features + (cash_fraction, exposure, self.drawdown, progress))

    def _info(self, **extra: Any) -> dict[str, Any]:
        bar = self.bars[self._index]
        result = {
            "index": self._index,
            "timestamp": bar.timestamp,
            "cash": self._cash,
            "shares": self._shares,
            "equity": self._equity,
            "drawdown": self.drawdown,
            "exposure": self._shares * bar.close / self._equity if self._equity > 0 else 0.0,
        }
        result.update(extra)
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._index = self.start_index
        self._cash = self.initial_cash
        self._shares = 0
        self._equity = self.initial_cash
        self._high_water_mark = self.initial_cash
        self._done = False
        self._last_info = self._info(seed=seed)
        return self._observation(), dict(self._last_info)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        if isinstance(action, bool) or not self.action_space.contains(action):
            raise ValueError(f"invalid discrete action: {action!r}")
        action = int(action)
        current_bar = self.bars[self._index]
        next_index = self._index + 1
        next_bar = self.bars[next_index]
        before_equity = self._cash + self._shares * current_bar.close
        if before_equity <= 0 or not math.isfinite(before_equity):
            self._done = True
            return self._observation(), -1.0, True, False, self._info(invalid_equity=True)

        target_weight = self.target_exposures[action]
        # A configured terminal closeout is precommitted and occurs at the
        # final bar's open.  It never sells at an already-known past open after
        # observing the final close, nor opens then immediately closes a trade.
        liquidating = self.terminal_liquidation and next_index == self.end_index
        executed_target = 0.0 if liquidating else target_weight
        fill = self.execution.rebalance(
            cash=self._cash,
            shares=self._shares,
            target_weight=executed_target,
            equity=before_equity,
            reference=FillPoint(next_bar.timestamp, next_bar.open),
        )
        if (
            not all(math.isfinite(value) for value in (fill.cash, fill.fees, fill.turnover))
            or fill.cash < -1e-7
            or fill.shares < 0
            or isinstance(fill.shares, bool)
            or int(fill.shares) != fill.shares
            or fill.fees < 0
            or fill.turnover < 0
        ):
            raise ValueError("injected execution returned an invalid long-only state")
        self._cash = float(fill.cash)
        self._shares = int(fill.shares)
        self._index = next_index
        after_equity = self._cash + self._shares * next_bar.close
        if after_equity <= 0 or not math.isfinite(after_equity):
            self._equity = max(0.0, after_equity)
            self._done = True
            reward = -1.0
            terminated = True
        else:
            self._equity = after_equity
            self._high_water_mark = max(self._high_water_mark, after_equity)
            reward = math.log(after_equity / before_equity)
            terminated = self._index >= self.end_index

        self._done = bool(terminated)
        info = self._info(
            action=action,
            target_exposure=target_weight,
            executed_target_exposure=executed_target,
            terminal_liquidation=liquidating,
            fill_index=next_index,
            fill_timestamp=next_bar.timestamp,
            fill_open=next_bar.open,
            previous_equity=before_equity,
            realized_fees=float(fill.fees),
            turnover=float(fill.turnover),
            buy_price=fill.buy_price,
            sell_price=fill.sell_price,
            decision_index=next_index - 1,
        )
        self._last_info = info
        return self._observation(), float(reward), bool(terminated), False, dict(info)

    def render(self) -> dict[str, Any]:
        """Return a serialisable snapshot instead of performing I/O."""

        return self._info()

    def close(self) -> None:
        return None


class TradingEnv(SingleInstrumentTradingEnv):
    """Deprecated compatibility environment.

    The supported financial simulation is
    :class:`trading_agent.paper_env.PaperTradingEnv`, which uses the canonical
    Decimal ledger, broker, risk supervisor, persistence, and task-specific
    termination contract.  This small injected-execution environment remains
    available only for backwards-compatible primitive/causality tests.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "trading_agent.env.TradingEnv is a compatibility fixture; use "
            "trading_agent.paper_env.PaperTradingEnv for financial training/evaluation",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "Candle",
    "CausalFeatureProvider",
    "ExecutionResult",
    "FillPoint",
    "MarketBar",
    "PaperExecution",
    "SimplePaperExecution",
    "SingleInstrumentTradingEnv",
    "TradingEnv",
    "default_causal_features",
]
