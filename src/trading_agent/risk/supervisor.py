"""Fail-closed risk rules kept independent from policy code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Mapping
from zoneinfo import ZoneInfo


def _decimal(value: Decimal | str | float, field_name: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Versioned safety thresholds for a cash-only long-only account."""

    version: str = "risk-v1"
    max_gross_exposure: Decimal | str | float = Decimal("1")
    max_symbol_weight: Decimal | str | float = Decimal("1")
    daily_loss_limit: Decimal | str | float = Decimal("0.05")
    max_drawdown: Decimal | str | float = Decimal("0.10")
    stale_after: timedelta = timedelta(minutes=10)
    permitted_symbols: frozenset[str] | set[str] | tuple[str, ...] = frozenset()
    require_positive_equity: bool = True
    session_timezone: str = "Asia/Kolkata"
    market_open: time | None = None
    market_close: time | None = None
    trading_weekdays: frozenset[int] = frozenset(range(5))
    closed_dates: frozenset[date] = frozenset()
    max_order_quantity: int | None = None
    max_order_notional: Decimal | str | float | None = None
    min_order_interval: timedelta = timedelta(0)
    max_orders_per_session: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("risk version must be non-empty")
        for name in ("max_gross_exposure", "max_symbol_weight", "daily_loss_limit", "max_drawdown"):
            value = _decimal(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
            object.__setattr__(self, name, value)
        if self.stale_after < timedelta(0):
            raise ValueError("stale_after must be non-negative")
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "permitted_symbols", frozenset(self.permitted_symbols))
        object.__setattr__(self, "trading_weekdays", frozenset(self.trading_weekdays))
        object.__setattr__(self, "closed_dates", frozenset(self.closed_dates))
        if (self.market_open is None) != (self.market_close is None):
            raise ValueError("market_open and market_close must be configured together")
        if self.market_open is not None:
            if self.market_open.tzinfo or self.market_close.tzinfo:
                raise ValueError("market hours are local wall times without timezone offsets")
            if self.market_open >= self.market_close:
                raise ValueError("market_open must be earlier than market_close")
        if not self.trading_weekdays or not self.trading_weekdays <= set(range(7)):
            raise ValueError("trading_weekdays must contain weekday numbers in [0, 6]")
        for name in ("max_order_quantity", "max_orders_per_session"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_order_notional is not None:
            amount = _decimal(self.max_order_notional, "max_order_notional")
            if amount <= 0:
                raise ValueError("max_order_notional must be positive")
            object.__setattr__(self, "max_order_notional", amount)
        if self.min_order_interval < timedelta(0):
            raise ValueError("min_order_interval must be non-negative")
        ZoneInfo(self.session_timezone)


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Immutable account/feed state supplied to a risk evaluation."""

    timestamp: datetime
    feed_timestamp: datetime | None
    cash: Decimal | str | float
    equity: Decimal | str | float
    high_water_mark: Decimal | str | float
    session_start_equity: Decimal | str | float
    positions: Mapping[str, int] = field(default_factory=dict)
    marks: Mapping[str, Decimal | str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("cash", "equity", "high_water_mark", "session_start_equity"):
            value = _decimal(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        if self.feed_timestamp is not None:
            if self.feed_timestamp.tzinfo is None or self.feed_timestamp.utcoffset() is None:
                raise ValueError("feed_timestamp must be timezone-aware")
            object.__setattr__(self, "feed_timestamp", self.feed_timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "positions", dict(self.positions))
        object.__setattr__(self, "marks", dict(self.marks))


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Auditable result of one independent risk evaluation."""

    approved: bool
    target_weights: Mapping[str, Decimal]
    reasons: tuple[str, ...]
    risk_version: str
    kill_switch_active: bool

    @property
    def reason(self) -> str:
        """Human-readable first reason, or ``approved`` for a pass."""

        return "; ".join(self.reasons) if self.reasons else "approved"


class RiskSupervisor:
    """Deterministic guard independent of policy output and broker APIs.

    The supervisor never places, edits, or cancels an order.  It only decides
    whether a requested target exposure is admissible.  The kill switch is
    latched until explicitly cleared by the controlling application.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._kill_switch = False
        self._kill_reason = ""
        self._loss_session: str | None = None
        self._daily_locked = False
        self._drawdown_locked = False
        self._last_execution: datetime | None = None
        self._execution_session: str | None = None
        self._session_executions = 0

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def engage_kill_switch(self, reason: str = "manual kill switch") -> None:
        """Latch the kill switch and block all subsequent new exposure."""

        reason = reason.strip()
        if not reason:
            raise ValueError("kill-switch reason must be non-empty")
        self._kill_switch = True
        self._kill_reason = reason

    def clear_kill_switch(self) -> None:
        """Explicitly clear the latch; this never changes configured limits."""

        self._kill_switch = False
        self._kill_reason = ""

    def evaluate(
        self,
        requested_target_weights: Mapping[str, Decimal | str | float],
        context: RiskContext,
    ) -> RiskDecision:
        """Approve a target-weight request only when every hard rule passes."""

        weights: dict[str, Decimal] = {}
        reasons: list[str] = []
        session = context.timestamp.astimezone(ZoneInfo(self.limits.session_timezone)).date().isoformat()
        if session != self._loss_session:
            self._loss_session = session
            self._daily_locked = False
        for symbol, raw_weight in requested_target_weights.items():
            if not isinstance(symbol, str) or not symbol.strip():
                reasons.append("symbol must be non-empty")
                continue
            try:
                weight = _decimal(raw_weight, f"weight[{symbol}]")
            except (ValueError, ArithmeticError):
                reasons.append(f"invalid target weight for {symbol}")
                continue
            weights[symbol] = weight
            if weight < 0:
                reasons.append(f"negative target weight for {symbol}")
            if weight > self.limits.max_symbol_weight:
                reasons.append(f"symbol cap exceeded for {symbol}")
            if self.limits.permitted_symbols and symbol not in self.limits.permitted_symbols:
                reasons.append(f"symbol not permitted: {symbol}")

        if self._kill_switch:
            reasons.append(f"kill switch active: {self._kill_reason}")
        if not self.market_is_open(context.timestamp):
            reasons.append("market session is closed")
        if context.cash < 0:
            reasons.append("cash invariant violated")
        if any(
            isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0
            for quantity in context.positions.values()
        ):
            reasons.append("invalid/short position detected")
        if self.limits.require_positive_equity and context.equity <= 0:
            reasons.append("equity is not positive")
        if context.feed_timestamp is None:
            reasons.append("missing feed timestamp")
        elif context.timestamp - context.feed_timestamp > self.limits.stale_after:
            reasons.append("stale market feed")
        elif context.feed_timestamp > context.timestamp:
            reasons.append("feed timestamp is in the future")

        gross = sum(weights.values(), Decimal("0"))
        if gross > self.limits.max_gross_exposure:
            reasons.append("gross exposure cap exceeded")
        if context.session_start_equity > 0:
            loss = Decimal("1") - context.equity / context.session_start_equity
            if loss >= self.limits.daily_loss_limit:
                self._daily_locked = True
        else:
            reasons.append("invalid session-start equity")
        if context.high_water_mark > 0:
            drawdown = Decimal("1") - context.equity / context.high_water_mark
            if drawdown >= self.limits.max_drawdown:
                self._drawdown_locked = True
        else:
            reasons.append("invalid high-water mark")
        if self._daily_locked:
            reasons.append("daily loss lockout")
        if self._drawdown_locked:
            reasons.append("drawdown lockout")

        return RiskDecision(
            approved=not reasons,
            target_weights=dict(weights) if not reasons else {},
            reasons=tuple(reasons),
            risk_version=self.limits.version,
            kill_switch_active=self._kill_switch,
        )

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        *,
        context: RiskContext,
    ) -> RiskDecision:
        """Check a direct paper order using its implied target weight.

        ``SELL`` is checked against current holdings.  ``BUY`` uses the supplied
        mark and current equity; no order is submitted by this method.
        """

        if not isinstance(side, str) or side.upper() not in ("BUY", "SELL"):
            return RiskDecision(False, {}, ("unsupported side",), self.limits.version, self._kill_switch)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return RiskDecision(False, {}, ("quantity must be a positive whole number",), self.limits.version, self._kill_switch)
        if any(
            isinstance(holding, bool) or not isinstance(holding, int) or holding < 0
            for holding in context.positions.values()
        ):
            return self.evaluate({}, context)
        current = context.positions.get(symbol, 0)
        if side.upper() == "SELL" and quantity > current:
            return RiskDecision(False, {}, ("sell quantity exceeds holding",), self.limits.version, self._kill_switch)
        if self.limits.max_order_quantity is not None and quantity > self.limits.max_order_quantity:
            return self._rejected("maximum order quantity exceeded")
        if (
            self._last_execution is not None
            and context.timestamp - self._last_execution < self.limits.min_order_interval
        ):
            return self._rejected("minimum order interval not reached")
        session = self.session_id(context.timestamp)
        if (
            self.limits.max_orders_per_session is not None
            and session == self._execution_session
            and self._session_executions >= self.limits.max_orders_per_session
        ):
            return self._rejected("maximum session order count exceeded")
        target_quantities = dict(context.positions)
        target_quantities[symbol] = current + (quantity if side.upper() == "BUY" else -quantity)
        weights: dict[str, Decimal] = {}
        for held_symbol, target_quantity in target_quantities.items():
            try:
                mark = _decimal(context.marks.get(held_symbol, 0), f"mark[{held_symbol}]")
            except (ValueError, ArithmeticError):
                mark = Decimal("0")
            if mark <= 0:
                return RiskDecision(False, {}, (f"missing/invalid mark for {held_symbol}",), self.limits.version, self._kill_switch)
            weights[held_symbol] = (
                Decimal(target_quantity) * mark / context.equity if context.equity > 0 else Decimal("0")
            )
            if held_symbol == symbol and side.upper() == "BUY" and mark * quantity > context.cash:
                return RiskDecision(False, {}, ("insufficient cash",), self.limits.version, self._kill_switch)
            if (
                held_symbol == symbol
                and self.limits.max_order_notional is not None
                and mark * quantity > self.limits.max_order_notional
            ):
                return self._rejected("maximum order notional exceeded")
        return self.evaluate(weights, context)

    def session_id(self, timestamp: datetime) -> str:
        """Return the configured exchange-local session date."""

        return timestamp.astimezone(ZoneInfo(self.limits.session_timezone)).date().isoformat()

    def market_is_open(self, timestamp: datetime) -> bool:
        """Check the independent configured schedule; no price data is used.

        Without market hours the supervisor permits an all-hours offline
        research session.  Configured sessions are half-open [open, close),
        with explicit weekdays and caller-supplied holiday dates.
        """

        if self.limits.market_open is None:
            return True
        local = timestamp.astimezone(ZoneInfo(self.limits.session_timezone))
        return (
            local.weekday() in self.limits.trading_weekdays
            and local.date() not in self.limits.closed_dates
            and self.limits.market_open <= local.time() < self.limits.market_close
        )

    def record_execution(self, timestamp: datetime) -> None:
        """Count a committed paper fill, rather than rejected policy proposals."""

        session = self.session_id(timestamp)
        if session != self._execution_session:
            self._execution_session, self._session_executions = session, 0
        self._last_execution = timestamp
        self._session_executions += 1

    def _rejected(self, reason: str) -> RiskDecision:
        return RiskDecision(False, {}, (reason,), self.limits.version, self._kill_switch)

    def export_state(self) -> dict[str, object]:
        """Return loss and kill-switch latches for restart-safe persistence."""

        return {
            "risk_version": self.limits.version,
            "kill_switch": self._kill_switch,
            "kill_reason": self._kill_reason,
            "loss_session": self._loss_session,
            "daily_locked": self._daily_locked,
            "drawdown_locked": self._drawdown_locked,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "execution_session": self._execution_session,
            "session_executions": self._session_executions,
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Restore latches only when they belong to the configured rule version."""

        if state.get("risk_version") != self.limits.version:
            raise ValueError("persisted risk version does not match configured limits")
        self._kill_switch = bool(state.get("kill_switch", False))
        self._kill_reason = str(state.get("kill_reason", ""))
        self._loss_session = state.get("loss_session")  # type: ignore[assignment]
        self._daily_locked = bool(state.get("daily_locked", False))
        self._drawdown_locked = bool(state.get("drawdown_locked", False))
        timestamp = state.get("last_execution")
        self._last_execution = datetime.fromisoformat(str(timestamp)) if timestamp else None
        self._execution_session = state.get("execution_session")  # type: ignore[assignment]
        self._session_executions = int(state.get("session_executions", 0))
