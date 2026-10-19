"""Fail-closed risk rules kept independent from policy code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

        if side.upper() not in ("BUY", "SELL"):
            return RiskDecision(False, {}, ("unsupported side",), self.limits.version, self._kill_switch)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return RiskDecision(False, {}, ("quantity must be a positive whole number",), self.limits.version, self._kill_switch)
        current = context.positions.get(symbol, 0)
        if side.upper() == "SELL" and quantity > current:
            return RiskDecision(False, {}, ("sell quantity exceeds holding",), self.limits.version, self._kill_switch)
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
        return self.evaluate(weights, context)

    def export_state(self) -> dict[str, object]:
        """Return loss and kill-switch latches for restart-safe persistence."""

        return {
            "risk_version": self.limits.version,
            "kill_switch": self._kill_switch,
            "kill_reason": self._kill_reason,
            "loss_session": self._loss_session,
            "daily_locked": self._daily_locked,
            "drawdown_locked": self._drawdown_locked,
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
