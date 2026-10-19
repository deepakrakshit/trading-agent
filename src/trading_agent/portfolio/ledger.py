"""Append-friendly cash-only, long-only portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Mapping

from ..data.bars import decimal_value, utc_timestamp
from .costs import CostBreakdown, Side, money


class LedgerError(ValueError):
    """Raised when a fill would violate account invariants."""


@dataclass(frozen=True, slots=True)
class Position:
    """Whole-share position and all-in cost basis."""

    quantity: int
    cost_basis: Decimal

    @property
    def average_cost(self) -> Decimal:
        """Average all-in cost per share, or zero for an empty position."""

        return self.cost_basis / self.quantity if self.quantity else Decimal("0")


@dataclass(frozen=True, slots=True)
class LedgerFill:
    """A fully costed fill that can be applied exactly once to a ledger."""

    fill_id: str
    client_order_id: str
    symbol: str
    side: Side
    quantity: int
    occurred_at: datetime
    cost: CostBreakdown

    def __post_init__(self) -> None:
        if not self.fill_id.strip() or not self.client_order_id.strip():
            raise ValueError("fill_id and client_order_id must be non-empty")
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if self.cost.side != self.side or self.cost.quantity != self.quantity:
            raise ValueError("fill side/quantity must match its cost breakdown")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")

    @property
    def reference_price(self) -> Decimal:
        return self.cost.reference_price

    @property
    def execution_price(self) -> Decimal:
        return self.cost.execution_price

    @property
    def fees(self) -> Decimal:
        return self.cost.fees_total

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable mapping for SQLite event persistence."""

        return {
            "fill_id": self.fill_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat(),
            "cost": self.cost.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Auditable account state at a mark or fill."""

    cash: Decimal
    equity: Decimal
    positions: Mapping[str, Position]
    realized_pnl: Decimal
    fees_paid: Decimal
    high_water_mark: Decimal
    drawdown: Decimal
    timestamp: datetime | None

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly account snapshot."""

        return {
            "cash": str(self.cash),
            "equity": str(self.equity),
            "positions": {
                symbol: {"quantity": position.quantity, "cost_basis": str(position.cost_basis)}
                for symbol, position in self.positions.items()
            },
            "realized_pnl": str(self.realized_pnl),
            "fees_paid": str(self.fees_paid),
            "high_water_mark": str(self.high_water_mark),
            "drawdown": str(self.drawdown),
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat()
            if self.timestamp is not None
            else None,
        }


class PortfolioLedger:
    """Single source of truth for cash, whole-share holdings, and equity.

    The ledger is deliberately unaware of policies and providers.  It accepts
    only already-costed paper fills, applies each fill ID once, and rejects
    negative cash or short positions before state changes become visible.
    """

    def __init__(self, initial_cash: Decimal | str | float) -> None:
        cash = money(initial_cash)
        if cash < 0:
            raise ValueError("initial_cash must be finite and non-negative")
        self._cash = cash
        self._positions: dict[str, Position] = {}
        self._last_prices: dict[str, Decimal] = {}
        self._applied_fill_ids: set[str] = set()
        self._fill_records: dict[str, dict[str, Any]] = {}
        self._applied_order_ids: set[str] = set()
        self._realized_pnl = Decimal("0")
        self._fees_paid = Decimal("0")
        self._high_water_mark = cash
        self._last_timestamp: datetime | None = None

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    @property
    def fees_paid(self) -> Decimal:
        return self._fees_paid

    @property
    def high_water_mark(self) -> Decimal:
        return self._high_water_mark

    @property
    def applied_fill_ids(self) -> frozenset[str]:
        return frozenset(self._applied_fill_ids)

    def position_quantity(self, symbol: str) -> int:
        return self._positions.get(symbol, Position(0, Decimal("0"))).quantity

    def mark_to_market(
        self,
        prices: Mapping[str, Decimal | str | float],
        *,
        timestamp: datetime | None = None,
    ) -> AccountSnapshot:
        """Update marks and return equity; every held symbol needs a price."""

        updated_prices = dict(self._last_prices)
        for symbol, raw_price in prices.items():
            price = decimal_value(raw_price, f"mark[{symbol}]")
            if price <= 0:
                raise LedgerError(f"mark price for {symbol} must be positive and finite")
            updated_prices[symbol] = price
        missing = [symbol for symbol in self._positions if symbol not in updated_prices]
        if missing:
            raise LedgerError(f"missing mark price for: {', '.join(sorted(missing))}")
        if timestamp is not None:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise LedgerError("ledger timestamp must be timezone-aware")
            timestamp = utc_timestamp(timestamp)
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                raise LedgerError("mark timestamp cannot move backwards")
        self._last_prices = updated_prices
        if timestamp is not None:
            self._last_timestamp = timestamp
        equity = self._equity()
        if equity > self._high_water_mark:
            self._high_water_mark = equity
        return self.snapshot()

    def apply_fill(self, fill: LedgerFill) -> AccountSnapshot:
        """Apply a fill once and return the resulting marked account state."""

        if fill.fill_id in self._applied_fill_ids:
            if self._fill_records.get(fill.fill_id) != fill.to_mapping():
                raise LedgerError(f"conflicting duplicate fill: {fill.fill_id}")
            return self.snapshot()
        if fill.client_order_id in self._applied_order_ids:
            raise LedgerError(f"another fill already applied for order {fill.client_order_id}")
        position = self._positions.get(fill.symbol, Position(0, Decimal("0")))
        notional = fill.cost.execution_notional
        fees = fill.fees
        if self._last_timestamp is not None and fill.occurred_at < self._last_timestamp:
            raise LedgerError("fill timestamp cannot move backwards")
        if notional < 0 or fees < 0:
            raise LedgerError("fill notional and fees must be non-negative")
        cash_after = self._cash + fill.cost.cash_effect
        if cash_after < 0:
            raise LedgerError("insufficient cash after fees")
        if fill.side == "BUY":
            self._positions[fill.symbol] = Position(
                quantity=position.quantity + fill.quantity,
                cost_basis=position.cost_basis + notional + fees,
            )
        else:
            if position.quantity < fill.quantity:
                raise LedgerError("cannot sell more shares than held")
            released_basis = (
                position.cost_basis
                if fill.quantity == position.quantity
                else money(position.average_cost * fill.quantity)
            )
            self._realized_pnl += notional - fees - released_basis
            remaining = position.quantity - fill.quantity
            if remaining:
                self._positions[fill.symbol] = Position(
                    quantity=remaining,
                    cost_basis=position.cost_basis - released_basis,
                )
            else:
                del self._positions[fill.symbol]
        self._cash = cash_after
        self._fees_paid += fees
        self._last_prices[fill.symbol] = fill.execution_price
        self._applied_fill_ids.add(fill.fill_id)
        self._fill_records[fill.fill_id] = fill.to_mapping()
        self._applied_order_ids.add(fill.client_order_id)
        self._last_timestamp = fill.occurred_at.astimezone(timezone.utc)
        equity = self._equity()
        if equity > self._high_water_mark:
            self._high_water_mark = equity
        return self.snapshot()

    def target_quantity(
        self,
        symbol: str,
        target_weight: Decimal | str | float,
        price: Decimal | str | float,
        *,
        cost_reserve_fraction: Decimal | str | float = Decimal("0"),
    ) -> int:
        """Return an affordable whole-share target for one symbol.

        This sizing helper is conservative: the reserve is applied before
        flooring, and the result can legally be zero for an expensive share.
        """

        weight = Decimal(str(target_weight))
        mark = Decimal(str(price))
        reserve = Decimal(str(cost_reserve_fraction))
        if not weight.is_finite() or not Decimal("0") <= weight <= Decimal("1"):
            raise LedgerError("target_weight must be between zero and one")
        if not mark.is_finite() or mark <= 0:
            raise LedgerError("price must be positive and finite")
        if not reserve.is_finite() or reserve < 0:
            raise LedgerError("cost_reserve_fraction must be non-negative")
        equity = self.mark_to_market({symbol: mark}).equity
        raw = (equity * weight) / (mark * (Decimal("1") + reserve))
        desired = max(0, int(raw.to_integral_value(rounding=ROUND_FLOOR)))
        affordable_extra = int(
            (self._cash / (mark * (Decimal("1") + reserve))).to_integral_value(rounding=ROUND_FLOOR)
        )
        return min(desired, self.position_quantity(symbol) + affordable_extra)

    def snapshot(self) -> AccountSnapshot:
        """Return a detached snapshot suitable for persistence."""

        equity = self._equity()
        drawdown = (
            max(Decimal("0"), Decimal("1") - equity / self._high_water_mark)
            if self._high_water_mark > 0
            else Decimal("1")
        )
        return AccountSnapshot(
            cash=self._cash,
            equity=equity,
            positions=dict(self._positions),
            realized_pnl=self._realized_pnl,
            fees_paid=self._fees_paid,
            high_water_mark=self._high_water_mark,
            drawdown=drawdown,
            timestamp=self._last_timestamp,
        )

    def export_state(self) -> dict[str, Any]:
        """Export restart-safe state, including idempotency keys."""

        return {
            "cash": str(self._cash),
            "positions": {
                symbol: {"quantity": p.quantity, "cost_basis": str(p.cost_basis)}
                for symbol, p in self._positions.items()
            },
            "last_prices": {symbol: str(price) for symbol, price in self._last_prices.items()},
            "applied_fill_ids": sorted(self._applied_fill_ids),
            "fill_records": self._fill_records.copy(),
            "applied_order_ids": sorted(self._applied_order_ids),
            "realized_pnl": str(self._realized_pnl),
            "fees_paid": str(self._fees_paid),
            "high_water_mark": str(self._high_water_mark),
            "last_timestamp": self._last_timestamp.isoformat() if self._last_timestamp else None,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PortfolioLedger":
        """Restore a ledger exported by :meth:`export_state`."""

        ledger = cls(state["cash"])
        ledger._positions = {
            symbol: Position(values["quantity"], decimal_value(values["cost_basis"], "cost_basis"))
            for symbol, values in state.get("positions", {}).items()
        }
        ledger._last_prices = {
            symbol: decimal_value(price, f"mark[{symbol}]")
            for symbol, price in state.get("last_prices", {}).items()
        }
        ledger._applied_fill_ids = set(state.get("applied_fill_ids", ()))
        ledger._fill_records = dict(state.get("fill_records", {}))
        ledger._applied_order_ids = set(state.get("applied_order_ids", ()))
        ledger._realized_pnl = decimal_value(state.get("realized_pnl", "0"), "realized_pnl")
        ledger._fees_paid = decimal_value(state.get("fees_paid", "0"), "fees_paid")
        ledger._high_water_mark = decimal_value(state.get("high_water_mark", state["cash"]), "high_water_mark")
        raw_timestamp = state.get("last_timestamp")
        ledger._last_timestamp = (
            utc_timestamp(datetime.fromisoformat(raw_timestamp))
            if raw_timestamp
            else None
        )
        invalid_positions = any(
            isinstance(position.quantity, bool)
            or not isinstance(position.quantity, int)
            or position.quantity <= 0
            or position.cost_basis < 0
            for position in ledger._positions.values()
        )
        if (
            ledger._cash < 0
            or ledger._fees_paid < 0
            or ledger._high_water_mark < 0
            or invalid_positions
            or any(price <= 0 for price in ledger._last_prices.values())
            or any(symbol not in ledger._last_prices for symbol in ledger._positions)
        ):
            raise LedgerError("persisted ledger violates cash/position invariants")
        return ledger

    def _equity(self) -> Decimal:
        return money(self._cash + sum(
            (position.quantity * self._last_prices[symbol] for symbol, position in self._positions.items()),
            Decimal("0"),
        ))
