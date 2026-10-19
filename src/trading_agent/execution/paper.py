"""Deterministic local paper fills with transactional restart checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
import json
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ..data import MarketBar
from ..data.bars import decimal_value, utc_timestamp
from ..portfolio import CostModel, LedgerFill, PortfolioLedger
from ..portfolio.costs import normalise_side
from ..risk import RiskContext, RiskSupervisor
from ..storage import PersistenceError, SQLiteRepository


class OrderStatus(str, Enum):
    """Lifecycle states supported by this immediate-or-reject paper simulator."""

    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """A whole-share order whose decision used data through ``submitted_at``."""

    client_order_id: str
    symbol: str
    side: str
    quantity: int
    submitted_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.client_order_id, str) or not self.client_order_id.strip():
            raise ValueError("client_order_id must be non-empty")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        object.__setattr__(self, "side", normalise_side(self.side))
        object.__setattr__(self, "symbol", self.symbol.strip())
        object.__setattr__(self, "client_order_id", self.client_order_id.strip())
        object.__setattr__(self, "submitted_at", utc_timestamp(self.submitted_at))

    def to_mapping(self) -> dict[str, Any]:
        """Return a canonical request for idempotency checks and replay."""

        return {**asdict(self), "submitted_at": self.submitted_at.isoformat()}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OrderRequest":
        """Restore and validate a persisted order request."""

        return cls(**{**payload, "submitted_at": datetime.fromisoformat(payload["submitted_at"])})


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Stable acknowledgement or terminal result for one client order ID."""

    client_order_id: str
    status: OrderStatus
    reason: str = ""
    fill: LedgerFill | None = None

    @property
    def filled(self) -> bool:
        """Whether the order completed with a costed paper fill."""

        return self.status == OrderStatus.FILLED and self.fill is not None

    def to_mapping(self) -> dict[str, Any]:
        """Return a restart-safe receipt including the original fill costs."""

        return {
            "client_order_id": self.client_order_id,
            "status": self.status.value,
            "reason": self.reason,
            "fill": self.fill.to_mapping() if self.fill else None,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExecutionReceipt":
        """Restore a terminal or pending order receipt."""

        return cls(
            payload["client_order_id"],
            OrderStatus(payload["status"]),
            payload["reason"],
            LedgerFill.from_mapping(payload["fill"]) if payload["fill"] else None,
        )


class PaperBroker:
    """Next-bar-open simulator with no network, credentials, or live route.

    Orders are full-fill-or-reject: cash, risk, or participation failures do not
    create a fractional/partial fill.  Duplicate IDs return the original result
    only when the entire request matches.  One repository belongs to one paper
    account and restores its ledger, pending requests, receipts and risk latches.
    """

    _STATE_KEY = "paper_broker_v1"

    def __init__(
        self,
        ledger: PortfolioLedger,
        *,
        cost_model: CostModel | None = None,
        repository: SQLiteRepository | None = None,
        risk_supervisor: RiskSupervisor | None = None,
        participation_limit: Decimal | str | float = Decimal("1"),
    ) -> None:
        self.ledger = ledger
        self.cost_model = cost_model or CostModel()
        self.repository = repository
        self.risk_supervisor = risk_supervisor or RiskSupervisor()
        self.participation_limit = decimal_value(participation_limit, "participation_limit")
        if not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        self._requests: dict[str, OrderRequest] = {}
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._last_bars: dict[str, MarketBar] = {}
        self._sequence = 0
        self._session: str | None = None
        self._session_start_equity = self.ledger.snapshot().equity
        if repository is not None:
            state = repository.load_state(self._STATE_KEY)
            if state is not None:
                if state["configuration"] != self._configuration():
                    raise ValueError("persisted paper-account configuration does not match")
                self._restore(state)
            else:
                self.checkpoint()

    def submit_order(
        self,
        order: OrderRequest,
        *,
        risk_context: RiskContext | None = None,
    ) -> ExecutionReceipt:
        """Queue a next-bar order and durably record its acknowledgement.

        A supplied decision-time context is checked immediately.  Independently
        of that optional context, the broker rechecks risk and affordability
        using the next bar's open before every actual paper fill.
        """

        prior = self._requests.get(order.client_order_id)
        if prior is not None:
            if prior != order:
                raise ValueError(f"conflicting duplicate order: {order.client_order_id}")
            return self._receipts[order.client_order_id]
        before = self._state()
        reason = ""
        if self.risk_supervisor.kill_switch_active:
            reason = "kill switch active: " + self.risk_supervisor.kill_reason
        elif any(pending.symbol == order.symbol for pending in self.pending_orders()):
            reason = "another order for this symbol is already pending"
        elif (
            self._last_bars.get(order.symbol)
            and order.submitted_at < self._last_bars[order.symbol].timestamp
        ):
            reason = "decision timestamp precedes the latest observed bar"
        elif risk_context is not None:
            decision = self.risk_supervisor.check_order(
                order.symbol, order.side, order.quantity, context=risk_context
            )
            if not decision.approved:
                reason = decision.reason
        receipt = ExecutionReceipt(
            order.client_order_id,
            OrderStatus.REJECTED if reason else OrderStatus.ACCEPTED,
            reason,
        )
        self._requests[order.client_order_id] = order
        self._receipts[order.client_order_id] = receipt
        try:
            with self.repository.transaction() if self.repository else nullcontext():
                if self.repository:
                    self.repository.record_order(
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        submitted_at=order.submitted_at,
                        status=receipt.status.value,
                        payload=order.to_mapping(),
                    )
                    self._record_event(
                        "ORDER", order.client_order_id, receipt.to_mapping(), order.submitted_at
                    )
                self.checkpoint()
        except BaseException:
            self._restore(before)
            raise
        return receipt

    def process_bar(
        self, bar: MarketBar, *, as_of: datetime | None = None
    ) -> tuple[ExecutionReceipt, ...]:
        """Fill queued orders at a strictly later bar's open, then mark its close.

        ``as_of`` is the paper replay clock, defaulting to the bar timestamp.
        Supplying a later clock makes stale-feed lockout reproducible offline.
        Duplicate bars are no-ops, conflicting corrections/out-of-order bars
        are errors, and a persistence failure restores all in-memory state.
        """

        clock = utc_timestamp(as_of or bar.timestamp)
        previous = self._last_bars.get(bar.symbol)
        if previous is not None and bar.timestamp <= previous.timestamp:
            if bar == previous:
                return ()
            raise ValueError("out-of-order or conflicting duplicate execution bar")
        if bar.timestamp > clock:
            raise ValueError("cannot execute on a future/unavailable bar")
        before = self._state()
        results: list[ExecutionReceipt] = []
        try:
            with self.repository.transaction() if self.repository else nullcontext():
                session = bar.timestamp.astimezone(
                    ZoneInfo(self.risk_supervisor.limits.session_timezone)
                ).date().isoformat()
                if session != self._session:
                    self._session = session
                    self._session_start_equity = self.ledger.snapshot().equity
                for order in self.pending_orders():
                    if order.symbol != bar.symbol or bar.timestamp <= order.submitted_at:
                        continue
                    candidate = PortfolioLedger.from_state(self.ledger.export_state())
                    candidate.mark_to_market({bar.symbol: bar.open}, timestamp=bar.timestamp)
                    context = self._risk_context(candidate, bar, clock)
                    decision = self.risk_supervisor.check_order(
                        order.symbol, order.side, order.quantity, context=context
                    )
                    reason = "" if decision.approved else decision.reason
                    available = int(
                        (bar.volume * self.participation_limit).to_integral_value(rounding=ROUND_FLOOR)
                    )
                    if not reason and order.quantity > available:
                        reason = "insufficient bar liquidity for a full fill"
                    fill = None
                    if not reason:
                        try:
                            cost = self.cost_model.estimate(order.side, order.quantity, bar.open)
                            fill = LedgerFill(
                                fill_id=f"{order.client_order_id}:fill",
                                client_order_id=order.client_order_id,
                                symbol=order.symbol,
                                side=normalise_side(order.side),
                                quantity=order.quantity,
                                occurred_at=bar.timestamp,
                                cost=cost,
                            )
                            candidate.apply_fill(fill)
                            # Fees and adverse prices reduce equity.  Recheck
                            # the resulting weights, not merely pre-cost size.
                            decision = self.risk_supervisor.evaluate(
                                self._weights(candidate), context
                            )
                            if not decision.approved:
                                reason, fill = decision.reason, None
                        except (ValueError, ArithmeticError) as exc:
                            reason, fill = str(exc), None
                    receipt = ExecutionReceipt(
                        order.client_order_id,
                        OrderStatus.REJECTED if reason else OrderStatus.FILLED,
                        reason,
                        fill,
                    )
                    self._receipts[order.client_order_id] = receipt
                    if fill is not None:
                        self.ledger.restore_state(candidate.export_state())
                    if self.repository:
                        self.repository.update_order_status(order.client_order_id, receipt.status.value)
                        if fill:
                            self.repository.record_fill(
                                fill_id=fill.fill_id,
                                client_order_id=fill.client_order_id,
                                symbol=fill.symbol,
                                side=fill.side,
                                quantity=fill.quantity,
                                occurred_at=fill.occurred_at,
                                payload=fill.to_mapping(),
                            )
                        self._record_event(
                            "FILL" if fill else "REJECTION",
                            order.client_order_id,
                            receipt.to_mapping(),
                            bar.timestamp,
                        )
                        self._record_event("RISK", order.client_order_id, asdict(decision), clock)
                    results.append(receipt)
                self.ledger.mark_to_market({bar.symbol: bar.close}, timestamp=bar.timestamp)
                self._last_bars[bar.symbol] = bar
                closing_risk = self.risk_supervisor.evaluate(
                    self._weights(self.ledger), self._risk_context(self.ledger, bar, clock)
                )
                if self.repository:
                    self.repository.record_snapshot(
                        self.ledger.snapshot().to_mapping(), occurred_at=bar.timestamp
                    )
                    self._record_event(
                        "MARKET_RISK", f"{bar.symbol}:{bar.timestamp.isoformat()}",
                        asdict(closing_risk), clock,
                    )
                self.checkpoint()
        except BaseException:
            self._restore(before)
            raise
        return tuple(results)

    def cancel_pending(
        self, reason: str = "cancelled by paper broker"
    ) -> tuple[ExecutionReceipt, ...]:
        """Reject pending paper orders and preserve holdings; no flattening occurs."""

        before = self._state()
        receipts: list[ExecutionReceipt] = []
        try:
            with self.repository.transaction() if self.repository else nullcontext():
                for order in self.pending_orders():
                    receipt = ExecutionReceipt(order.client_order_id, OrderStatus.REJECTED, reason)
                    self._receipts[order.client_order_id] = receipt
                    if self.repository:
                        self.repository.update_order_status(order.client_order_id, receipt.status.value)
                        self._record_event(
                            "CANCELLATION", order.client_order_id,
                            receipt.to_mapping(), order.submitted_at,
                        )
                    receipts.append(receipt)
                self.checkpoint()
        except BaseException:
            self._restore(before)
            raise
        return tuple(receipts)

    def receipt(self, client_order_id: str) -> ExecutionReceipt | None:
        """Return the latest durable result, including results loaded on restart."""

        return self._receipts.get(client_order_id)

    def pending_orders(self) -> tuple[OrderRequest, ...]:
        """Return queued orders in submission order."""

        return tuple(
            order for key, order in self._requests.items()
            if self._receipts[key].status == OrderStatus.ACCEPTED
        )

    def checkpoint(self) -> None:
        """Persist control state, rejecting a stale concurrent account writer."""

        if self.repository:
            previous_sequence = self._sequence
            try:
                with self.repository.transaction():
                    current = self.repository.load_state(self._STATE_KEY)
                    persisted_sequence = current["sequence"] if current else 0
                    if persisted_sequence != self._sequence:
                        raise PersistenceError("paper account changed in another broker; reopen it")
                    self._sequence += 1
                    self.repository.save_state(self._STATE_KEY, self._state())
            except BaseException:
                self._sequence = previous_sequence
                raise

    def _record_event(
        self, event_type: str, order_id: str,
        payload: Mapping[str, Any], timestamp: datetime,
    ) -> None:
        if self.repository:
            self.repository.append_event(
                f"{event_type}:{order_id}", event_type, payload,
                aggregate_id=order_id, occurred_at=timestamp, idempotency_key=order_id,
            )

    def _risk_context(self, ledger: PortfolioLedger, bar: MarketBar, clock: datetime) -> RiskContext:
        snapshot = ledger.snapshot()
        return RiskContext(
            timestamp=clock,
            feed_timestamp=bar.timestamp,
            cash=snapshot.cash,
            equity=snapshot.equity,
            high_water_mark=snapshot.high_water_mark,
            session_start_equity=self._session_start_equity,
            positions={symbol: position.quantity for symbol, position in snapshot.positions.items()},
            marks=ledger.last_prices,
        )

    @staticmethod
    def _weights(ledger: PortfolioLedger) -> dict[str, Decimal]:
        equity = ledger.snapshot().equity
        if equity <= 0:
            return {}
        marks = ledger.last_prices
        return {
            symbol: Decimal(position.quantity) * marks[symbol] / equity
            for symbol, position in ledger.positions.items()
        }

    def _configuration(self) -> dict[str, Any]:
        def canonical(value: Any) -> Any:
            if isinstance(value, (set, frozenset)):
                return sorted(value)
            return str(value)

        values = {
            "cost_model": asdict(self.cost_model),
            "risk_limits": asdict(self.risk_supervisor.limits),
            "participation_limit": str(self.participation_limit),
        }
        return json.loads(json.dumps(values, default=canonical, sort_keys=True))

    def _state(self) -> dict[str, Any]:
        return {
            "sequence": self._sequence,
            "configuration": self._configuration(),
            "ledger": self.ledger.export_state(),
            "requests": [order.to_mapping() for order in self._requests.values()],
            "receipts": {key: receipt.to_mapping() for key, receipt in self._receipts.items()},
            "last_bars": {symbol: bar.to_mapping() for symbol, bar in self._last_bars.items()},
            "risk": self.risk_supervisor.export_state(),
            "session": self._session,
            "session_start_equity": str(self._session_start_equity),
        }

    def _restore(self, state: Mapping[str, Any]) -> None:
        self._sequence = state["sequence"]
        self.ledger.restore_state(state["ledger"])
        requests = (OrderRequest.from_mapping(row) for row in state["requests"])
        self._requests = {order.client_order_id: order for order in requests}
        self._receipts = {
            key: ExecutionReceipt.from_mapping(row) for key, row in state["receipts"].items()
        }
        self._last_bars = {
            symbol: MarketBar.from_mapping(row) for symbol, row in state["last_bars"].items()
        }
        self.risk_supervisor.restore_state(state["risk"])
        self._session = state["session"]
        self._session_start_equity = Decimal(state["session_start_equity"])
