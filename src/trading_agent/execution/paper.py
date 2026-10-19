"""Deterministic local paper fills with transactional restart checkpoints."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
import json
from typing import Any, Iterator, Mapping
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
class PaperOpenEvent:
    """Information observable at an eligible paper execution event.

    There is deliberately no high, low, close or completed-candle volume.
    ``available_volume`` is optional explicit *event-time* liquidity.  When it
    is absent, the broker conservatively uses the preceding observed candle's
    volume as a versioned proxy, never the arriving candle's future volume.
    """

    symbol: str
    timestamp: datetime
    price: Decimal | str | float
    available_volume: Decimal | str | float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("open-event symbol must be non-empty")
        price = decimal_value(self.price, "open-event price")
        if price <= 0:
            raise ValueError("open-event price must be positive")
        object.__setattr__(self, "symbol", self.symbol.strip())
        object.__setattr__(self, "timestamp", utc_timestamp(self.timestamp))
        object.__setattr__(self, "price", price)
        if self.available_volume is not None:
            volume = decimal_value(self.available_volume, "event-time available volume")
            if volume < 0:
                raise ValueError("event-time available volume must be non-negative")
            object.__setattr__(self, "available_volume", volume)

    def to_mapping(self) -> dict[str, Any]:
        """Return the narrow event in a stable, JSON-friendly form."""

        return {
            "symbol": self.symbol, "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "available_volume": str(self.available_volume) if self.available_volume is not None else None,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperOpenEvent":
        """Restore an event without broadening its available information."""

        return cls(**{**payload, "timestamp": datetime.fromisoformat(payload["timestamp"])})


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

    STATE_KEY = "paper_broker_v1"
    EXECUTION_VERSION = "causal-open-v2"
    _STATE_KEY = STATE_KEY

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
        self._last_opens: dict[str, PaperOpenEvent] = {}
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

    @property
    def revision(self) -> int:
        """The committed broker checkpoint revision used by an enclosing env."""

        return self._sequence

    @property
    def session_start_equity(self) -> Decimal:
        """The broker's actual exchange-session opening equity baseline."""

        return self._session_start_equity

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Roll back broker memory and SQLite together, including outer work.

        An environment can include transitions and its cursor checkpoint in
        this boundary.  Nested broker operations use SQLite savepoints; their
        successful in-memory changes are still restored if the outer step fails.
        """

        before = self._state()
        try:
            with self.repository.transaction() if self.repository else nullcontext():
                yield
        except BaseException:
            self._restore(before)
            raise

    def export_state(self) -> dict[str, Any]:
        """Return a detached, validated replay/checkpoint view of broker state."""

        return deepcopy(self._state())

    def plan_target_order(
        self,
        client_order_id: str,
        target_weight: Decimal | str | float,
        *,
        decision_at: datetime,
        event: PaperOpenEvent,
    ) -> OrderRequest | None:
        """Size a precommitted target using only an eligible open event.

        The target is chosen at ``decision_at``; whole-share sizing is deferred
        until the eligible price is known.  The ledger and versioned cost model
        provide all sizing inputs.  This is planning only, and never moves cash.
        """

        if event.timestamp <= utc_timestamp(decision_at):
            raise ValueError("a target requires a strictly later eligible open")
        weight = decimal_value(target_weight, "target_weight")
        if not 0 <= weight <= 1:
            raise ValueError("target_weight must lie in [0, 1]")
        candidate = PortfolioLedger.from_state(self.ledger.export_state())
        candidate.mark_to_market({event.symbol: event.price})
        equity = candidate.snapshot().equity
        current = candidate.position_quantity(event.symbol)
        desired = candidate.target_quantity(event.symbol, weight, event.price)
        if desired == current:
            return None
        side = "BUY" if desired > current else "SELL"
        quantity = abs(desired - current)
        if side == "BUY":
            # Find a cost-aware affordable target.  Quoting does not charge a
            # fee; only applying the eventual LedgerFill does that once.
            low, high = 0, quantity
            while low < high:
                amount = (low + high + 1) // 2
                quote = self.cost_model.estimate(side, amount, event.price)
                post_equity = equity - quote.total_cost
                fits = (
                    -quote.cash_effect <= candidate.cash
                    and post_equity > 0
                    and Decimal(current + amount) * event.price <= weight * post_equity
                )
                if fits:
                    low = amount
                else:
                    high = amount - 1
            quantity = low
        else:
            # Sale fees can make the retained weight larger than requested.
            while quantity < current:
                quote = self.cost_model.estimate(side, quantity, event.price)
                if Decimal(current - quantity) * event.price <= weight * (equity - quote.total_cost):
                    break
                quantity += 1
        if not quantity:
            return None
        return OrderRequest(client_order_id, event.symbol, side, quantity, decision_at)

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
        """Execute a narrow next-open event, then observe the completed candle.

        Open execution sees only ``symbol/timestamp/open``.  The arriving
        candle's high, low, close and volume become available only afterwards.
        Prime the broker with :meth:`observe_bar` for historical participation,
        or provide explicit eligible liquidity through :meth:`process_open`.
        """

        previous = self._last_bars.get(bar.symbol)
        if previous is not None and bar.timestamp <= previous.timestamp:
            if bar == previous:
                return ()
            raise ValueError("out-of-order or conflicting duplicate execution bar")
        with self.atomic():
            results = self.process_open(
                PaperOpenEvent(bar.symbol, bar.timestamp, bar.open), as_of=as_of
            )
            self.observe_bar(bar, as_of=as_of)
        return results

    def process_open(
        self, event: PaperOpenEvent, *, as_of: datetime | None = None
    ) -> tuple[ExecutionReceipt, ...]:
        """Fill queued paper orders with event-time information only.

        Participation defaults to the *previously observed* candle's volume.
        With neither that history nor explicit event liquidity, fills fail
        closed.  Post-fill safety uses the actual candidate account context.
        """

        clock = utc_timestamp(as_of or event.timestamp)
        previous_open = self._last_opens.get(event.symbol)
        if previous_open is not None and event.timestamp <= previous_open.timestamp:
            if event == previous_open:
                return ()
            raise ValueError("out-of-order or conflicting duplicate open event")
        prior_bar = self._last_bars.get(event.symbol)
        if prior_bar is not None and event.timestamp <= prior_bar.timestamp:
            raise ValueError("open event must follow the last completed candle")
        if event.timestamp > clock:
            raise ValueError("cannot execute on a future/unavailable bar")
        observed_volume = event.available_volume
        if observed_volume is None and prior_bar is not None:
            observed_volume = prior_bar.volume
        available = int(
            ((observed_volume or Decimal("0")) * self.participation_limit)
            .to_integral_value(rounding=ROUND_FLOOR)
        )
        results: list[ExecutionReceipt] = []
        with self.atomic():
            self._begin_session(event.timestamp)
            self.ledger.mark_to_market({event.symbol: event.price}, timestamp=event.timestamp)
            for order in self.pending_orders():
                if order.symbol != event.symbol or event.timestamp <= order.submitted_at:
                    continue
                candidate = PortfolioLedger.from_state(self.ledger.export_state())
                context = self._risk_context(candidate, event, clock)
                decision = self.risk_supervisor.check_order(
                    order.symbol, order.side, order.quantity, context=context
                )
                reason = "" if decision.approved else decision.reason
                if not reason and order.quantity > available:
                    reason = (
                        "missing previously observed/event liquidity"
                        if observed_volume is None
                        else "insufficient observed liquidity for a full fill"
                    )
                fill = None
                if not reason:
                    try:
                        cost = self.cost_model.estimate(order.side, order.quantity, event.price)
                        fill = LedgerFill(
                            f"{order.client_order_id}:fill", order.client_order_id,
                            order.symbol, normalise_side(order.side), order.quantity,
                            event.timestamp, cost,
                        )
                        candidate.apply_fill(fill)
                        decision = self.risk_supervisor.evaluate(
                            self._weights(candidate), self._risk_context(candidate, event, clock)
                        )
                        if not decision.approved:
                            reason, fill = decision.reason, None
                    except (ValueError, ArithmeticError) as exc:
                        reason, fill = str(exc), None
                receipt = ExecutionReceipt(
                    order.client_order_id,
                    OrderStatus.REJECTED if reason else OrderStatus.FILLED,
                    reason, fill,
                )
                self._receipts[order.client_order_id] = receipt
                if fill:
                    self.ledger.restore_state(candidate.export_state())
                    self.risk_supervisor.record_execution(event.timestamp)
                if self.repository:
                    self.repository.update_order_status(order.client_order_id, receipt.status.value)
                    if fill:
                        self.repository.record_fill(
                            fill_id=fill.fill_id, client_order_id=fill.client_order_id,
                            symbol=fill.symbol, side=fill.side, quantity=fill.quantity,
                            occurred_at=fill.occurred_at, payload=fill.to_mapping(),
                        )
                    self._record_event(
                        "FILL" if fill else "REJECTION", order.client_order_id,
                        receipt.to_mapping(), event.timestamp,
                    )
                    self._record_event("RISK", order.client_order_id, asdict(decision), clock)
                results.append(receipt)
            self._last_opens[event.symbol] = event
            self.risk_supervisor.evaluate(
                self._weights(self.ledger), self._risk_context(self.ledger, event, clock)
            )
            self.checkpoint()
        return tuple(results)

    def observe_bar(self, bar: MarketBar, *, as_of: datetime | None = None) -> None:
        """Observe a completed candle without retroactively placing any fill."""

        clock = utc_timestamp(as_of or bar.timestamp)
        previous = self._last_bars.get(bar.symbol)
        if previous is not None and bar.timestamp <= previous.timestamp:
            if bar == previous:
                return
            raise ValueError("out-of-order or conflicting duplicate execution bar")
        if bar.timestamp > clock:
            raise ValueError("cannot observe a future/unavailable bar")
        with self.atomic():
            self._begin_session(bar.timestamp)
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

    def _begin_session(self, timestamp: datetime) -> None:
        session = self.risk_supervisor.session_id(timestamp)
        if session != self._session:
            self._session = session
            self._session_start_equity = self.ledger.snapshot().equity

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

    def _risk_context(
        self, ledger: PortfolioLedger, bar: MarketBar | PaperOpenEvent, clock: datetime
    ) -> RiskContext:
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
            "execution_version": self.EXECUTION_VERSION,
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
            "last_opens": {symbol: event.to_mapping() for symbol, event in self._last_opens.items()},
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
        self._last_opens = {
            symbol: PaperOpenEvent.from_mapping(row) for symbol, row in state["last_opens"].items()
        }
        self.risk_supervisor.restore_state(state["risk"])
        self._session = state["session"]
        self._session_start_equity = Decimal(state["session_start_equity"])
