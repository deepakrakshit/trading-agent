"""A local paper broker with next-bar fills and idempotent order IDs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from ..data import MarketBar
from ..portfolio import CostModel, LedgerFill, PortfolioLedger
from ..risk import RiskContext, RiskSupervisor
from ..storage import SQLiteRepository


class OrderStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """A whole-share market order request for the paper broker."""

    client_order_id: str
    symbol: str
    side: str
    quantity: int
    submitted_at: datetime

    def __post_init__(self) -> None:
        if not self.client_order_id.strip() or not self.symbol.strip():
            raise ValueError("client_order_id and symbol must be non-empty")
        if self.side.upper() not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None:
            raise ValueError("submitted_at must be timezone-aware")
        object.__setattr__(self, "side", self.side.upper())
        object.__setattr__(self, "submitted_at", self.submitted_at.astimezone(timezone.utc))


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Stable acknowledgement or terminal result for an order."""

    client_order_id: str
    status: OrderStatus
    reason: str = ""
    fill: LedgerFill | None = None

    @property
    def filled(self) -> bool:
        return self.status == OrderStatus.FILLED and self.fill is not None


class PaperBroker:
    """Offline-only broker simulator; it exposes no live-order operation."""

    def __init__(
        self,
        ledger: PortfolioLedger,
        *,
        cost_model: CostModel | None = None,
        repository: SQLiteRepository | None = None,
        risk_supervisor: RiskSupervisor | None = None,
    ) -> None:
        self.ledger = ledger
        self.cost_model = cost_model or CostModel()
        self.repository = repository
        self.risk_supervisor = risk_supervisor
        self._pending: dict[str, OrderRequest] = {}
        self._receipts: dict[str, ExecutionReceipt] = {}

    def submit_order(
        self,
        order: OrderRequest,
        *,
        risk_context: RiskContext | None = None,
    ) -> ExecutionReceipt:
        """Accept a valid order for a future bar, or reject it deterministically."""

        previous = self._receipts.get(order.client_order_id)
        if previous is not None:
            return previous
        if self.repository is not None and self.repository.get_order(order.client_order_id) is not None:
            raise ValueError(f"order ID already exists outside this broker: {order.client_order_id}")
        if self.risk_supervisor is not None and risk_context is not None:
            decision = self.risk_supervisor.check_order(
                order.symbol, order.side, order.quantity, context=risk_context
            )
            if not decision.approved:
                receipt = ExecutionReceipt(order.client_order_id, OrderStatus.REJECTED, decision.reason)
                self._receipts[order.client_order_id] = receipt
                self._persist_order(order, receipt)
                return receipt
        receipt = ExecutionReceipt(order.client_order_id, OrderStatus.ACCEPTED)
        self._pending[order.client_order_id] = order
        self._receipts[order.client_order_id] = receipt
        self._persist_order(order, receipt)
        return receipt

    def process_bar(self, bar: MarketBar) -> tuple[ExecutionReceipt, ...]:
        """Fill eligible orders for ``bar`` at its open, never its close.

        An order submitted at the same timestamp as a completed bar becomes
        eligible only on a later bar.  Orders for other symbols remain queued.
        """

        receipts: list[ExecutionReceipt] = []
        for order_id, order in tuple(self._pending.items()):
            if order.symbol != bar.symbol or bar.timestamp <= order.submitted_at:
                continue
            try:
                cost = self.cost_model.estimate(order.side, order.quantity, bar.open)
                fill = LedgerFill(
                    fill_id=f"{order.client_order_id}:fill",
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,  # type: ignore[arg-type]
                    quantity=order.quantity,
                    occurred_at=bar.timestamp,
                    cost=cost,
                )
                self.ledger.apply_fill(fill)
            except (ValueError, ArithmeticError) as exc:
                receipt = ExecutionReceipt(order_id, OrderStatus.REJECTED, str(exc))
                self._receipts[order_id] = receipt
                self._pending.pop(order_id, None)
                self._persist_status(order, receipt)
            else:
                receipt = ExecutionReceipt(order_id, OrderStatus.FILLED, fill=fill)
                self._receipts[order_id] = receipt
                self._pending.pop(order_id, None)
                self._persist_fill(order, receipt)
            receipts.append(receipt)
        return tuple(receipts)

    def cancel_pending(self) -> tuple[ExecutionReceipt, ...]:
        """Reject queued orders without touching holdings."""

        receipts: list[ExecutionReceipt] = []
        for order_id, order in tuple(self._pending.items()):
            receipt = ExecutionReceipt(order_id, OrderStatus.REJECTED, "cancelled by paper broker")
            self._receipts[order_id] = receipt
            self._pending.pop(order_id, None)
            self._persist_status(order, receipt)
            receipts.append(receipt)
        return tuple(receipts)

    def receipt(self, client_order_id: str) -> ExecutionReceipt | None:
        """Return the last in-process receipt for an order ID."""

        return self._receipts.get(client_order_id)

    def pending_orders(self) -> tuple[OrderRequest, ...]:
        """Return queued orders in submission order."""

        return tuple(self._pending.values())

    def _persist_order(self, order: OrderRequest, receipt: ExecutionReceipt) -> None:
        if self.repository is None:
            return
        self.repository.record_order(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            submitted_at=order.submitted_at,
            status=receipt.status.value,
            payload={"order": order.__dict__ if hasattr(order, "__dict__") else {
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "submitted_at": order.submitted_at.isoformat(),
            }, "reason": receipt.reason},
        )
        self.repository.append_event(
            event_id=f"order:{order.client_order_id}",
            event_type="ORDER",
            aggregate_id=order.client_order_id,
            idempotency_key=order.client_order_id,
            occurred_at=order.submitted_at,
            payload={"status": receipt.status.value, "reason": receipt.reason},
        )

    def _persist_status(self, order: OrderRequest, receipt: ExecutionReceipt) -> None:
        if self.repository is None:
            return
        self.repository.update_order_status(order.client_order_id, receipt.status.value)
        self.repository.append_event(
            event_id=f"order-status:{order.client_order_id}:{receipt.status.value}",
            event_type="ORDER_STATUS",
            aggregate_id=order.client_order_id,
            idempotency_key=f"{order.client_order_id}:{receipt.status.value}",
            occurred_at=order.submitted_at,
            payload={"status": receipt.status.value, "reason": receipt.reason},
        )

    def _persist_fill(self, order: OrderRequest, receipt: ExecutionReceipt) -> None:
        if self.repository is None or receipt.fill is None:
            return
        self.repository.update_order_status(order.client_order_id, receipt.status.value)
        self.repository.record_fill(
            fill_id=receipt.fill.fill_id,
            client_order_id=receipt.fill.client_order_id,
            symbol=receipt.fill.symbol,
            side=receipt.fill.side,
            quantity=receipt.fill.quantity,
            occurred_at=receipt.fill.occurred_at,
            payload=receipt.fill.to_mapping(),
        )
        snapshot = self.ledger.snapshot()
        self.repository.record_snapshot(snapshot.to_mapping(), occurred_at=snapshot.timestamp)
        self.repository.append_event(
            event_id=f"fill:{receipt.fill.fill_id}",
            event_type="FILL",
            aggregate_id=receipt.fill.client_order_id,
            idempotency_key=receipt.fill.fill_id,
            occurred_at=receipt.fill.occurred_at,
            payload=receipt.fill.to_mapping(),
        )
