"""Next-bar-only paper fills and independent execution-time guards."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from trading_agent.data import MarketBar
from trading_agent.execution import OrderRequest, OrderStatus, PaperBroker
from trading_agent.portfolio import CostModel, PortfolioLedger
from trading_agent.risk import RiskLimits, RiskSupervisor

NOW = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)


def bar(minutes: int, *, open_price=100, close=100, volume=100) -> MarketBar:
    return MarketBar("ABC", NOW + timedelta(minutes=minutes), open_price, max(open_price, close) + 1, min(open_price, close) - 1, close, volume)


def order(order_id: str = "buy", *, side="BUY", quantity=2, minutes=0) -> OrderRequest:
    return OrderRequest(order_id, "ABC", side, quantity, NOW + timedelta(minutes=minutes))


class PaperExecutionTests(unittest.TestCase):
    def test_next_bar_open_gap_is_used_and_same_bar_cannot_fill(self) -> None:
        ledger = PortfolioLedger(1000)
        broker = PaperBroker(ledger)
        request = order()
        self.assertEqual(broker.submit_order(request).status, OrderStatus.ACCEPTED)
        self.assertEqual(broker.process_bar(bar(0, close=105)), ())
        result, = broker.process_bar(bar(5, open_price=120, close=125))
        self.assertTrue(result.filled)
        self.assertEqual(result.fill.reference_price, Decimal("120"))
        self.assertEqual(ledger.cash, Decimal("760"))
        self.assertEqual(ledger.snapshot().equity, Decimal("1010"))
        self.assertEqual(broker.process_bar(bar(5, open_price=120, close=125)), ())
        self.assertEqual(broker.submit_order(request), result)
        self.assertEqual(ledger.cash, Decimal("760"))

    def test_conflicting_retries_and_backwards_or_corrected_bars_fail(self) -> None:
        broker = PaperBroker(PortfolioLedger(1000))
        request = order()
        broker.submit_order(request)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate order"):
            broker.submit_order(replace(request, quantity=3))
        broker.process_bar(bar(5))
        for invalid in (bar(0), bar(5, close=101)):
            with self.assertRaisesRegex(ValueError, "out-of-order or conflicting"):
                broker.process_bar(invalid)

    def test_insufficient_cash_after_charges_rejects_without_side_effects(self) -> None:
        ledger = PortfolioLedger(200)
        broker = PaperBroker(ledger, cost_model=CostModel(brokerage_rate="0.01"))
        broker.submit_order(order())
        receipt, = broker.process_bar(bar(5))
        self.assertEqual(receipt.status, OrderStatus.REJECTED)
        self.assertIn("insufficient cash", receipt.reason)
        self.assertEqual(ledger.cash, Decimal("200"))
        self.assertEqual(ledger.positions, {})

    def test_pending_duplicate_targets_and_no_liquidity_do_not_create_fills(self) -> None:
        ledger = PortfolioLedger(1000)
        broker = PaperBroker(ledger, participation_limit="0.1")
        broker.submit_order(order())
        self.assertEqual(broker.submit_order(order("another")).status, OrderStatus.REJECTED)
        receipt, = broker.process_bar(bar(5, volume=10))
        self.assertIn("liquidity", receipt.reason)
        self.assertEqual(ledger.cash, Decimal("1000"))

    def test_cost_adjusted_exposure_cannot_exceed_symbol_cap(self) -> None:
        ledger = PortfolioLedger(1000)
        broker = PaperBroker(
            ledger,
            cost_model=CostModel(brokerage_rate="0.01"),
            risk_supervisor=RiskSupervisor(RiskLimits(max_symbol_weight="0.5")),
        )
        broker.submit_order(order(quantity=5))
        receipt, = broker.process_bar(bar(5))
        self.assertIn("symbol cap", receipt.reason)
        self.assertEqual(ledger.cash, Decimal("1000"))

    def test_loss_latch_is_monitored_on_bars_without_a_policy_order(self) -> None:
        broker = PaperBroker(PortfolioLedger(1000))
        broker.submit_order(order(quantity=5))
        broker.process_bar(bar(5))
        broker.process_bar(bar(10, open_price=80, close=80))
        broker.process_bar(bar(15, open_price=100, close=100))
        broker.submit_order(order("after-recovery", quantity=1, minutes=15))
        receipt, = broker.process_bar(bar(20))
        self.assertIn("loss lockout", receipt.reason)
        self.assertEqual(broker.ledger.position_quantity("ABC"), 5)

    def test_kill_switch_and_stale_clock_recheck_previously_accepted_orders(self) -> None:
        for kill in (True, False):
            ledger = PortfolioLedger(1000)
            guard = RiskSupervisor(RiskLimits(stale_after=timedelta(minutes=1)))
            broker = PaperBroker(ledger, risk_supervisor=guard)
            broker.submit_order(order())
            if kill:
                guard.engage_kill_switch("after acceptance")
            receipts = broker.process_bar(bar(5), as_of=NOW + timedelta(minutes=5 if kill else 7))
            self.assertEqual(receipts[0].status, OrderStatus.REJECTED)
            self.assertIn("kill switch" if kill else "stale", receipts[0].reason)
            self.assertEqual(ledger.positions, {})
            self.assertEqual(broker.pending_orders(), ())

    def test_cancel_policy_preserves_positions_and_blocks_no_future_data(self) -> None:
        broker = PaperBroker(PortfolioLedger(1000))
        broker.submit_order(order())
        with self.assertRaisesRegex(ValueError, "future/unavailable"):
            broker.process_bar(bar(5), as_of=NOW)
        cancelled, = broker.cancel_pending()
        self.assertEqual(cancelled.status, OrderStatus.REJECTED)
        self.assertEqual(broker.process_bar(bar(5)), ())
        self.assertEqual(broker.ledger.cash, Decimal("1000"))
