"""Hand-calculated whole-share, cost, and cash invariants."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from trading_agent.portfolio import CostModel, LedgerFill, PortfolioLedger
from trading_agent.portfolio.ledger import LedgerError

NOW = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
D = Decimal


def fill(side: str, quantity: int, price: str, *, order_id: str, model: CostModel | None = None, minutes: int = 0) -> LedgerFill:
    return LedgerFill(order_id + ":fill", order_id, "ABC", side, quantity, NOW + timedelta(minutes=minutes), (model or CostModel()).estimate(side, quantity, price))


class PortfolioTests(unittest.TestCase):
    def test_slippage_is_in_execution_price_once_and_fees_are_separate(self) -> None:
        model = CostModel(brokerage_rate="0.01", slippage_bps=100)
        cost = model.estimate("BUY", 10, 100)
        self.assertEqual(cost.execution_price, D("101.00"))
        self.assertEqual(cost.execution_cost, D("10.00"))
        self.assertEqual(cost.fees_total, D("10.10"))
        self.assertEqual(cost.cash_effect, D("-1020.10"))
        ledger = PortfolioLedger(2000)
        ledger.apply_fill(fill("BUY", 10, "100", order_id="buy", model=model))
        self.assertEqual(ledger.snapshot().equity, D("1979.90"))
        self.assertEqual(ledger.snapshot().unrealized_pnl, -cost.total_cost)
        snapshot = ledger.mark_to_market({"ABC": 100})
        self.assertEqual(snapshot.cash, D("979.90"))
        self.assertEqual(snapshot.equity, D("1979.90"))
        self.assertEqual(snapshot.equity, D("2000") - cost.total_cost)

    def test_round_trip_reconciles_basis_fees_cash_and_realized_profit(self) -> None:
        ledger = PortfolioLedger(1000)
        model = CostModel(brokerage_rate="0.01")
        ledger.apply_fill(fill("BUY", 5, "100", order_id="buy", model=model))
        self.assertEqual(ledger.positions["ABC"].average_cost, D("101"))
        ledger.apply_fill(fill("SELL", 2, "110", order_id="sell-part", model=model, minutes=5))
        self.assertEqual(ledger.positions["ABC"].quantity, 3)
        self.assertEqual(ledger.realized_pnl, D("15.80"))
        ledger.apply_fill(fill("SELL", 3, "110", order_id="sell-final", model=model, minutes=10))
        self.assertEqual(ledger.cash, D("1039.50"))
        self.assertEqual(ledger.realized_pnl, D("39.50"))
        self.assertEqual(ledger.fees_paid, D("10.50"))
        self.assertEqual(ledger.positions, {})

    def test_all_fee_components_round_to_paise_and_correct_side(self) -> None:
        model = CostModel(brokerage_rate="0.01", transaction_fee_rate="0.001", buy_tax_rate="0.002", sell_tax_rate="0.003", gst_rate="0.18", stamp_duty_buy_rate="0.0001")
        buy = model.estimate("BUY", 1, 100)
        sell = model.estimate("SELL", 1, 100)
        self.assertEqual(dict(buy.fee_components), {"brokerage": D("1.00"), "transaction_fee": D("0.10"), "tax": D("0.20"), "gst": D("0.20"), "stamp_duty": D("0.01")})
        self.assertEqual(buy.fees_total, D("1.51"))
        self.assertEqual(sell.fees_total, D("1.60"))

    def test_impossible_fills_are_atomic_and_quantity_is_integral(self) -> None:
        ledger = PortfolioLedger(100)
        before = ledger.export_state()
        for bad_fill in (fill("BUY", 2, "100", order_id="expensive"), fill("SELL", 1, "100", order_id="short")):
            with self.assertRaises(LedgerError):
                ledger.apply_fill(bad_fill)
            self.assertEqual(ledger.export_state(), before)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(quantity=invalid), self.assertRaises(ValueError):
                CostModel().estimate("BUY", invalid, 100)
        self.assertEqual(ledger.target_quantity("ABC", 1, 101), 0)
        self.assertEqual(ledger.target_quantity("ABC", 1, 30), 3)

    def test_duplicate_fills_are_idempotent_and_conflicts_fail(self) -> None:
        ledger = PortfolioLedger(1000)
        buy = fill("BUY", 2, "100", order_id="same")
        ledger.apply_fill(buy)
        before = ledger.export_state()
        self.assertEqual(ledger.apply_fill(buy).cash, D("800"))
        self.assertEqual(ledger.export_state(), before)
        with self.assertRaisesRegex(LedgerError, "conflicting"):
            ledger.apply_fill(replace(buy, symbol="XYZ"))
        restored = PortfolioLedger.from_state(before)
        self.assertEqual(restored.apply_fill(buy).cash, D("800"))

    def test_drawdown_and_zero_cash_restore_are_valid(self) -> None:
        ledger = PortfolioLedger(100)
        ledger.apply_fill(fill("BUY", 1, "100", order_id="all-cash"))
        restored = PortfolioLedger.from_state(ledger.export_state())
        snapshot = restored.mark_to_market({"ABC": 90})
        self.assertEqual(snapshot.equity, D("90"))
        self.assertEqual(snapshot.high_water_mark, D("100"))
        self.assertEqual(snapshot.drawdown, D("0.1"))
