"""Durable fill/ledger recovery and SQLite transactional idempotency."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from trading_agent.data import MarketBar
from trading_agent.execution import OrderRequest, OrderStatus, PaperBroker
from trading_agent.portfolio import CostModel, PortfolioLedger
from trading_agent.storage import PersistenceError, SQLiteRepository

NOW = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
NEXT = NOW + timedelta(minutes=5)
BAR = MarketBar("ABC", NEXT, 100, 110, 99, 110, 100)
ORDER = OrderRequest("buy", "ABC", "BUY", 2, NOW)


class PersistenceTests(unittest.TestCase):
    def test_events_and_transitions_are_idempotent_but_conflicts_fail(self) -> None:
        with SQLiteRepository() as repo:
            self.assertTrue(repo.append_event("one", "TEST", {"value": 1}, occurred_at=NOW, idempotency_key="key"))
            self.assertFalse(repo.append_event("one", "TEST", {"value": 1}, occurred_at=NOW, idempotency_key="key"))
            with self.assertRaises(PersistenceError):
                repo.append_event("one", "TEST", {"value": 2}, occurred_at=NOW, idempotency_key="key")
            self.assertTrue(repo.record_transition("t1", {"action": 0, "reward": 0.01, "policy_version": "v1"}, occurred_at=NOW))
            self.assertFalse(repo.record_transition("t1", {"action": 0, "reward": 0.01, "policy_version": "v1"}, occurred_at=NOW))
            with self.assertRaises(PersistenceError):
                repo.record_transition("t1", {"action": 1}, occurred_at=NOW)
            self.assertEqual(json.loads(repo.get_transition("t1")["payload"])["action"], 0)
            with self.assertRaisesRegex(ValueError, "timezone-aware"):
                repo.append_event("bad-time", "TEST", {}, occurred_at=NOW.replace(tzinfo=None))

    def test_transaction_and_nested_savepoint_roll_back_together(self) -> None:
        with SQLiteRepository() as repo:
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with repo.transaction():
                    repo.append_event("one", "TEST", {}, occurred_at=NOW)
                    with repo.transaction():
                        repo.record_transition("t1", {"reward": 0}, occurred_at=NOW)
                    raise RuntimeError("abort")
            self.assertEqual(repo.events(), ())
            self.assertIsNone(repo.get_transition("t1"))
            with repo.transaction():
                repo.append_event("after", "TEST", {}, occurred_at=NOW)
            self.assertEqual(len(repo.events()), 1)

    def test_account_and_pending_orders_survive_close_reopen_and_retry(self) -> None:
        with TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "paper.sqlite3"
            with SQLiteRepository(path) as repo:
                broker = PaperBroker(PortfolioLedger(1000), repository=repo)
                broker.submit_order(ORDER)
            with SQLiteRepository(path) as repo:
                ledger = PortfolioLedger(9999)
                restarted = PaperBroker(ledger, repository=repo)
                self.assertEqual(ledger.cash, Decimal("1000"))
                self.assertEqual(restarted.pending_orders(), (ORDER,))
                receipt, = restarted.process_bar(BAR)
                self.assertTrue(receipt.filled)
                before = ledger.export_state()
                self.assertEqual(len(repo.events(event_type="FILL")), 1)
            with SQLiteRepository(path) as repo:
                restarted = PaperBroker(PortfolioLedger(1), repository=repo)
                self.assertEqual(restarted.ledger.export_state(), before)
                self.assertEqual(restarted.submit_order(ORDER), receipt)
                self.assertEqual(restarted.process_bar(BAR), ())
                self.assertEqual(restarted.ledger.export_state(), before)
                self.assertEqual(repo.get_order("buy")["status"], OrderStatus.FILLED.value)
                fill = repo.get_fill("buy:fill")
                self.assertEqual(Decimal(json.loads(fill["payload"])["cost"]["execution_price"]), Decimal("100.00"))
                self.assertEqual(len(repo.events(event_type="FILL")), 1)

    def test_failed_fill_persistence_rolls_back_cash_fill_status_and_checkpoint(self) -> None:
        with SQLiteRepository() as repo:
            ledger = PortfolioLedger(1000)
            broker = PaperBroker(ledger, repository=repo)
            broker.submit_order(ORDER)
            before = ledger.export_state()
            with patch.object(repo, "record_snapshot", side_effect=RuntimeError("disk failure")):
                with self.assertRaisesRegex(RuntimeError, "disk failure"):
                    broker.process_bar(BAR)
            self.assertEqual(ledger.export_state(), before)
            self.assertEqual(broker.pending_orders(), (ORDER,))
            self.assertEqual(repo.get_order("buy")["status"], OrderStatus.ACCEPTED.value)
            self.assertIsNone(repo.get_fill("buy:fill"))
            self.assertEqual(repo.events(event_type="FILL"), ())
            self.assertTrue(broker.process_bar(BAR)[0].filled)
            self.assertEqual(ledger.cash, Decimal("800"))

    def test_fill_records_reject_duplicate_conflicts_and_restore_kill_latch(self) -> None:
        with SQLiteRepository() as repo:
            broker = PaperBroker(PortfolioLedger(1000), repository=repo)
            broker.submit_order(ORDER)
            fill = broker.process_bar(BAR)[0].fill
            values = dict(fill_id=fill.fill_id, client_order_id=fill.client_order_id, symbol=fill.symbol, side=fill.side, quantity=fill.quantity, occurred_at=fill.occurred_at, payload=fill.to_mapping())
            self.assertFalse(repo.record_fill(**values))
            with self.assertRaises(PersistenceError):
                repo.record_fill(**{**values, "quantity": 3})
            broker.risk_supervisor.engage_kill_switch("persisted latch")
            broker.checkpoint()
            restarted = PaperBroker(PortfolioLedger(0), repository=repo)
            result = restarted.submit_order(OrderRequest("later", "ABC", "BUY", 1, NEXT))
            self.assertIn("kill switch", result.reason)

    def test_changed_execution_configuration_cannot_silently_resume_account(self) -> None:
        with SQLiteRepository() as repo:
            PaperBroker(PortfolioLedger(1000), repository=repo)
            with self.assertRaisesRegex(ValueError, "configuration"):
                PaperBroker(PortfolioLedger(1000), repository=repo, cost_model=CostModel(slippage_bps=1))

    def test_stale_second_broker_cannot_overwrite_newer_account_state(self) -> None:
        with SQLiteRepository() as repo:
            first = PaperBroker(PortfolioLedger(1000), repository=repo)
            second = PaperBroker(PortfolioLedger(1000), repository=repo)
            first.submit_order(ORDER)
            with self.assertRaisesRegex(PersistenceError, "another broker"):
                second.submit_order(OrderRequest("stale-writer", "ABC", "BUY", 1, NOW))
            self.assertIsNone(repo.get_order("stale-writer"))
            self.assertEqual(second.pending_orders(), ())
            self.assertEqual(PaperBroker(PortfolioLedger(0), repository=repo).pending_orders(), (ORDER,))
