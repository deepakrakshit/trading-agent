"""Deterministic independent kill/loss/staleness/exposure checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from trading_agent.risk import RiskContext, RiskLimits, RiskSupervisor

NOW = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)


def context(**overrides) -> RiskContext:
    values = dict(timestamp=NOW, feed_timestamp=NOW, cash=1000, equity=1000, high_water_mark=1000, session_start_equity=1000, marks={"ABC": 100})
    values.update(overrides)
    return RiskContext(**values)


class RiskTests(unittest.TestCase):
    def test_kill_switch_blocks_even_cash_request_until_explicitly_cleared(self) -> None:
        guard = RiskSupervisor()
        guard.engage_kill_switch("operator test")
        decision = guard.evaluate({"ABC": 0}, context())
        self.assertFalse(decision.approved)
        self.assertIn("kill switch", decision.reason)
        restored = RiskSupervisor()
        restored.restore_state(guard.export_state())
        self.assertFalse(restored.evaluate({"ABC": 0.5}, context()).approved)
        guard.clear_kill_switch()
        self.assertTrue(guard.evaluate({"ABC": 0.5}, context()).approved)

    def test_stale_missing_and_future_feeds_fail_closed(self) -> None:
        guard = RiskSupervisor(RiskLimits(stale_after=timedelta(minutes=5)))
        for timestamp in (None, NOW - timedelta(minutes=6), NOW + timedelta(seconds=1)):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(guard.evaluate({"ABC": 0.5}, context(feed_timestamp=timestamp)).approved)
        self.assertTrue(guard.evaluate({"ABC": 0.5}, context(feed_timestamp=NOW - timedelta(minutes=5))).approved)

    def test_daily_loss_and_drawdown_latch_at_the_threshold(self) -> None:
        daily = RiskSupervisor(RiskLimits(daily_loss_limit="0.05", max_drawdown="0.5"))
        self.assertIn("daily loss", daily.evaluate({"ABC": 0.5}, context(equity=950)).reason)
        self.assertIn("daily loss", daily.evaluate({"ABC": 0.5}, context()).reason)
        tomorrow = NOW + timedelta(days=1)
        self.assertTrue(daily.evaluate({"ABC": 0.5}, context(timestamp=tomorrow, feed_timestamp=tomorrow)).approved)
        drawdown = RiskSupervisor(RiskLimits(daily_loss_limit="0.5", max_drawdown="0.1"))
        self.assertIn("drawdown", drawdown.evaluate({"ABC": 0.5}, context(equity=900)).reason)
        self.assertIn("drawdown", drawdown.evaluate({"ABC": 0.5}, context()).reason)

    def test_exposure_short_cash_and_symbol_rules_are_policy_independent(self) -> None:
        guard = RiskSupervisor(RiskLimits(max_symbol_weight="0.6", max_gross_exposure="0.8", permitted_symbols={"ABC", "XYZ"}))
        for weights in ({"ABC": -0.1}, {"ABC": 0.7}, {"ABC": 0.5, "XYZ": 0.5}, {"OTHER": 0.1}, {"ABC": float("nan")}):
            with self.subTest(weights=weights):
                self.assertFalse(guard.evaluate(weights, context()).approved)
        self.assertFalse(guard.evaluate({"ABC": 0.1}, context(cash=-1)).approved)
        self.assertFalse(guard.check_order("ABC", "BUY", 11, context=context()).approved)
        self.assertFalse(guard.check_order("ABC", "SELL", 1, context=context()).approved)
        self.assertFalse(guard.check_order("ABC", "BUY", 1, context=context(marks={})).approved)
        self.assertTrue(guard.check_order("ABC", "BUY", 5, context=context()).approved)
