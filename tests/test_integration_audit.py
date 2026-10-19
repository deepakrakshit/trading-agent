from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_agent.data import MarketBar
from trading_agent.paper_env import PaperTradingEnv
from trading_agent.portfolio import CostModel


def test_brokerage_cap_sebi_ipft_and_gst_are_separate_reconciled_components():
    cost = CostModel(
        brokerage_rate="0.0003", brokerage_cap="20", transaction_fee_rate="0.0000307",
        sebi_fee_rate="0.000001", ipft_fee_rate="0.000000001", gst_rate="0.18",
    ).estimate("BUY", 10000, 100)
    fees = dict(cost.fee_components)
    assert fees["brokerage"] == Decimal("20.00")
    assert fees["sebi_fee"] == Decimal("1.00")
    assert fees["ipft_fee"] == Decimal("0.00")
    assert fees["gst"] == Decimal("9.31")
    assert cost.fees_total == Decimal("61.01")
    assert cost.cash_effect == Decimal("-1000061.01")


def test_bar_start_labels_have_explicit_completed_availability_and_next_open():
    start = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
    bars = tuple(MarketBar("X", start + timedelta(minutes=5 * i), 100, 101, 99, 100, 10000) for i in range(3))
    env = PaperTradingEnv(bars, terminal_liquidation=False)
    env.reset()
    _, _, _, _, info = env.step(1)
    assert info["decision_available_at"] == bars[0].timestamp + timedelta(minutes=5)
    assert info["fill_timestamp"] >= info["decision_available_at"]
    assert info["reward_available_at"] == bars[1].timestamp + timedelta(minutes=5)


def test_overlapping_candle_cannot_fill_before_decision_is_complete():
    start = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
    bars = tuple(MarketBar("X", start + timedelta(minutes=i), 100, 101, 99, 100, 10000) for i in range(3))
    env = PaperTradingEnv(bars)
    env.reset()
    with pytest.raises(ValueError, match="completion"):
        env.step(1)
