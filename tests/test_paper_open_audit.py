"""Open-fill causality and projected-account safety regression checks."""

from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from trading_agent.data import MarketBar
from trading_agent.execution import OrderRequest, PaperBroker, PaperOpenEvent
from trading_agent.portfolio import CostModel, PortfolioLedger
from trading_agent.risk import RiskLimits, RiskSupervisor
from trading_agent.storage import SQLiteRepository


NOW = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
PREVIOUS = MarketBar("ABC", NOW, 100, 100, 100, 100, 100)
NEXT = NOW + timedelta(minutes=5)


@pytest.mark.parametrize("volume", [0, 1, 1_000_000])
def test_next_candle_volume_and_ohlc_cannot_change_an_open_fill(volume):
    broker = PaperBroker(PortfolioLedger(1000), participation_limit="0.1")
    broker.observe_bar(PREVIOUS)
    broker.submit_order(OrderRequest("buy", "ABC", "BUY", 2, NOW))
    arriving = MarketBar("ABC", NEXT, 100, 900, 1, 500, volume)
    receipt, = broker.process_bar(arriving)
    assert receipt.filled
    assert receipt.fill.execution_price == Decimal("100")
    assert receipt.fill.quantity == 2
    assert broker.ledger.cash == 800


def test_participation_uses_past_volume_and_fails_closed_without_history():
    for history in (None, replace(PREVIOUS, volume=10)):
        broker = PaperBroker(PortfolioLedger(1000), participation_limit="0.1")
        if history is not None:
            broker.observe_bar(history)
        broker.submit_order(OrderRequest("buy", "ABC", "BUY", 2, NOW))
        receipt, = broker.process_bar(MarketBar("ABC", NEXT, 100, 100, 100, 100, 1_000_000))
        assert not receipt.filled
        assert "liquidity" in receipt.reason
        assert broker.ledger.cash == 1000


def test_explicit_open_event_has_no_future_candle_fields_and_is_idempotent():
    broker = PaperBroker(PortfolioLedger(1000))
    event = PaperOpenEvent("ABC", NEXT, 100, available_volume=2)
    assert not hasattr(event, "close") and not hasattr(event, "volume")
    broker.submit_order(OrderRequest("buy", "ABC", "BUY", 2, NOW))
    assert broker.process_open(event)[0].filled
    assert broker.process_open(event) == ()
    assert broker.ledger.cash == 800


def test_post_fill_loss_guard_sees_equity_after_fees_not_opening_equity():
    broker = PaperBroker(
        PortfolioLedger(10_000),
        cost_model=CostModel(brokerage_rate="0.03"),
        risk_supervisor=RiskSupervisor(RiskLimits(daily_loss_limit="0.005")),
    )
    broker.observe_bar(PREVIOUS)
    broker.submit_order(OrderRequest("buy", "ABC", "BUY", 20, NOW))
    receipt, = broker.process_open(PaperOpenEvent("ABC", NEXT, 100))
    assert not receipt.filled
    assert "daily loss" in receipt.reason
    assert broker.ledger.cash == 10_000 and not broker.ledger.positions
    broker.submit_order(OrderRequest("smaller", "ABC", "BUY", 1, NEXT))
    assert broker.process_open(PaperOpenEvent("ABC", NEXT + timedelta(minutes=5), 100))[0].filled


def test_order_notional_limit_includes_the_actual_adverse_execution_price():
    broker = PaperBroker(
        PortfolioLedger(1000), cost_model=CostModel(slippage_bps=100),
        risk_supervisor=RiskSupervisor(RiskLimits(max_order_notional=200)),
    )
    broker.observe_bar(PREVIOUS)
    broker.submit_order(OrderRequest("buy", "ABC", "BUY", 2, NOW))
    receipt, = broker.process_open(PaperOpenEvent("ABC", NEXT, 100))
    assert "execution order notional" in receipt.reason
    assert broker.ledger.cash == 1000 and not broker.ledger.positions


def test_enclosing_transaction_restores_successful_broker_operations_in_memory():
    with SQLiteRepository() as repository:
        broker = PaperBroker(PortfolioLedger(1000), repository=repository)
        broker.observe_bar(PREVIOUS)
        before = broker.export_state()
        with pytest.raises(RuntimeError, match="outer abort"):
            with broker.atomic():
                broker.submit_order(OrderRequest("buy", "ABC", "BUY", 2, NOW))
                assert broker.process_open(PaperOpenEvent("ABC", NEXT, 100))[0].filled
                raise RuntimeError("outer abort")
        assert broker.export_state() == before
        assert repository.load_state(PaperBroker.STATE_KEY) == before
        assert repository.get_order("buy") is None


def test_order_hours_size_and_frequency_gates_are_in_the_supervisor():
    guard = RiskSupervisor(RiskLimits(
        market_open=time(9, 15), market_close=time(15, 30),
        max_order_quantity=2, max_order_notional=200,
        min_order_interval=timedelta(minutes=10),
    ))
    broker = PaperBroker(PortfolioLedger(1000), risk_supervisor=guard)
    broker.observe_bar(PREVIOUS)
    broker.submit_order(OrderRequest("large", "ABC", "BUY", 3, NOW))
    assert "quantity" in broker.process_open(PaperOpenEvent("ABC", NEXT, 100))[0].reason
    broker.submit_order(OrderRequest("one", "ABC", "BUY", 1, NEXT))
    assert broker.process_open(PaperOpenEvent("ABC", NEXT + timedelta(minutes=5), 100))[0].filled
    broker.submit_order(OrderRequest("soon", "ABC", "BUY", 1, NEXT + timedelta(minutes=5)))
    assert "interval" in broker.process_open(PaperOpenEvent("ABC", NEXT + timedelta(minutes=6), 100))[0].reason
    after_close = NOW.replace(hour=10, minute=0)  # 15:30 Asia/Kolkata
    broker.submit_order(OrderRequest("closed", "ABC", "BUY", 1, NEXT + timedelta(minutes=6)))
    assert "session is closed" in broker.process_open(PaperOpenEvent("ABC", after_close, 100))[0].reason
