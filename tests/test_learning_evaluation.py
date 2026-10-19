from datetime import datetime, timedelta, timezone
import math
import statistics

import pytest

from trading_agent.evaluation import (
    CashBaseline,
    RandomBaseline,
    WalkForwardSplit,
    chronological_splits,
    compute_metrics,
    evaluate_baselines,
    evaluate_policy,
    session_returns,
    summarize_walk_forward,
    walk_forward_evaluate,
)


def test_daily_metrics_use_session_closes_not_intraday_bar_annualisation():
    equity = [100, 110, 105, 90, 99, 108.9]
    sessions = ["a", "a", "b", "b", "c", "c"]
    expected = (0.1, 90 / 110 - 1, 108.9 / 90 - 1)
    metrics = compute_metrics(
        equity, session_ids=sessions, total_fees=2, turnover_notional=200,
        exposures=[0, 1, 0.5, 0.5, 0], trade_pnls=[10, -5, 2], gross_profit=20,
    )
    assert session_returns(equity, sessions) == pytest.approx(expected)
    assert metrics.net_return == pytest.approx(0.089)
    assert metrics.max_drawdown == pytest.approx(1 - 90 / 110)
    assert metrics.daily_sharpe == pytest.approx(statistics.mean(expected) / statistics.stdev(expected) * math.sqrt(252))
    assert metrics.expected_shortfall == pytest.approx(1 - 90 / 110)
    assert metrics.sessions == 3 and metrics.trades == 3
    assert metrics.turnover == 2 and metrics.total_fees == 2
    assert metrics.win_rate == pytest.approx(2 / 3) and metrics.expectancy == pytest.approx(7 / 3)
    assert metrics.cost_fraction_of_gross_profit == 0.1
    assert metrics.time_in_cash == 0.4


def test_cash_and_missing_sampling_have_no_infinite_ratios():
    metrics = compute_metrics([100, 100, 100], session_ids=["a", "b", "c"])
    assert metrics.net_return == metrics.max_drawdown == metrics.expected_shortfall == 0
    assert metrics.daily_sharpe is metrics.daily_sortino is metrics.calmar_ratio is None
    assert metrics.win_rate is metrics.expectancy is None
    without_sessions = compute_metrics([100, 90])
    assert without_sessions.daily_sharpe is without_sessions.expected_shortfall is None
    assert without_sessions.sessions == 0


def test_initial_capital_snapshot_is_not_an_extra_trading_session():
    assert session_returns([100, 110, 99], ["initial", "day1", "day2"]) == pytest.approx((0.1, -0.1))
    metrics = compute_metrics([100, 110, 99], session_ids=["initial", "day1", "day2"])
    assert metrics.sessions == 2
    assert metrics.mean_exposure is metrics.time_in_cash is None


@pytest.mark.parametrize("curve", [[100], [100, 0], [100, float("nan")], [100, -1]])
def test_invalid_equity_rejected(curve):
    with pytest.raises(ValueError):
        compute_metrics(curve)


def test_session_alignment_and_nonchronological_data_rejected():
    with pytest.raises(ValueError):
        compute_metrics([100, 110, 120], session_ids=["a", "b", "a"])
    with pytest.raises(ValueError):
        compute_metrics([100, 110], timestamps=["2026-01-02", "2026-01-01"])
    with pytest.raises(ValueError):
        compute_metrics([100, 110], session_ids=["a"])


def test_purged_walk_forward_splits_are_half_open_and_holdout_is_untouched():
    splits = chronological_splits(100, train_size=20, validation_size=10, test_size=10, purge=2, embargo=3, final_holdout_size=10)
    assert splits[0] == WalkForwardSplit(0, 20, 25, 35, 40, 50)
    assert splits[1].train_start == 0 and splits[1].train_end == 30
    assert splits[-1].test_end == 90
    assert all(first.test_end <= second.test_start for first, second in zip(splits, splits[1:]))
    assert all(split.train_end + 5 == split.validation_start and split.validation_end + 5 == split.test_start for split in splits)
    rolling = chronological_splits(100, train_size=20, validation_size=10, test_size=10, expanding=False)
    assert rolling[1].train_start == 10 and rolling[1].train_end == 30
    with pytest.raises(ValueError):
        chronological_splits(20, train_size=20, validation_size=5, test_size=5)
    with pytest.raises(ValueError):
        chronological_splits(100, train_size=20, validation_size=10, test_size=10, step_size=5)


def market_bars(count):
    pytest.importorskip("gymnasium")
    from trading_agent.env import MarketBar

    start = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    return [MarketBar(start + timedelta(days=index), 100, 102, 99, 101 if index % 2 else 100, 10) for index in range(count)]


def test_policy_and_baselines_have_identical_windows_costs_and_seed():
    from trading_agent.env import SimplePaperExecution, TradingEnv

    bars = market_bars(10)
    factory = lambda: TradingEnv(bars, execution=SimplePaperExecution(0.001, 5))
    cash = evaluate_policy(factory(), CashBaseline())
    reports = evaluate_baselines(factory, seed=19)
    again = evaluate_baselines(factory, seed=19)
    assert set(reports) == {"cash", "rule", "random"}
    assert reports == again
    assert reports["cash"] == cash
    assert cash.metrics.net_return == 0 and cash.metrics.total_fees == 0
    assert reports["random"].metrics.total_fees > 0
    assert all(report.timestamps == cash.timestamps for report in reports.values())
    first, second = RandomBaseline(5, seed=9), RandomBaseline(5, seed=9)
    assert [first(()) for _ in range(20)] == [second(()) for _ in range(20)]
    with pytest.raises(ValueError, match="action"):
        evaluate_policy(factory(), lambda observation: 0.5)


def test_fitter_never_receives_test_or_final_holdout_and_multiple_seeds_reported():
    from trading_agent.env import TradingEnv

    bars = market_bars(40)
    splits = chronological_splits(40, train_size=10, validation_size=5, test_size=5, final_holdout_size=5)
    fitting_inputs = []

    def fit(training, validation, seed):
        fitting_inputs.append((training, validation, seed))
        return CashBaseline()

    reports = walk_forward_evaluate(bars, splits, fit, lambda rows: TradingEnv(rows), seeds=(7, 11))
    assert len(reports) == len(splits) * 2
    for report, (training, validation, seed) in zip(reports, fitting_inputs):
        assert seed == report.seed
        assert training[-1].timestamp < validation[0].timestamp
        assert validation[-1].timestamp < report.candidate.timestamps[0]
        assert report.candidate.timestamps == tuple(bar.timestamp for bar in bars[report.split.test])
        assert max(bar.timestamp for bar in training + validation) < bars[-5].timestamp
    summary = summarize_walk_forward(reports)
    assert summary["seeds"] == 2 and summary["windows"] == len(splits)
    assert summary["median_net_return"] == summary["net_return_stdev"] == 0
