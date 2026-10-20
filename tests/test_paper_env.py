"""Canonical Gym accounting, causal fills, risk, and training interfaces."""

from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from trading_agent.data import MarketBar
from trading_agent.evaluation import CashBaseline, evaluate_baselines, evaluate_policy
from trading_agent.learning import DQNConfig, DQNTrainer
from trading_agent.paper_env import PaperTradingEnv
from trading_agent.portfolio import CostModel
from trading_agent.risk import RiskLimits


START = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
RESEARCH_LIMITS = RiskLimits(daily_loss_limit="0.99", max_drawdown="0.99")


def candles(prices=(100, 100, 110, 110, 112), *, start=START, volume=1000):
    return tuple(
        MarketBar("ABC", start + timedelta(minutes=5 * index), price, price + 1, price - 1, price, volume)
        for index, price in enumerate(prices)
    )


def test_fees_slippage_rewards_and_terminal_liquidation_use_one_canonical_ledger():
    env = PaperTradingEnv(
        candles((100, 100, 110, 110)), initial_cash=1000,
        target_exposures=(0, 1), risk_limits=RESEARCH_LIMITS,
        cost_model=CostModel(brokerage_rate="0.01", slippage_bps=100),
        policy_version="frozen-policy-v7",
    )
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    rewards = []
    _, reward, terminated, _, first = env.step(1)
    rewards.append(reward)
    assert not terminated
    assert first["buy_price"] == 101
    assert first["shares"] == 9 and first["cash"] == 81.91
    assert first["realized_fees"] == 9.09 and first["execution_cost"] == 9
    assert first["equity"] == 981.91
    assert reward == pytest.approx(math.log(981.91 / 1000))
    _, reward, _, _, second = env.step(1)
    rewards.append(reward)
    assert second["realized_fees"] == 0 and second["turnover"] == 0
    _, reward, terminated, truncated, last = env.step(1)
    rewards.append(reward)
    assert terminated and not truncated and last["terminal_liquidation"]
    assert last["termination_reason"] == "dataset_end" and last["episode_horizon_end"]
    assert last["sell_price"] == 108.9 and last["shares"] == 0
    assert last["equity"] == 1052.21 and env.ledger.cash == Decimal("1052.21")
    assert env.ledger.fees_paid == Decimal("18.89")
    assert sum(rewards) == pytest.approx(math.log(1052.21 / 1000))
    assert last["closed_trade_pnls"] == [52.21]
    assert all(row["policy_version"] == "frozen-policy-v7" for row in env.transition_records())
    assert env.replay_transitions()[0].action == 1


def test_next_open_gap_is_real_and_open_fill_does_not_see_future_volume():
    original = candles((100, 120, 122, 124))
    altered = (original[0], replace(original[1], high=500, low=1, close=300, volume=0), *original[2:])
    outputs = []
    for data in (original, altered):
        env = PaperTradingEnv(data, initial_cash=1000, target_exposures=(0, 1), risk_limits=RESEARCH_LIMITS, terminal_liquidation=False)
        before, _ = env.reset(seed=1)
        _, _, _, _, info = env.step(1)
        outputs.append((before, [fill["cost"] for fill in info["fills"]], info["cash"], info["shares"]))
        assert info["buy_price"] == 120 and info["shares"] == 8
    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1:] == outputs[1][1:]


def test_explicit_clock_cannot_read_next_bar_before_close_or_move_backwards():
    env = PaperTradingEnv(candles((100, 101, 102, 103)), risk_limits=RESEARCH_LIMITS, terminal_liquidation=False)
    env.reset()
    with pytest.raises(ValueError, match="reward availability"):
        env.step(0, as_of=candles()[1].timestamp)
    env.step(0, as_of=START + timedelta(hours=1))
    with pytest.raises(ValueError, match="move backwards"):
        env.step(0, as_of=START + timedelta(minutes=20))


@pytest.mark.parametrize("kind", ["kill", "stale", "size", "exposure", "hours"])
def test_risk_rejections_are_independent_of_the_policy_and_leave_cash_intact(kind):
    limits = RESEARCH_LIMITS
    start = START
    if kind == "size":
        limits = replace(limits, max_order_quantity=1)
    elif kind == "exposure":
        limits = replace(limits, max_symbol_weight="0.25")
    elif kind == "hours":
        limits = replace(limits, market_open=time(9, 15), market_close=time(15, 30))
        start = START.replace(hour=12)  # 17:30 exchange-local, market closed.
    env = PaperTradingEnv(candles(start=start), initial_cash=1000, risk_limits=limits, terminal_liquidation=False)
    env.reset()
    if kind == "kill":
        env.risk_supervisor.engage_kill_switch("operator stop")
    kwargs = {"as_of": start + timedelta(minutes=30)} if kind == "stale" else {}
    _, reward, _, _, info = env.step(4, **kwargs)
    assert info["rejections"]
    assert not info["fills"] and env.cash == 1000 and env.shares == 0
    assert reward == 0


def test_session_exit_is_precommitted_at_a_later_tradable_open_not_known_close():
    start = START.replace(hour=9, minute=45)  # 15:15, 15:20, 15:25, 15:30 local.
    env = PaperTradingEnv(candles((100, 100, 110, 999), start=start), initial_cash=1000)
    env.reset()
    env.step(4)
    _, _, _, _, info = env.step(4)
    assert info["session_liquidation"] and info["sell_price"] == 110
    assert info["shares"] == 0 and not info["liquidation_unfilled"]
    _, _, terminated, _, final = env.step(4)
    assert terminated and not final["fills"]  # Closed session cannot invent a fill.


def test_kill_switch_does_not_secretly_bypass_guards_for_terminal_exit():
    env = PaperTradingEnv(candles((100, 100, 100)), initial_cash=1000)
    env.reset()
    env.step(4)
    env.risk_supervisor.engage_kill_switch("retain positions, disable all orders")
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["terminal_liquidation"] and info["liquidation_unfilled"]
    assert env.shares == 10 and info["rejections"]


def test_random_training_dates_starts_cash_and_lengths_are_seeded_and_ephemeral():
    data = candles() + candles(start=START + timedelta(days=1))
    env = PaperTradingEnv(data, random_start=True, random_start_date=True, initial_cash_range=(500, 1500), episode_length=2)
    one, info_one = env.reset(seed=19)
    bounds = (env.start_index, env.end_index)
    two, info_two = env.reset(seed=19)
    assert np.array_equal(one, two) and info_one == info_two
    assert (env.start_index, env.end_index) == bounds and bounds[1] - bounds[0] == 2
    assert 500 <= env.cash <= 1500
    _, explicit = env.reset(options={"start_date": "2026-01-06", "initial_cash": 750, "random_start": False})
    assert explicit["timestamp"].date().isoformat() == "2026-01-06" and env.cash == 750


def test_short_training_window_truncates_and_expensive_share_is_valid_no_trade():
    env = PaperTradingEnv(candles(), initial_cash=10, episode_length=1)
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(4)
    # episode_length is a task-defined terminal horizon.  It is not a
    # continuation/time-limit truncation because the next reset gets a fresh
    # account; the DQN must mask this transition.
    assert terminated and not truncated and reward == 0
    assert info["episode_horizon_end"] and info["termination_reason"] == "episode_horizon"
    assert info["shares"] == 0 and info["cash"] == 10 and not info["fills"]
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(0)


def test_artificial_horizon_liquidates_open_position_and_closes_account():
    env = PaperTradingEnv(
        candles((100, 100, 110, 120)),
        initial_cash=1000,
        episode_length=2,
        target_exposures=(0, 1),
        risk_limits=RESEARCH_LIMITS,
        cost_model=CostModel(brokerage_rate="0.01", slippage_bps=100),
    )
    env.reset(seed=5)
    _, first_reward, first_terminated, first_truncated, first = env.step(1)
    assert not first_terminated and not first_truncated
    assert first["shares"] > 0 and first["cash"] < 1000
    _, final_reward, terminated, truncated, final = env.step(1)
    assert terminated and not truncated
    assert final["episode_horizon_end"] and final["termination_reason"] == "episode_horizon"
    assert final["terminal_liquidation"] and final["shares"] == 0
    assert final["cash"] == final["equity"]
    assert final["realized_fees"] > 0 and final["closed_trade_pnls"]
    assert float(env.ledger.realized_pnl) == pytest.approx(final["closed_trade_pnls"][0])
    assert env.ledger.snapshot().unrealized_pnl == 0
    assert first_reward + final_reward == pytest.approx(math.log(final["equity"] / 1000))


def test_artificial_horizon_is_masked_as_terminal_for_dqn_targets():
    torch = pytest.importorskip("torch")
    from trading_agent.learning import DQNConfig, DQNTrainer

    env = PaperTradingEnv(candles(), initial_cash=10, episode_length=1, risk_limits=RESEARCH_LIMITS)
    observation, _ = env.reset(seed=2)
    next_observation, reward, terminated, truncated, _ = env.step(0)
    trainer = DQNTrainer(DQNConfig(
        observation_dim=env.observation_dim,
        n_actions=env.n_actions,
        hidden_sizes=(),
        batch_size=1,
        learning_starts=1,
        replay_capacity=2,
        gamma=0.5,
    ))
    with torch.no_grad():
        trainer.online[0].bias.zero_()
        trainer.target[0].bias.fill_(2.0)
    targets = trainer._targets(
        torch.tensor([reward]),
        torch.tensor(np.asarray([next_observation])),
        torch.tensor([float(terminated or (truncated and not trainer.config.bootstrap_truncated))]),
    )
    assert terminated and not truncated and targets.item() == pytest.approx(reward)


def test_observation_schema_matches_evaluation_baselines_and_gym_checker():
    env = PaperTradingEnv(candles())
    assert env.feature_schema[3] == "log_return"
    check_env(env, skip_render_check=True)
    cash = evaluate_policy(env, CashBaseline())
    assert cash.metrics.net_return == 0 and cash.metrics.total_fees == 0
    results = evaluate_baselines(lambda: PaperTradingEnv(candles()))
    assert set(results) == {"cash", "rule", "random"}
    assert all(result.terminated for result in results.values())


def test_real_dqn_trainer_consumes_the_canonical_environment_without_float_execution():
    env = PaperTradingEnv(candles(), random_start=True, episode_length=2, initial_cash_range=(500, 1000))
    trainer = DQNTrainer(DQNConfig(
        observation_dim=env.observation_dim, n_actions=env.n_actions,
        hidden_sizes=(8,), batch_size=2, learning_starts=2,
        replay_capacity=16, seed=4,
    ), metadata={"feature_schema": env.feature_schema, "dataset_hash": env.dataset_hash})
    result = trainer.train(env, 6)
    assert result["steps"] == 6 and result["updates"] > 0
    assert len(trainer.replay) == 6
    assert env.ledger.cash >= 0 and env.shares >= 0


@pytest.mark.parametrize("action", [-1, 5, True, 1.2])
def test_invalid_action_is_atomic(action):
    env = PaperTradingEnv(candles())
    env.reset()
    before = env.broker.export_state()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert env.current_index == 0 and env.broker.export_state() == before
