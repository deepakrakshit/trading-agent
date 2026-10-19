"""Hand-calculated paper environment and causal-boundary checks."""

import math

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

from trading_agent.env import (
    FillPoint,
    MarketBar,
    SimplePaperExecution,
    TradingEnv,
)


def bars():
    return [
        MarketBar(0, 100, 100, 100, 100, 10),
        MarketBar(1, 120, 132, 120, 132, 15),
        MarketBar(2, 110, 110, 100, 100, 10),
        MarketBar(3, 105, 106, 104, 105, 12),
    ]


def test_next_bar_fill_and_log_reward_reconcile():
    env = TradingEnv(bars(), initial_cash=1_000, target_exposures=(0, 1))
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["equity"] == 1_000
    _, reward, terminated, truncated, info = env.step(1)
    assert (terminated, truncated) == (False, False)
    assert info["buy_price"] == 120  # Never the decision bar's close of 100.
    assert info["shares"] == 8
    assert info["cash"] == 40
    assert info["equity"] == 1_096
    assert info["decision_index"] == 0
    assert reward == pytest.approx(math.log(1_096 / 1_000))
    _, reward2, _, _, info2 = env.step(0)
    assert info2["sell_price"] == 110
    assert info2["equity"] == 920
    assert reward + reward2 == pytest.approx(math.log(920 / 1_000))


def test_fees_and_slippage_once_and_no_overdraft():
    env = TradingEnv(
        bars(), initial_cash=1_000, target_exposures=(0, 1),
        execution=SimplePaperExecution(fee_rate=0.01, slippage_bps=100),
    )
    env.reset()
    _, reward, _, _, info = env.step(1)
    fill = 120 * 1.01
    fee = 8 * fill * 0.01
    expected_cash = 1_000 - 8 * fill - fee
    assert info["shares"] == 8
    assert info["cash"] == pytest.approx(expected_cash)
    assert info["realized_fees"] == pytest.approx(fee)
    assert info["equity"] == pytest.approx(expected_cash + 8 * 132)
    assert reward == pytest.approx(math.log(info["equity"] / 1_000))
    assert env.cash >= 0


def test_reset_clears_account_high_water_mark_and_done():
    env = TradingEnv(bars(), initial_cash=1_000, target_exposures=(0, 1))
    initial, _ = env.reset(seed=11)
    for action in (1, 1, 0):
        env.step(action)
    with pytest.raises(RuntimeError):
        env.step(0)
    restored, info = env.reset(seed=11)
    assert list(initial) == list(restored)
    assert (env.cash, env.shares, env.high_water_mark, env.drawdown) == (1_000, 0, 1_000, 0)
    assert info["index"] == 0


def test_future_changes_cannot_change_current_observation_or_prior_reward():
    recorded_prefixes = []

    def features(prefix, index):
        recorded_prefixes.append((len(prefix), index))
        assert len(prefix) == index + 1
        return (prefix[-1].close / 100,)

    changed = bars()
    changed[-1] = MarketBar(3, 10_000, 20_000, 9_000, 15_000, 100_000)
    original = TradingEnv(bars(), feature_provider=features)
    altered = TradingEnv(changed, feature_provider=features)
    assert list(original.reset()[0]) == list(altered.reset()[0])
    step_a, step_b = original.step(4), altered.step(4)
    assert list(step_a[0]) == list(step_b[0])
    assert step_a[1:] == step_b[1:]
    assert all(length == index + 1 for length, index in recorded_prefixes)


def test_execution_receives_only_next_open_event_and_terminal_flatten_is_next_open():
    events = []
    delegate = SimplePaperExecution(fee_rate=0.001)

    class Execution:
        def rebalance(self, **kwargs):
            events.append(kwargs["reference"])
            assert isinstance(kwargs["reference"], FillPoint)
            assert not hasattr(kwargs["reference"], "close")
            return delegate.rebalance(**kwargs)

    env = TradingEnv(bars(), execution=Execution(), terminal_liquidation=True)
    env.reset()
    env.step(4)
    env.step(4)
    _, _, terminated, truncated, info = env.step(4)
    assert terminated and not truncated
    assert env.shares == 0
    assert info["sell_price"] == 105
    assert info["terminal_liquidation"]
    assert events == [FillPoint(1, 120), FillPoint(2, 110), FillPoint(3, 105)]


@pytest.mark.parametrize("action", [-1, 5, 1.5, True])
def test_invalid_actions_fail_without_mutating_state(action):
    env = TradingEnv(bars())
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.current_index == 0 and env.cash == 10_000 and env.shares == 0


def test_unaffordable_share_is_valid_no_trade():
    env = TradingEnv(bars(), initial_cash=10)
    env.reset()
    _, reward, _, _, info = env.step(4)
    assert info["shares"] == 0 and info["cash"] == 10 and reward == 0


def test_rejects_incomplete_or_nonchronological_inputs():
    with pytest.raises(ValueError, match="completed"):
        MarketBar(1, 100, 101, 99, 100, completed=False)
    with pytest.raises(ValueError, match="chronological"):
        TradingEnv(list(reversed(bars())))


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(TradingEnv(bars()), skip_render_check=True)
