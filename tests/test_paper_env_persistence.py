"""Exact mid-run resume, atomic cursor/transition commits and request retries."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest

from trading_agent.data import MarketBar
from trading_agent.execution import PaperBroker
from trading_agent.paper_env import PaperTradingEnv
from trading_agent.portfolio import CostModel, PortfolioLedger
from trading_agent.risk import RiskLimits
from trading_agent.storage import PersistenceError, SQLiteRepository


START = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
BARS = tuple(
    MarketBar("ABC", START + timedelta(minutes=5 * index), price, price + 1, price - 1, price, 1000)
    for index, price in enumerate((100, 100, 102, 101, 103, 102, 104))
)
KWARGS = {
    "initial_cash": 1000, "policy_version": "incumbent-7", "config_version": "experiment-2",
    "cost_model": CostModel(brokerage_rate="0.001", slippage_bps=5),
    "risk_limits": RiskLimits(daily_loss_limit="0.5", max_drawdown="0.5"),
}


@pytest.fixture
def database():
    # Keep generated account files inside this isolated worktree.
    with TemporaryDirectory(dir=".") as folder:
        yield Path(folder) / "paper.sqlite3"


def assert_result_equal(first, second):
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_midrun_close_reopen_resumes_exactly_like_an_uninterrupted_episode(database):
    actions = (4, 4, 0, 2, 4, 4)
    reference = PaperTradingEnv(BARS, **KWARGS)
    reference.reset(seed=7)
    expected = [reference.step(action, transition_id=f"step-{index}") for index, action in enumerate(actions)]
    with SQLiteRepository(database) as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset(seed=7)
        for index in range(2):
            assert_result_equal(env.step(actions[index], transition_id=f"step-{index}"), expected[index])
        ledger_before = env.ledger.export_state()
        revision = env.broker.revision
    with SQLiteRepository(database) as repo:
        restarted = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        observation, info = restarted.reset(options={"resume": True})
        assert restarted.np_random_seed == 7
        assert np.array_equal(observation, expected[1][0]) and info == expected[1][-1]
        assert restarted.current_index == 2 and restarted.ledger.export_state() == ledger_before
        assert restarted.broker.revision == revision
        # This models a committed step whose caller never saw the response.
        assert_result_equal(restarted.step(actions[1], transition_id="step-1"), expected[1])
        assert restarted.current_index == 2 and restarted.broker.revision == revision
        for index in range(2, len(actions)):
            assert_result_equal(restarted.step(actions[index], transition_id=f"step-{index}"), expected[index])
        assert restarted.ledger.export_state() == reference.ledger.export_state()
        assert restarted.replay_transitions() == reference.replay_transitions()
        assert len(repo.transitions()) == len(actions)
        assert len(repo.events(event_type="TRANSITION")) == len(actions)
        ids = [record["transition_id"] for record in restarted.transition_records()]
        assert len(set(ids)) == len(actions)
        assert repo.load_state(PaperTradingEnv.CHECKPOINT_KEY)["broker_revision"] == restarted.broker.revision


@pytest.mark.parametrize("failure_point", ["transition", "event", "cursor"])
def test_step_transition_cursor_and_broker_rollback_as_one_commit(failure_point):
    with SQLiteRepository() as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset(seed=4)
        broker_before = env.broker.export_state()
        cursor_before = repo.load_state(PaperTradingEnv.CHECKPOINT_KEY)
        original_transition = repo.record_transition
        original_event = repo.append_event
        original_state = repo.save_state

        def record_transition(*args, **kwargs):
            if failure_point == "transition":
                raise RuntimeError("injected step failure")
            return original_transition(*args, **kwargs)

        def record_event(*args, **kwargs):
            if failure_point == "event" and len(args) > 1 and args[1] == "TRANSITION":
                raise RuntimeError("injected step failure")
            return original_event(*args, **kwargs)

        def save_state(key, payload):
            if failure_point == "cursor" and key == PaperTradingEnv.CHECKPOINT_KEY:
                raise RuntimeError("injected step failure")
            return original_state(key, payload)

        with (
            patch.object(repo, "record_transition", side_effect=record_transition),
            patch.object(repo, "append_event", side_effect=record_event),
            patch.object(repo, "save_state", side_effect=save_state),
        ):
            with pytest.raises(RuntimeError, match="injected step failure"):
                env.step(4, transition_id="atomic-step")
        assert env.current_index == 0 and env.broker.export_state() == broker_before
        assert repo.load_state(PaperBroker.STATE_KEY) == broker_before
        assert repo.load_state(PaperTradingEnv.CHECKPOINT_KEY) == cursor_before
        assert repo.get_order("atomic-step:order") is None
        assert repo.transitions() == () and not repo.events(event_type="FILL")
        result = env.step(4, transition_id="atomic-step")
        assert result[-1]["fills"] and len(repo.transitions()) == 1
        request = json.loads(repo.get_order("atomic-step:order")["payload"])
        assert request["submitted_at"] == BARS[0].timestamp.isoformat()
        assert request["sized_at"] == BARS[1].timestamp.isoformat()
        assert repo.events(event_type="ORDER")[0]["occurred_at"] == BARS[1].timestamp.isoformat()
        before_retry = env.broker.export_state()
        assert_result_equal(env.step(4, transition_id="atomic-step"), result)
        assert env.broker.export_state() == before_retry
        with pytest.raises(PersistenceError, match="conflicting duplicate"):
            env.step(0, transition_id="atomic-step")


@pytest.mark.parametrize("change", ["dataset", "policy", "configuration", "costs", "risk"])
def test_resume_validates_dataset_policy_and_entire_versioned_configuration(database, change):
    with SQLiteRepository(database) as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset()
        env.step(4)
        before = repo.load_state(PaperBroker.STATE_KEY)
    data, arguments = BARS, dict(KWARGS)
    if change == "dataset":
        data = (*BARS[:-1], replace(BARS[-1], volume=2000))
    elif change == "policy":
        arguments["policy_version"] = "other-policy"
    elif change == "configuration":
        arguments["config_version"] = "other-config"
    elif change == "costs":
        arguments["cost_model"] = replace(KWARGS["cost_model"], slippage_bps=10)
    elif change == "risk":
        arguments["risk_limits"] = replace(KWARGS["risk_limits"], max_symbol_weight="0.5")
    with SQLiteRepository(database) as repo:
        with pytest.raises(ValueError, match="mismatch"):
            PaperTradingEnv(data, repository=repo, **arguments)
        assert repo.load_state(PaperBroker.STATE_KEY) == before
        assert len(repo.transitions()) == 1


def test_random_reset_cannot_recapitalise_or_rewind_a_persistent_account():
    with SQLiteRepository() as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset()
        env.step(4)
        before = env.broker.export_state()
        for options in ({}, {"random_start": True}, {"initial_cash": 999999}, {"start_index": 3}):
            with pytest.raises(RuntimeError, match="use resume"):
                env.reset(seed=9, options=options)
            assert env.broker.export_state() == before
        with pytest.raises(ValueError, match="cannot reseed"):
            env.reset(seed=9, options={"resume": True})
        assert env.resume()[1]["index"] == 1


def test_environment_will_not_adopt_an_unmanaged_broker_or_mismatched_revision():
    with SQLiteRepository() as repo:
        PaperBroker(PortfolioLedger(1000), repository=repo)
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        with pytest.raises(ValueError, match="unmanaged broker"):
            env.reset()
    with SQLiteRepository() as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset()
        env.broker.checkpoint()  # An independent writer updated only the broker.
        with pytest.raises(PersistenceError, match="revisions"):
            PaperTradingEnv(BARS, repository=repo, **KWARGS).resume()


def test_manual_kill_switch_checkpoint_survives_restart_and_stale_runner_is_rejected():
    with SQLiteRepository() as repo:
        first = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        first.reset()
        stale = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        stale.resume()
        first.risk_supervisor.engage_kill_switch("operator stop")
        first.checkpoint()
        with pytest.raises(PersistenceError, match="another runner"):
            stale.step(4)
        resumed = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        resumed.resume()
        assert resumed.risk_supervisor.kill_switch_active
        _, reward, _, _, info = resumed.step(4)
        assert reward == 0 and not info["fills"]
        assert "kill switch" in info["rejections"][0]


def test_replaying_saved_actions_uses_the_same_canonical_costed_path():
    first = PaperTradingEnv(BARS, **KWARGS)
    first.reset(seed=7)
    for action in (4, 2, 0, 4, 0, 4):
        first.step(action)
    replay = PaperTradingEnv(BARS, **KWARGS)
    replay.reset(seed=7)
    for record in first.transition_records():
        result = replay.step(record["action"], transition_id=record["transition_id"])
        assert result[1] == record["reward"]
        assert result[-1]["fills"] == record["fills"]
    assert replay.ledger.export_state() == first.ledger.export_state()


@pytest.mark.parametrize("corruption", ["missing-prefix", "changed-reward"])
def test_resume_requires_the_entire_unchanged_transition_history(corruption):
    with SQLiteRepository() as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset()
        env.step(4, transition_id="first")
        env.step(0, transition_id="second")
        with repo.transaction():
            if corruption == "missing-prefix":
                repo.connection.execute("DELETE FROM transitions WHERE transition_id='first'")
            else:
                record = json.loads(repo.get_transition("first")["payload"])
                record["reward"] = 123
                repo.connection.execute(
                    "UPDATE transitions SET payload=? WHERE transition_id='first'",
                    (json.dumps(record, sort_keys=True),),
                )
        restarted = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        with pytest.raises(PersistenceError, match="transition history"):
            restarted.resume()
        with pytest.raises(RuntimeError, match="reset.*resume"):
            restarted.step(0)


@pytest.mark.parametrize("corruption", ["done", "last-info", "episode-options", "rng"])
def test_semantically_corrupt_environment_cursor_fails_before_resume(corruption):
    with SQLiteRepository() as repo:
        env = PaperTradingEnv(BARS, repository=repo, **KWARGS)
        env.reset(seed=12)
        state = deepcopy(repo.load_state(PaperTradingEnv.CHECKPOINT_KEY))
        if corruption == "done":
            state["done"] = True
        elif corruption == "last-info":
            state["last_info"]["index"] = 999
        elif corruption == "episode-options":
            state["episode_options"]["start_index"] = 999
        else:
            state["rng_state"] = {"invalid": True}
        repo.save_state(PaperTradingEnv.CHECKPOINT_KEY, state)
        with pytest.raises((PersistenceError, ValueError), match="checkpoint|cursor|RNG|rng|last_info|episode"):
            PaperTradingEnv(BARS, repository=repo, **KWARGS)
