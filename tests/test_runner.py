"""Public replay, restart, kill, inspection, and honest candidate boundaries."""

from copy import deepcopy

import pytest

from trading_agent.config import load_config
from trading_agent.runner import inspect_account, run_paper, set_kill_switch
from trading_agent.workflow import generate_synthetic, train_experiment


@pytest.fixture
def experiment(tmp_path):
    data = tmp_path / "synthetic.parquet"
    generate_synthetic(data, sessions=4)
    config = deepcopy(load_config())
    config["training"].update(hidden_sizes=[8], batch_size=4, learning_starts=8, replay_capacity=64, steps=32, episode_length=20)
    output = tmp_path / "model"
    train_experiment(data, output, config)
    return output


def test_candidate_requires_shadow_and_loaded_model_runs_without_updating(experiment, tmp_path):
    db = tmp_path / "candidate.sqlite"
    with pytest.raises(ValueError, match="shadow"):
        run_paper(experiment, db, strategy="candidate")
    report = run_paper(experiment, db, strategy="candidate", shadow=True)
    assert report["mode"] == "SHADOW_PAPER_REPLAY"
    assert report["provenance"] == ["SYNTHETIC"]
    assert report["terminated"] and report["transitions"] == 74
    assert all(row["payload"]["policy_version"] == "model" for row in inspect_account(db, view="transitions"))


def test_persistent_split_replay_matches_uninterrupted_execution(experiment, tmp_path):
    full = run_paper(experiment, tmp_path / "full.sqlite", strategy="rule", shadow=True)
    partial_db = tmp_path / "partial.sqlite"
    partial = run_paper(experiment, partial_db, strategy="rule", shadow=True, max_steps=25)
    assert partial["transitions"] == 25 and not partial["terminated"]
    resumed = run_paper(None, partial_db, resume=True)
    assert resumed["account"] == full["account"]
    assert resumed["metrics"] == full["metrics"]
    assert resumed["attribution"] == full["attribution"]
    assert resumed["metrics"]["trades"] > 0 and float(resumed["account"]["fees_paid"]) > 0
    assert run_paper(None, partial_db, resume=True) == resumed
    assert len(inspect_account(partial_db, view="fills")) == resumed["metrics"]["trades"]
    summary = inspect_account(partial_db)
    assert summary["paper_only"] and summary["account"] == resumed["account"]


def test_kill_switch_is_atomic_durable_and_blocks_orders_after_restart(experiment, tmp_path):
    db = tmp_path / "kill.sqlite"
    run_paper(experiment, db, strategy="rule", shadow=True, max_steps=2)
    receipt = set_kill_switch(db)
    assert receipt["kill_switch"] and receipt["pending_orders"] == 0
    count_before = len(inspect_account(db, view="fills"))
    report = run_paper(None, db, resume=True, max_steps=5)
    assert report["kill_switch"] and report["rejections"] >= 5
    assert len(inspect_account(db, view="fills")) == count_before
    assert not set_kill_switch(db, enabled=False)["kill_switch"]


def test_no_approved_policy_means_cash_not_automatic_candidate_use(experiment, tmp_path):
    report = run_paper(experiment, tmp_path / "active.sqlite", registry_path=tmp_path / "missing.sqlite")
    assert report["policy_version"] == "cash-v1"
    assert report["account"]["equity"] == "10000.00"
    assert report["metrics"]["trades"] == 0
