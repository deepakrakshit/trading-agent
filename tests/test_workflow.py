"""Credential-free vertical slice: real training, evaluation, reload and gates."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trading_agent.config import load_config
from trading_agent.learning import DQNTrainer
from trading_agent.workflow import (
    evaluate_experiment, experiment_plan, generate_synthetic, load_market,
    read_json, train_experiment,
)


def small_config():
    config = deepcopy(load_config())
    config["training"].update(
        hidden_sizes=[8], batch_size=4, learning_starts=8, replay_capacity=64,
        train_frequency=2, steps=40, walk_forward_steps=24, checkpoint_every=20,
        episode_length=20,
    )
    return config


def test_synthetic_snapshot_is_explicit_immutable_and_session_aligned(tmp_path):
    path = tmp_path / "candles.parquet"
    manifest = generate_synthetic(path, sessions=4)
    assert manifest["provenance"] == "SYNTHETIC" and manifest["row_count"] == 300
    bars = load_market(path, small_config())
    plan = experiment_plan(bars, small_config())
    assert plan["train"][1] < plan["validation"][0] < plan["test"][0]
    assert all(fold["test_end"] <= plan["test"][0] for fold in plan["folds"])
    with pytest.raises(FileExistsError):
        generate_synthetic(path, sessions=4)


def test_train_resume_evaluate_baselines_and_fail_closed_promotion(tmp_path):
    data = tmp_path / "candles.csv"
    generate_synthetic(data, sessions=4)
    output = tmp_path / "candidate-v1"
    config = small_config()
    report = train_experiment(data, output, config)
    assert report["steps"] == 40 and report["updates"] > 0
    assert report["reloaded_action"] in range(5)
    trainer = DQNTrainer.load_checkpoint(output / "candidate.pt")
    assert trainer.environment_steps == 40
    resumed = train_experiment(data, output, config, steps=20, resume=True)
    assert resumed["steps"] == 60
    with pytest.raises(FileExistsError):
        train_experiment(data, output, config)
    evaluation = evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")
    assert set(evaluation["baselines"]) == {"cash", "rule", "random"}
    assert evaluation["baselines"]["cash"]["metrics"]["net_return"] == 0
    assert not evaluation["promotion"]["accepted"]
    assert "shadow_failed" in evaluation["promotion"]["reasons"]
    assert evaluation["promotion"]["active_policy_id"] is None
    assert len(evaluation["walk_forward"]) >= 3
    assert evaluation["provenance"] == ["SYNTHETIC"]
    assert evaluation["session_count_audit"]["counts_match"]
    assert set(evaluation["session_count_audit"]["per_seed_compounded_net_returns"]) == {"0", "1", "2"}
    assert evaluation["session_count_audit"]["between_seed_stdev"] >= 0
    assert evaluation["walk_forward_summary"]["metrics"]["sessions"] == len(set(label for row in evaluation["walk_forward"] for label in row["session_ids"]))
    assert evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite") == evaluation
    with pytest.raises(ValueError, match="immutable"):
        train_experiment(data, output, config, resume=True)
    assert read_json(output / "experiment.json")["checkpoint_sha256"]


def test_walk_forward_checkpoint_receipts_are_verified_before_reuse(tmp_path):
    data = tmp_path / "candles.csv"
    generate_synthetic(data, sessions=4)
    output = tmp_path / "candidate-v1"
    config = small_config()
    train_experiment(data, output, config)
    evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")
    fold_dir = output / "walk_forward" / "fold-0-seed-0"
    checkpoint = fold_dir / "candidate.pt"
    receipt = fold_dir / "training.json"
    original_checkpoint = checkpoint.read_bytes()
    original_receipt = receipt.read_bytes()
    substitute = (output / "walk_forward" / "fold-0-seed-1" / "candidate.pt").read_bytes()
    evaluation = output / "evaluation.json"

    checkpoint.write_bytes(b"modified checkpoint")
    evaluation.unlink()
    with pytest.raises(ValueError, match="integrity mismatch"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")

    checkpoint.write_bytes(substitute)
    evaluation.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="integrity mismatch"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")

    checkpoint.write_bytes(original_checkpoint)
    receipt.unlink()
    evaluation.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="no training receipt"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")

    receipt.write_bytes(original_receipt)
    checkpoint.unlink()
    evaluation.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="receipt exists without checkpoint"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")

    checkpoint.write_bytes(original_checkpoint)
    evaluation.unlink(missing_ok=True)
    restored = evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")
    assert restored["promotion"]["accepted"] is False


def test_evaluation_rejects_replaced_snapshot_manifest(tmp_path):
    data = tmp_path / "candles.csv"
    generate_synthetic(data, sessions=4)
    output = tmp_path / "candidate-v1"
    config = small_config()
    train_experiment(data, output, config, steps=8)
    sidecar = Path(str(data) + ".manifest.json")
    payload = json.loads(sidecar.read_text())
    payload["source"] = "REPLACED_SOURCE"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")


def test_consumed_evaluation_receipt_is_hash_checked(tmp_path):
    data = tmp_path / "candles.csv"
    generate_synthetic(data, sessions=4)
    output = tmp_path / "candidate-v1"
    config = small_config()
    train_experiment(data, output, config, steps=8)
    evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")
    evaluation = output / "evaluation.json"
    evaluation.write_text(evaluation.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt hash"):
        evaluate_experiment(output, registry_path=tmp_path / "registry.sqlite")
