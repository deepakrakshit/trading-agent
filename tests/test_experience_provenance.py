from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.experience import (
    ExperienceProvenanceError,
    build_manifest,
    history_digest,
    validate_manifest,
    validate_rows,
)


START = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
SCHEMA = [
    "open_to_close", "high_to_close", "low_to_close", "log_return",
    "relative_volume", "cash_fraction", "exposure", "drawdown", "episode_progress",
]
EXPECTED = {
    "compatibility_version": "paper-compat-v1",
    "config_version": "research-v1",
    "observation_dim": 9,
    "n_actions": 2,
    "feature_schema": SCHEMA,
    "feature_version": "paper-features-v1",
    "target_exposures": [0.0, 1.0],
    "reward_version": "log-post-cost-equity-v1",
    "execution_version": "causal-open-v2",
    "bar_interval_seconds": 300,
    "cost_model_version": "cost-v1",
    "risk_version": "risk-v1",
    "environment_config_hash": "environment-hash",
    "instrument": "ABC",
    "policy_id": "source-policy",
}


def manifest():
    return build_manifest(
        dataset_hash="full-dataset-hash",
        snapshot_manifest_sha256="source-manifest-hash",
        bar_interval_seconds=EXPECTED["bar_interval_seconds"],
        slice_hash="source-slice-hash",
        source_dataset="data/source.parquet",
        source_range=(10, 20),
        source_start_timestamp=START,
        source_end_timestamp=START + timedelta(minutes=45),
        instrument="ABC",
        environment_hash=EXPECTED["environment_config_hash"],
        account_config_hash="account-config-hash",
        experiment_config_hash="experiment-config-hash",
        config_version=EXPECTED["config_version"],
        feature_version=EXPECTED["feature_version"],
        feature_schema=SCHEMA,
        reward_version=EXPECTED["reward_version"],
        execution_version=EXPECTED["execution_version"],
        cost_model_version=EXPECTED["cost_model_version"],
        risk_version=EXPECTED["risk_version"],
        policy_version="source-policy",
        episode_id="episode-1",
    )


def rows():
    return [{
        "transition_id": "episode-1:10",
        "episode_id": "episode-1",
        "dataset_hash": "source-slice-hash",
        "source_dataset_hash": "full-dataset-hash",
        "environment_hash": "environment-hash",
        "instrument": "ABC",
        "feature_version": "paper-features-v1",
        "feature_schema": SCHEMA,
        "reward_version": "log-post-cost-equity-v1",
        "execution_version": "causal-open-v2",
        "cost_model_version": "cost-v1",
        "risk_version": "risk-v1",
        "policy_version": "source-policy",
        "config_hash": "account-config-hash",
        "index": 0,
        "next_index": 1,
        "timestamp": (START + timedelta(minutes=5)).isoformat(),
        "as_of": (START + timedelta(minutes=10)).isoformat(),
        "observation": [0.0] * 9,
        "action": 1,
        "reward": 0.01,
        "next_observation": [0.1] * 9,
        "terminated": False,
        "truncated": False,
    }]


def test_matching_manifest_and_rows_pass_with_later_dataset_hash_allowed():
    source = manifest()
    expected = dict(EXPECTED)
    expected["environment_config_hash"] = "environment-hash"
    checked = validate_manifest(
        source,
        expected_environment=expected,
        expected_symbol="ABC",
        expected_experiment_config_hash="experiment-config-hash",
        cutoff=START + timedelta(hours=1),
    )
    checked_rows = validate_rows(
        rows(), checked, cutoff=START + timedelta(hours=1), expected_environment=expected
    )
    assert checked_rows[0]["source_dataset_hash"] == "full-dataset-hash"
    assert history_digest("episode-1", checked_rows)


def test_incompatible_instrument_is_rejected():
    source = manifest()
    expected = dict(EXPECTED)
    source["instrument"] = "OTHER"
    with pytest.raises(ExperienceProvenanceError, match="instrument mismatch"):
        validate_manifest(
            source,
            expected_environment=expected,
            expected_symbol="ABC",
            expected_experiment_config_hash="experiment-config-hash",
            cutoff=START + timedelta(hours=1),
        )


@pytest.mark.parametrize(
    "field,value,expected_text",
    [("feature_schema", ["wrong"] * 9, "feature_schema"), ("execution_version", "other-execution", "execution_version")],
)
def test_incompatible_environment_semantics_are_rejected(field, value, expected_text):
    source = manifest()
    expected = dict(EXPECTED)
    expected[field] = value
    with pytest.raises(ExperienceProvenanceError, match=expected_text):
        validate_manifest(
            source,
            expected_environment=expected,
            expected_symbol="ABC",
            expected_experiment_config_hash="experiment-config-hash",
            cutoff=START + timedelta(hours=1),
        )


def test_dataset_identity_tampering_is_rejected_at_row_boundary():
    source = manifest()
    expected = dict(EXPECTED)
    source["dataset_hash"] = "tampered-full-dataset"
    with pytest.raises(ExperienceProvenanceError, match="source dataset hash"):
        validate_rows(rows(), source, cutoff=START + timedelta(hours=1), expected_environment=expected)


def test_row_level_corruption_and_late_source_are_rejected():
    source = manifest()
    expected = dict(EXPECTED)
    corrupted = rows()
    corrupted[0]["reward_version"] = "future-reward"
    with pytest.raises(ExperienceProvenanceError, match="reward_version"):
        validate_rows(corrupted, source, cutoff=START + timedelta(hours=1), expected_environment=expected)
    with pytest.raises(ExperienceProvenanceError, match="after training cutoff"):
        validate_manifest(
            source,
            expected_environment=expected,
            expected_symbol="ABC",
            expected_experiment_config_hash="experiment-config-hash",
            cutoff=START,
        )
