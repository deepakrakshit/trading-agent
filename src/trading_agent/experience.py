"""Provenance and integrity checks for imported paper experience.

Paper transitions are useful training data only when their source identity and
numerical contract are explicit.  The manifest is intentionally separate from
the learner checkpoint: a later experiment may import an earlier dataset, but
it must be able to prove which earlier slice, instrument, policy, and execution
semantics produced the rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any


EXPERIENCE_MANIFEST_FORMAT = "paper-experience-v1"


class ExperienceProvenanceError(ValueError):
    """Raised when persisted experience cannot be trusted for import."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _required_text(manifest: Mapping[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ExperienceProvenanceError(f"experience manifest field {name!r} is missing or invalid")
    return value


def _parse_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ExperienceProvenanceError(f"experience {name} timestamp is missing or invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ExperienceProvenanceError(f"experience {name} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExperienceProvenanceError(f"experience {name} timestamp must be timezone-aware")
    return parsed


def build_manifest(
    *,
    dataset_hash: str,
    snapshot_manifest_sha256: str,
    bar_interval_seconds: int,
    slice_hash: str,
    source_dataset: str,
    source_range: Sequence[int],
    source_start_timestamp: datetime,
    source_end_timestamp: datetime,
    instrument: str,
    environment_hash: str,
    account_config_hash: str,
    experiment_config_hash: str,
    config_version: str,
    feature_version: str,
    feature_schema: Sequence[str],
    reward_version: str,
    execution_version: str,
    cost_model_version: str,
    risk_version: str,
    policy_version: str,
    episode_id: str,
) -> dict[str, Any]:
    """Create the explicit manifest stored beside a paper account."""

    if len(source_range) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in source_range):
        raise ValueError("source_range must contain two integer bounds")
    if source_range[0] < 0 or source_range[0] >= source_range[1]:
        raise ValueError("source_range must be ordered and non-empty")
    if source_start_timestamp.tzinfo is None or source_end_timestamp.tzinfo is None:
        raise ValueError("source timestamps must be timezone-aware")
    if source_start_timestamp > source_end_timestamp:
        raise ValueError("source timestamps must be chronological")
    values = {
        "format": EXPERIENCE_MANIFEST_FORMAT,
        "dataset_hash": dataset_hash,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "bar_interval_seconds": bar_interval_seconds,
        "slice_hash": slice_hash,
        "source_dataset": source_dataset,
        "source_range": list(source_range),
        "source_start_timestamp": source_start_timestamp.isoformat(),
        "source_end_timestamp": source_end_timestamp.isoformat(),
        "instrument": instrument,
        "environment_hash": environment_hash,
        "account_config_hash": account_config_hash,
        "experiment_config_hash": experiment_config_hash,
        "config_version": config_version,
        "feature_version": feature_version,
        "feature_schema": list(feature_schema),
        "reward_version": reward_version,
        "execution_version": execution_version,
        "cost_model_version": cost_model_version,
        "risk_version": risk_version,
        "policy_version": policy_version,
        "episode_id": episode_id,
    }
    for name, value in values.items():
        if name not in {"source_range", "feature_schema", "bar_interval_seconds"} and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"experience manifest field {name!r} must be non-empty")
    if isinstance(bar_interval_seconds, bool) or not isinstance(bar_interval_seconds, int) or bar_interval_seconds <= 0:
        raise ValueError("bar_interval_seconds must be a positive integer")
    if not values["feature_schema"] or any(not isinstance(item, str) or not item for item in values["feature_schema"]):
        raise ValueError("feature_schema must contain non-empty names")
    return values


def _compare(expected: Mapping[str, Any], actual: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ExperienceProvenanceError(
                f"{context} mismatch for {field!r}: source={actual.get(field)!r}, expected={expected.get(field)!r}"
            )


def validate_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    expected_environment: Mapping[str, Any],
    expected_symbol: str,
    expected_experiment_config_hash: str,
    cutoff: datetime,
    expected_snapshot_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate manifest identity and compatibility before reading transitions."""

    if not isinstance(manifest, Mapping):
        raise ExperienceProvenanceError(
            "paper experience has no explicit manifest; legacy accounts must be migrated, not imported"
        )
    if manifest.get("format") != EXPERIENCE_MANIFEST_FORMAT:
        raise ExperienceProvenanceError("unsupported or missing paper experience manifest format")
    required = {
        "format", "dataset_hash", "snapshot_manifest_sha256", "bar_interval_seconds", "slice_hash", "source_dataset", "source_range",
        "source_start_timestamp", "source_end_timestamp", "instrument", "environment_hash",
        "account_config_hash", "experiment_config_hash", "config_version", "feature_version",
        "feature_schema", "reward_version", "execution_version", "cost_model_version",
        "risk_version", "policy_version", "episode_id",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ExperienceProvenanceError(
            f"paper experience manifest is incomplete: missing {', '.join(missing)}"
        )
    for field in required - {"source_range", "feature_schema", "bar_interval_seconds"}:
        _required_text(manifest, field)
    if not isinstance(manifest["source_range"], list) or len(manifest["source_range"]) != 2:
        raise ExperienceProvenanceError("experience source_range is invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in manifest["source_range"]):
        raise ExperienceProvenanceError("experience source_range must contain integer bounds")
    if (
        isinstance(manifest["bar_interval_seconds"], bool)
        or not isinstance(manifest["bar_interval_seconds"], int)
        or manifest["bar_interval_seconds"] <= 0
    ):
        raise ExperienceProvenanceError("experience bar_interval_seconds is invalid")
    if manifest["source_range"][0] < 0 or manifest["source_range"][0] >= manifest["source_range"][1]:
        raise ExperienceProvenanceError("experience source_range is not ordered")
    if not isinstance(manifest["feature_schema"], list) or not manifest["feature_schema"]:
        raise ExperienceProvenanceError("experience feature_schema is invalid")
    if any(not isinstance(item, str) or not item for item in manifest["feature_schema"]):
        raise ExperienceProvenanceError("experience feature_schema contains invalid names")
    source_start = _parse_timestamp(manifest["source_start_timestamp"], "source_start")
    source_end = _parse_timestamp(manifest["source_end_timestamp"], "source_end")
    if source_start > source_end:
        raise ExperienceProvenanceError("experience source timestamps are not chronological")
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("experience import cutoff must be timezone-aware")
    if source_end > cutoff:
        raise ExperienceProvenanceError(
            f"paper experience source ends at {source_end.isoformat()}, after training cutoff {cutoff.isoformat()}"
        )
    if manifest["instrument"] != expected_symbol:
        raise ExperienceProvenanceError(
            f"paper experience instrument mismatch: source={manifest['instrument']!r}, expected={expected_symbol!r}"
        )
    if manifest["experiment_config_hash"] != expected_experiment_config_hash:
        raise ExperienceProvenanceError(
            "paper experience experiment configuration hash does not match the new training configuration"
        )
    if expected_snapshot_manifest_sha256 is not None and manifest["snapshot_manifest_sha256"] != expected_snapshot_manifest_sha256:
        raise ExperienceProvenanceError("paper experience snapshot manifest hash does not match its source dataset")

    expected_fields = (
        "environment_hash", "config_version", "feature_version", "feature_schema",
        "reward_version", "execution_version", "bar_interval_seconds",
        "cost_model_version", "risk_version", "instrument",
    )
    expected_mapping = dict(expected_environment)
    expected_mapping["instrument"] = expected_symbol
    expected_mapping["environment_hash"] = expected_environment.get("environment_config_hash")
    _compare(expected_mapping, manifest, expected_fields, context="paper experience environment")
    return dict(manifest)


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    cutoff: datetime,
    expected_environment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate every row's provenance, causal order, and numerical shape."""

    if not rows:
        raise ExperienceProvenanceError("paper experience database contains no transitions")
    result: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
    for offset, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ExperienceProvenanceError(f"paper experience row {offset} is not a mapping")
        row = dict(raw)
        for field in (
            "transition_id", "episode_id", "dataset_hash", "source_dataset_hash", "environment_hash",
            "instrument", "feature_version", "reward_version", "execution_version",
            "cost_model_version", "risk_version", "policy_version", "config_hash", "timestamp", "as_of",
            "index", "next_index", "terminated", "truncated",
        ):
            if field not in row:
                raise ExperienceProvenanceError(f"paper experience row {offset} is missing {field!r}")
        if row["episode_id"] != manifest["episode_id"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected episode_id")
        if row["dataset_hash"] != manifest["slice_hash"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected source slice hash")
        if row["source_dataset_hash"] != manifest["dataset_hash"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected source dataset hash")
        if row["environment_hash"] != manifest["environment_hash"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected environment hash")
        if row["instrument"] != manifest["instrument"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected instrument")
        if row["feature_version"] != manifest["feature_version"] or list(row.get("feature_schema", ())) != list(manifest["feature_schema"]):
            raise ExperienceProvenanceError(f"paper experience row {offset} has incompatible feature semantics")
        for field in ("reward_version", "execution_version", "cost_model_version", "risk_version", "policy_version"):
            if row[field] != manifest[field]:
                raise ExperienceProvenanceError(f"paper experience row {offset} has incompatible {field}")
        if row["config_hash"] != manifest["account_config_hash"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} has an unexpected account config hash")
        if not isinstance(row["transition_id"], str) or not row["transition_id"].strip():
            raise ExperienceProvenanceError(f"paper experience row {offset} has an invalid transition ID")
        if (
            isinstance(row["index"], bool)
            or not isinstance(row["index"], int)
            or isinstance(row["next_index"], bool)
            or not isinstance(row["next_index"], int)
            or row["index"] != offset
            or row["next_index"] != offset + 1
        ):
            raise ExperienceProvenanceError(f"paper experience row {offset} is not a contiguous episode transition")
        try:
            timestamp = _parse_timestamp(row["timestamp"], f"row {offset}")
            as_of = _parse_timestamp(row["as_of"], f"row {offset} as_of")
        except ExperienceProvenanceError:
            raise
        if timestamp > cutoff or as_of < timestamp + timedelta(seconds=manifest["bar_interval_seconds"]):
            raise ExperienceProvenanceError(f"paper experience row {offset} crosses the causal training cutoff")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ExperienceProvenanceError("paper experience transition timestamps are not strictly chronological")
        previous_timestamp = timestamp
        if not isinstance(row["terminated"], bool) or not isinstance(row["truncated"], bool):
            raise ExperienceProvenanceError(f"paper experience row {offset} termination flags are invalid")
        if row["terminated"] and row["truncated"]:
            raise ExperienceProvenanceError(f"paper experience row {offset} is both terminated and truncated")
        if not isinstance(row.get("observation"), list) or not isinstance(row.get("next_observation"), list):
            raise ExperienceProvenanceError(f"paper experience row {offset} has invalid observations")
        if len(row["observation"]) != len(expected_environment["feature_schema"]) or len(row["next_observation"]) != len(row["observation"]):
            raise ExperienceProvenanceError(f"paper experience row {offset} observation dimension is incompatible")
        for vector_name in ("observation", "next_observation"):
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in row[vector_name]):
                raise ExperienceProvenanceError(f"paper experience row {offset} contains a non-finite observation")
        if isinstance(row.get("action"), bool) or not isinstance(row.get("action"), int):
            raise ExperienceProvenanceError(f"paper experience row {offset} has an invalid action")
        if isinstance(row.get("reward"), bool) or not isinstance(row.get("reward"), (int, float)) or not math.isfinite(float(row["reward"])):
            raise ExperienceProvenanceError(f"paper experience row {offset} has an invalid reward")
        result.append(row)
    return tuple(result)


def history_digest(episode_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Recompute the environment's tamper-evident transition-chain digest."""

    digest = _hash({"episode": episode_id})
    for row in rows:
        digest = _hash({"previous": digest, "transition": row})
    return digest


__all__ = [
    "EXPERIENCE_MANIFEST_FORMAT",
    "ExperienceProvenanceError",
    "build_manifest",
    "history_digest",
    "validate_manifest",
    "validate_rows",
]
