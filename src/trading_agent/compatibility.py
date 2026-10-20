"""Strict semantic compatibility checks for learned-policy checkpoints.

Tensor shapes are not enough to prove that a policy is safe to reload.  A
checkpoint can have the same input/output dimensions while the feature order,
target exposure mapping, reward, execution, or risk semantics have changed.
This module contains the one shared validation boundary used by training,
evaluation, paper replay, and promotion preparation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


CHECKPOINT_FORMAT = "trading-agent-dqn"
CHECKPOINT_VERSION = 2
COMPATIBILITY_VERSION = "paper-compat-v1"

REQUIRED_COMPATIBILITY_FIELDS = (
    "compatibility_version",
    "config_version",
    "observation_dim",
    "n_actions",
    "feature_schema",
    "feature_version",
    "target_exposures",
    "reward_version",
    "execution_version",
    "bar_interval_seconds",
    "cost_model_version",
    "risk_version",
    "environment_config_hash",
    "instrument",
    "policy_id",
)


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint is structurally or semantically incompatible."""


def _normalise_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CheckpointCompatibilityError(f"checkpoint compatibility field {field!r} must be a sequence")
    return list(value)


def _validate_shape_fields(compatibility: Mapping[str, Any], *, context: str) -> None:
    observation_dim = compatibility.get("observation_dim")
    n_actions = compatibility.get("n_actions")
    if (
        isinstance(observation_dim, bool)
        or not isinstance(observation_dim, int)
        or observation_dim <= 0
    ):
        raise CheckpointCompatibilityError(f"{context} has an invalid observation_dim")
    if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions <= 0:
        raise CheckpointCompatibilityError(f"{context} has an invalid n_actions")

    schema = _normalise_sequence(compatibility.get("feature_schema"), "feature_schema")
    if len(schema) != observation_dim or any(not isinstance(item, str) or not item for item in schema):
        raise CheckpointCompatibilityError(
            f"{context} feature_schema length/content does not match observation_dim"
        )
    exposures = _normalise_sequence(compatibility.get("target_exposures"), "target_exposures")
    if len(exposures) != n_actions:
        raise CheckpointCompatibilityError(
            f"{context} target_exposures length does not match n_actions"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in exposures
    ):
        raise CheckpointCompatibilityError(f"{context} target_exposures are invalid")
    if exposures != sorted(set(exposures)) or exposures[0] != 0.0:
        raise CheckpointCompatibilityError(
            f"{context} target_exposures must be sorted, unique, and include cash"
        )


def validate_checkpoint_compatibility(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str = "checkpoint",
) -> None:
    """Require exact equality for every production checkpoint contract field.

    ``expected`` is supplied by the actual target environment.  The policy ID
    is intentionally part of the contract: a checkpoint may be evaluated under
    a different environment policy *version* only when the caller explicitly
    gives that expected identity.  The environment hash itself excludes that
    policy label and represents the numerical/accounting semantics.
    """

    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError(f"{context} metadata is missing or not a mapping")
    actual = metadata.get("checkpoint_compatibility")
    if not isinstance(actual, Mapping):
        raise CheckpointCompatibilityError(
            f"{context} is missing required checkpoint_compatibility metadata; retrain or migrate it"
        )
    missing = [field for field in REQUIRED_COMPATIBILITY_FIELDS if field not in actual]
    if missing:
        raise CheckpointCompatibilityError(
            f"{context} is missing required compatibility fields: {', '.join(missing)}"
        )
    _validate_shape_fields(actual, context=context)
    _validate_shape_fields(expected, context="target environment")

    for field in REQUIRED_COMPATIBILITY_FIELDS:
        actual_value = actual.get(field)
        expected_value = expected.get(field)
        if field in ("feature_schema", "target_exposures"):
            actual_value = list(actual_value)
            expected_value = list(expected_value)
        if actual_value != expected_value:
            raise CheckpointCompatibilityError(
                f"{context} compatibility mismatch for {field!r}: "
                f"checkpoint={actual_value!r}, expected={expected_value!r}"
            )


def validate_checkpoint_identity(
    metadata: Mapping[str, Any],
    *,
    policy_id: str,
    context: str = "checkpoint",
) -> None:
    """Validate the policy identity independently of numerical semantics."""

    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("policy_id must be a non-empty string")
    compatibility = metadata.get("checkpoint_compatibility") if isinstance(metadata, Mapping) else None
    if not isinstance(compatibility, Mapping) or compatibility.get("policy_id") != policy_id:
        actual = compatibility.get("policy_id") if isinstance(compatibility, Mapping) else None
        raise CheckpointCompatibilityError(
            f"{context} policy identity mismatch: checkpoint={actual!r}, expected={policy_id!r}"
        )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "COMPATIBILITY_VERSION",
    "CheckpointCompatibilityError",
    "REQUIRED_COMPATIBILITY_FIELDS",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_identity",
]
