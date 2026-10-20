from dataclasses import replace

import pytest

from trading_agent.evaluation import PerformanceMetrics
from trading_agent.registry import (
    CandidateRegistry,
    EvaluationSummary,
    EvaluationWindow,
    PromotionEvidence,
    PromotionGates,
)


def summary(policy_id, net_return=0.06):
    return EvaluationSummary(
        policy_id, "SYNTHETIC-data:features-v1:costs-v1:log-equity-v1:splits-v1",
        PerformanceMetrics(net_return, 0.05, sessions=30, trades=20, expected_shortfall=0.02),
        (EvaluationWindow(0, 100, net_return / 2, "up"), EvaluationWindow(100, 200, net_return / 2, "down")),
        (3, 7),
    )


def evidence(policy_id, *, incumbent=None, net_return=0.06, evaluation_id="trial-1"):
    receipts = {
        name: {
            "version": "promotion-evidence-v1",
            "name": name,
            "passed": True,
            "evaluation_id": evaluation_id,
        }
        for name in ("accounting", "data_integrity", "risk", "shadow", "recovery")
    }
    return PromotionEvidence(
        evaluation_id, summary(policy_id, net_return),
        {"cash": summary("cash", 0), "rule": summary("rule", 0.01), "random": summary("random", -0.02)},
        incumbent, accounting_ok=True, data_integrity_ok=True, risk_ok=True,
        shadow_passed=True, recovery_passed=True, receipts=receipts,
    )


def register(registry, directory, policy_id):
    path = directory / f"{policy_id}.pt"
    path.write_bytes(f"immutable-checkpoint:{policy_id}".encode())
    return registry.register(policy_id, path, metadata={"data_provenance": "SYNTHETIC-test"})


def test_promotion_rejection_persistence_and_rollback(tmp_path):
    database = tmp_path / "registry.sqlite"
    with CandidateRegistry(database) as registry:
        register(registry, tmp_path, "first")
        assert registry.active is None
        promoted = registry.consider("first", evidence("first"))
        assert promoted.accepted and promoted.reasons == ()
        register(registry, tmp_path, "worse")
        rejected = registry.consider("worse", evidence("worse", incumbent=summary("first"), net_return=0.005))
        assert not rejected.accepted
        assert "performance_deterioration:incumbent" in rejected.reasons
        assert "performance_deterioration:rule" in rejected.reasons
        assert registry.active_policy_id == "first"
    with CandidateRegistry(database) as registry:
        assert registry.active.policy_id == "first"
        register(registry, tmp_path, "better")
        assert registry.consider("better", evidence("better", incumbent=summary("first"), net_return=0.08)).accepted
        assert registry.active_policy_id == "better"
        rollback = registry.rollback(reason="shadow regression after promotion")
        assert rollback.previous_policy_id == "better" and rollback.active_policy_id == "first"
        assert registry.active_policy_id == "first"
        assert [row["kind"] for row in registry.history()] == ["registered", "promoted", "registered", "rejected", "registered", "promoted", "rollback"]
    with CandidateRegistry(database) as registry:
        assert registry.active_policy_id == "first"
        with pytest.raises(ValueError, match="previous"):
            registry.rollback(reason="cannot roll back to an unaccepted policy")


@pytest.mark.parametrize("flag,reason", [
    ("accounting_ok", "accounting_failed"), ("data_integrity_ok", "data_integrity_failed"),
    ("risk_ok", "risk_failed"), ("shadow_passed", "shadow_failed"), ("recovery_passed", "recovery_failed"),
])
def test_safety_evidence_fails_closed_and_logs_rejection(tmp_path, flag, reason):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "candidate")
        report = replace(evidence("candidate"), **{flag: False})
        decision = registry.consider("candidate", report)
        assert not decision.accepted and reason in decision.reasons
        assert registry.active_policy_id is None
        assert registry.history()[-1]["kind"] == "rejected"


def test_missing_versioned_receipts_fail_even_when_boolean_flags_are_true(tmp_path):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "candidate")
        decision = registry.consider("candidate", replace(evidence("candidate"), receipts={}))
        assert not decision.accepted
        assert "accounting_receipt_missing_or_invalid" in decision.reasons
        assert "shadow_receipt_missing_or_invalid" in decision.reasons


def test_disabling_shadow_and_recovery_gate_switches_cannot_bypass_hard_receipts(tmp_path):
    gates = replace(PromotionGates(), require_shadow=False, require_recovery=False)
    with CandidateRegistry(tmp_path / "registry.sqlite", gates=gates) as registry:
        register(registry, tmp_path, "candidate")
        decision = registry.consider(
            "candidate",
            replace(evidence("candidate"), shadow_passed=False, recovery_passed=False, receipts={}),
        )
        assert not decision.accepted
        assert "shadow_failed" in decision.reasons
        assert "recovery_failed" in decision.reasons


@pytest.mark.parametrize("metric,value,reason", [
    ("net_return", -0.01, "net_return_below_minimum"),
    ("max_drawdown", 0.21, "drawdown_limit"),
    ("expected_shortfall", 0.06, "tail_loss_limit"),
    ("expected_shortfall", None, "tail_loss_unavailable"),
    ("sessions", 1, "insufficient_sessions"),
    ("trades", 1, "insufficient_trades"),
    ("net_return", float("nan"), "invalid_candidate_metrics"),
    ("net_return", None, "invalid_candidate_metrics"),
])
def test_metric_and_sample_gates(tmp_path, metric, value, reason):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "candidate")
        report = evidence("candidate")
        candidate = replace(report.candidate, metrics=replace(report.candidate.metrics, **{metric: value}))
        decision = registry.consider("candidate", replace(report, candidate=candidate))
        assert not decision.accepted and reason in decision.reasons
        assert registry.active_policy_id is None
        assert registry.history()[-1]["payload"]["reasons"] == list(decision.reasons)


@pytest.mark.parametrize("change,reason", [
    ({"seeds": (3, 3)}, "insufficient_or_duplicate_seeds"),
    ({"windows": (EvaluationWindow(0, 100, 0.03, "up"),)}, "insufficient_windows"),
    ({"windows": (EvaluationWindow(0, 100, 0.03, "up"), EvaluationWindow(90, 200, 0.03, "down"))}, "invalid_or_overlapping_windows"),
    ({"windows": (EvaluationWindow(0, 100, 0.03, "up"), EvaluationWindow(100, 200, 0.03, "up"))}, "insufficient_regimes"),
    ({"windows": (EvaluationWindow(0, 100, 0.20, "up"), EvaluationWindow(100, 200, -0.01, "down"))}, "window_concentration"),
])
def test_chronological_multiseed_and_regime_gates(tmp_path, change, reason):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "candidate")
        report = evidence("candidate")
        decision = registry.consider("candidate", replace(report, candidate=replace(report.candidate, **change)))
        assert not decision.accepted and reason in decision.reasons


def test_baselines_must_be_present_and_comparable(tmp_path):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "candidate")
        report = evidence("candidate")
        baselines = dict(report.baselines)
        del baselines["cash"]
        baselines["rule"] = replace(baselines["rule"], comparison_key="different-costs")
        decision = registry.consider("candidate", replace(report, baselines=baselines))
        assert "missing_baseline:cash" in decision.reasons
        assert "comparison_mismatch:rule" in decision.reasons
        assert not decision.accepted


def test_decisions_are_idempotent_but_trials_and_candidates_immutable(tmp_path):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        first = register(registry, tmp_path, "candidate")
        assert registry.register("candidate", tmp_path / "candidate.pt", metadata=first.metadata) == first
        report = evidence("candidate")
        decision = registry.consider("candidate", report)
        assert registry.consider("candidate", report) == decision
        assert len(registry.history()) == 2
        with pytest.raises(ValueError, match="evaluation IDs"):
            registry.consider("candidate", replace(report, risk_ok=False))
        with pytest.raises(ValueError, match="immutable"):
            registry.register("candidate", tmp_path / "candidate.pt", metadata={"different": True})
        assert registry.active_policy_id == "candidate"


def test_predeclared_gates_cannot_change_on_reopen(tmp_path):
    path = tmp_path / "registry.sqlite"
    gates = replace(PromotionGates(), min_sessions=40)
    CandidateRegistry(path, gates=gates).close()
    with CandidateRegistry(path) as restored:
        assert restored.gates == gates
        with pytest.raises(AttributeError):
            restored.gates = PromotionGates()
    with pytest.raises(ValueError, match="fixed"):
        CandidateRegistry(path, gates=PromotionGates())


def test_metadata_is_normalized_without_breaking_registration_idempotency(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"checkpoint")
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        first = registry.register("candidate", path, metadata={"feature_schema": ("return", "cash")})
        again = registry.register("candidate", path, metadata={"feature_schema": ("return", "cash")})
        assert first == again == registry.get("candidate")
        assert len(registry.history()) == 1


def test_checkpoint_integrity_and_failed_rollback_leave_active_unchanged(tmp_path):
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        register(registry, tmp_path, "first")
        assert registry.consider("first", evidence("first")).accepted
        register(registry, tmp_path, "second")
        assert registry.consider("second", evidence("second", incumbent=summary("first"), net_return=0.08)).accepted
        (tmp_path / "first.pt").write_bytes(b"changed")
        count = len(registry.history())
        with pytest.raises(ValueError, match="changed"):
            registry.rollback(reason="test corrupted rollback target")
        assert registry.active_policy_id == "second" and len(registry.history()) == count
        register(registry, tmp_path, "third")
        (tmp_path / "third.pt").unlink()
        rejected = registry.consider("third", evidence("third", incumbent=summary("second", 0.08), net_return=0.10))
        assert not rejected.accepted and "checkpoint_missing_or_changed" in rejected.reasons
        assert registry.active_policy_id == "second"


def test_same_evidence_produces_identical_decisions_and_event_history(tmp_path):
    histories, decisions = [], []
    for name in ("run-a", "run-b"):
        directory = tmp_path / name
        directory.mkdir()
        with CandidateRegistry(directory / "registry.sqlite") as registry:
            register(registry, directory, "candidate")
            decisions.append(registry.consider("candidate", evidence("candidate")))
            histories.append(registry.history())
    assert decisions[0] == decisions[1]
    assert histories[0] == histories[1]
