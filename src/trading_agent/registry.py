"""Deterministic, durable promotion/rejection/rollback for paper policies.

The registry accepts externally produced evaluation/shadow/recovery evidence;
it never trains a model, mutates risk limits, or contacts an execution service.
Gates are frozen into a SQLite registry before the first trial.  Every decision
and rollback is appended to an audit log in the same transaction as the active
policy pointer.  Rejected candidates cannot alter that pointer.

Checkpoints are immutable references plus SHA-256 hashes.  Keep each model in
its own versioned file.  Paths are stored relative to the registry directory,
so the artifacts and registry can move together.  Gates' sample thresholds are
engineering configuration, not claims of statistical/financial sufficiency.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .evaluation import PerformanceMetrics


def _json_safe(value: Any) -> Any:
    # Non-finite evidence must be rejected AND logged, never silently accepted
    # or emitted as non-standard JSON NaN/Infinity.
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_number": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PromotionGates:
    min_net_return: float = 0.0
    max_drawdown: float = 0.20
    max_expected_shortfall: float = 0.05
    min_sessions: int = 20
    min_trades: int = 10
    min_windows: int = 2
    min_seeds: int = 2
    min_regimes: int = 2
    min_positive_window_fraction: float = 0.5
    min_window_return: float = -0.10
    max_positive_return_concentration: float = 0.8
    max_baseline_deterioration: float = 0.0
    max_incumbent_deterioration: float = 0.0
    required_baselines: tuple[str, ...] = ("cash", "rule", "random")
    require_shadow: bool = True
    require_recovery: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_baselines", tuple(self.required_baselines))
        counts = (self.min_sessions, self.min_trades, self.min_windows, self.min_seeds, self.min_regimes)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("promotion sample requirements must be non-negative integers")
        values = (self.min_net_return, self.max_drawdown, self.max_expected_shortfall, self.min_positive_window_fraction, self.min_window_return, self.max_positive_return_concentration, self.max_baseline_deterioration, self.max_incumbent_deterioration)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("promotion thresholds must be finite")
        if any(not 0 <= value <= 1 for value in (self.max_drawdown, self.max_expected_shortfall, self.min_positive_window_fraction, self.max_positive_return_concentration)):
            raise ValueError("drawdown/tail/fraction thresholds must lie in [0, 1]")
        if self.max_baseline_deterioration < 0 or self.max_incumbent_deterioration < 0:
            raise ValueError("deterioration tolerances cannot be negative")
        if len(set(self.required_baselines)) != len(self.required_baselines) or any(not name for name in self.required_baselines):
            raise ValueError("required baseline names must be non-empty and unique")
        if not isinstance(self.require_shadow, bool) or not isinstance(self.require_recovery, bool):
            raise ValueError("shadow/recovery gate switches must be booleans")


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """Independent half-open dataset bounds with a declared market regime."""

    start: int
    end: int
    net_return: float
    regime: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    policy_id: str
    comparison_key: str
    metrics: PerformanceMetrics
    windows: tuple[EvaluationWindow, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "seeds", tuple(self.seeds))


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """Caller-owned immutable trial identity and independent safety receipts.

    ``comparison_key`` identifies dataset/window/feature/reward/cost/execution
    settings shared by candidate, incumbent, and baselines.  Window returns
    should be aggregated across the declared seeds, not the best seed.
    This boundary records supplied evidence; it does not manufacture shadow
    trading or recovery-test success.
    """

    evaluation_id: str
    candidate: EvaluationSummary
    baselines: Mapping[str, EvaluationSummary] = field(default_factory=dict)
    incumbent: EvaluationSummary | None = None
    accounting_ok: bool = False
    data_integrity_ok: bool = False
    risk_ok: bool = False
    shadow_passed: bool = False
    recovery_passed: bool = False


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    policy_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    policy_id: str
    evaluation_id: str
    accepted: bool
    previous_policy_id: str | None
    active_policy_id: str | None
    reasons: tuple[str, ...]
    event_sequence: int


@dataclass(frozen=True, slots=True)
class RollbackDecision:
    previous_policy_id: str
    active_policy_id: str
    reason: str
    event_sequence: int


def _valid_metrics(metrics: PerformanceMetrics) -> bool:
    data = asdict(metrics)
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)) for value in data.values()):
        return False
    if any(data[name] is None for name in ("net_return", "max_drawdown", "sessions", "trades", "turnover", "total_fees")):
        return False
    if not isinstance(metrics.sessions, int) or not isinstance(metrics.trades, int):
        return False
    if metrics.sessions < 0 or metrics.trades < 0 or metrics.net_return < -1 or not 0 <= metrics.max_drawdown <= 1:
        return False
    if metrics.expected_shortfall is not None and not 0 <= metrics.expected_shortfall <= 1:
        return False
    return metrics.total_fees >= 0 and metrics.turnover >= 0


def _window_bounds(summary: EvaluationSummary) -> tuple[tuple[int, int], ...]:
    return tuple((window.start, window.end) for window in summary.windows)


class CandidateRegistry:
    """SQLite-backed policy pointer and append-only deterministic decisions."""

    def __init__(self, path: str | Path, *, gates: PromotionGates | None = None) -> None:
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"registry directory does not exist: {self.path.parent}")
        self.root = self.path.parent.resolve()
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates (
                policy_id TEXT PRIMARY KEY, checkpoint_path TEXT NOT NULL,
                checkpoint_sha256 TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, policy_id TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                policy_id TEXT NOT NULL REFERENCES candidates(policy_id),
                evaluation_id TEXT NOT NULL, evidence_hash TEXT NOT NULL,
                decision TEXT NOT NULL, PRIMARY KEY (policy_id, evaluation_id)
            );
            CREATE TABLE IF NOT EXISTS active_stack (
                position INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL REFERENCES candidates(policy_id)
            );
        """)
        try:
            with self._write():
                stored = self._connection.execute("SELECT value FROM config WHERE key='gates'").fetchone()
                if stored is None:
                    self._gates = gates or PromotionGates()
                    self._connection.execute("INSERT INTO config VALUES ('gates', ?)", (_json(asdict(self.gates)),))
                    self._connection.execute("INSERT INTO config VALUES ('schema_version', '1')")
                else:
                    self._gates = PromotionGates(**json.loads(stored["value"]))
                    if gates is not None and gates != self.gates:
                        raise ValueError("registry promotion gates are fixed; use a versioned registry for new gates")
                    version = self._connection.execute("SELECT value FROM config WHERE key='schema_version'").fetchone()
                    if version is None or version["value"] != "1":
                        raise ValueError("unsupported registry schema version")
        except BaseException:
            self._connection.close()
            raise

    @property
    def gates(self) -> PromotionGates:
        """Predeclared thresholds; changing gates requires a new registry."""

        return self._gates

    @contextmanager
    def _write(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def _event(self, kind: str, policy_id: str, payload: Any) -> int:
        cursor = self._connection.execute("INSERT INTO events(kind, policy_id, payload) VALUES (?, ?, ?)", (kind, policy_id, _json(payload)))
        return int(cursor.lastrowid)

    def register(self, policy_id: str, checkpoint_path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> CandidateRecord:
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        clean_metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False, sort_keys=True))
        record = CandidateRecord(policy_id, os.path.relpath(checkpoint, self.root), _sha256(checkpoint), clean_metadata)
        payload = asdict(record)
        with self._write():
            row = self._connection.execute("SELECT * FROM candidates WHERE policy_id=?", (policy_id,)).fetchone()
            if row is not None:
                if self.get(policy_id) != record:
                    raise ValueError("candidate IDs are immutable; register a new policy version")
                return self.get(policy_id)
            self._connection.execute("INSERT INTO candidates VALUES (?, ?, ?, ?)", (policy_id, record.checkpoint_path, record.checkpoint_sha256, _json(record.metadata)))
            self._event("registered", policy_id, payload)
        return record

    def get(self, policy_id: str) -> CandidateRecord:
        row = self._connection.execute("SELECT * FROM candidates WHERE policy_id=?", (policy_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown candidate {policy_id!r}")
        return CandidateRecord(row["policy_id"], row["checkpoint_path"], row["checkpoint_sha256"], json.loads(row["metadata"]))

    def _checkpoint_matches(self, record: CandidateRecord) -> bool:
        path = self.root / record.checkpoint_path
        return path.is_file() and _sha256(path) == record.checkpoint_sha256

    @property
    def active_policy_id(self) -> str | None:
        row = self._connection.execute("SELECT policy_id FROM active_stack ORDER BY position DESC LIMIT 1").fetchone()
        return str(row["policy_id"]) if row is not None else None

    @property
    def active(self) -> CandidateRecord | None:
        policy_id = self.active_policy_id
        if policy_id is None:
            return None
        record = self.get(policy_id)
        if not self._checkpoint_matches(record):
            raise ValueError("active checkpoint is missing or changed")
        return record

    def _reasons(self, policy_id: str, evidence: PromotionEvidence, incumbent_id: str | None) -> tuple[str, ...]:
        gates = self.gates
        candidate = evidence.candidate
        metrics = candidate.metrics
        reasons = []
        if candidate.policy_id != policy_id:
            reasons.append("candidate_identity_mismatch")
        if not candidate.comparison_key:
            reasons.append("missing_comparison_key")
        if policy_id == incumbent_id:
            reasons.append("already_incumbent")
        if not self._checkpoint_matches(self.get(policy_id)):
            reasons.append("checkpoint_missing_or_changed")
        for name in ("accounting_ok", "data_integrity_ok", "risk_ok"):
            if getattr(evidence, name) is not True:
                reasons.append(name.removesuffix("_ok") + "_failed")
        if gates.require_shadow and evidence.shadow_passed is not True:
            reasons.append("shadow_failed")
        if gates.require_recovery and evidence.recovery_passed is not True:
            reasons.append("recovery_failed")
        valid = _valid_metrics(metrics)
        if not valid:
            reasons.append("invalid_candidate_metrics")
        else:
            if metrics.net_return < gates.min_net_return:
                reasons.append("net_return_below_minimum")
            if metrics.max_drawdown > gates.max_drawdown:
                reasons.append("drawdown_limit")
            if metrics.expected_shortfall is None:
                reasons.append("tail_loss_unavailable")
            elif metrics.expected_shortfall > gates.max_expected_shortfall:
                reasons.append("tail_loss_limit")
            if metrics.sessions < gates.min_sessions:
                reasons.append("insufficient_sessions")
            if metrics.trades < gates.min_trades:
                reasons.append("insufficient_trades")
        windows = candidate.windows
        if len(windows) < gates.min_windows:
            reasons.append("insufficient_windows")
        if len(set(candidate.seeds)) < gates.min_seeds or len(set(candidate.seeds)) != len(candidate.seeds):
            reasons.append("insufficient_or_duplicate_seeds")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in candidate.seeds):
            reasons.append("invalid_seed")
        valid_windows = all(isinstance(window.start, int) and isinstance(window.end, int) and 0 <= window.start < window.end and math.isfinite(window.net_return) for window in windows)
        if not valid_windows or any(first.end > second.start for first, second in zip(windows, windows[1:])):
            reasons.append("invalid_or_overlapping_windows")
        else:
            returns = [window.net_return for window in windows]
            if any(value < gates.min_window_return for value in returns):
                reasons.append("window_loss_limit")
            if returns and sum(value > 0 for value in returns) / len(returns) < gates.min_positive_window_fraction:
                reasons.append("insufficient_positive_windows")
            positive = [value for value in returns if value > 0]
            if positive and max(positive) / sum(positive) > gates.max_positive_return_concentration:
                reasons.append("window_concentration")
        if len({window.regime for window in windows if window.regime}) < gates.min_regimes:
            reasons.append("insufficient_regimes")

        def compare(reference: EvaluationSummary, label: str, tolerance: float) -> None:
            if reference.comparison_key != candidate.comparison_key or _window_bounds(reference) != _window_bounds(candidate) or set(reference.seeds) != set(candidate.seeds):
                reasons.append(f"comparison_mismatch:{label}")
            elif not _valid_metrics(reference.metrics):
                reasons.append(f"invalid_metrics:{label}")
            elif valid and metrics.net_return + tolerance < reference.metrics.net_return:
                reasons.append(f"performance_deterioration:{label}")

        for name in gates.required_baselines:
            baseline = evidence.baselines.get(name)
            if baseline is None:
                reasons.append(f"missing_baseline:{name}")
            elif baseline.policy_id != name:
                reasons.append(f"baseline_identity_mismatch:{name}")
            else:
                compare(baseline, name, gates.max_baseline_deterioration)
        if incumbent_id is not None:
            if evidence.incumbent is None or evidence.incumbent.policy_id != incumbent_id:
                reasons.append("incumbent_identity_mismatch")
            else:
                compare(evidence.incumbent, "incumbent", gates.max_incumbent_deterioration)
        elif evidence.incumbent is not None:
            reasons.append("unexpected_incumbent")
        return tuple(reasons)

    def consider(self, policy_id: str, evidence: PromotionEvidence) -> PromotionDecision:
        if not isinstance(evidence.evaluation_id, str) or not evidence.evaluation_id.strip():
            raise ValueError("evaluation_id must be non-empty and unique to a trial")
        payload = asdict(evidence)
        digest = hashlib.sha256(_json(payload).encode()).hexdigest()
        with self._write():
            self.get(policy_id)
            previous = self._connection.execute("SELECT * FROM decisions WHERE policy_id=? AND evaluation_id=?", (policy_id, evidence.evaluation_id)).fetchone()
            if previous is not None:
                if previous["evidence_hash"] != digest:
                    raise ValueError("evaluation IDs are immutable; changed evidence needs a new trial ID")
                data = json.loads(previous["decision"])
                data["reasons"] = tuple(data["reasons"])
                return PromotionDecision(**data)
            incumbent = self.active_policy_id
            reasons = self._reasons(policy_id, evidence, incumbent)
            accepted = not reasons
            if accepted:
                self._connection.execute("INSERT INTO active_stack(policy_id) VALUES (?)", (policy_id,))
            active_id = policy_id if accepted else incumbent
            sequence = self._event("promoted" if accepted else "rejected", policy_id, {"evaluation": payload, "gates": asdict(self.gates), "previous_policy_id": incumbent, "active_policy_id": active_id, "reasons": reasons})
            decision = PromotionDecision(policy_id, evidence.evaluation_id, accepted, incumbent, active_id, reasons, sequence)
            self._connection.execute("INSERT INTO decisions VALUES (?, ?, ?, ?)", (policy_id, evidence.evaluation_id, digest, _json(asdict(decision))))
            return decision

    promote = consider

    def rollback(self, *, reason: str) -> RollbackDecision:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rollback requires a non-empty reason")
        with self._write():
            rows = self._connection.execute("SELECT position, policy_id FROM active_stack ORDER BY position DESC LIMIT 2").fetchall()
            if len(rows) < 2:
                raise ValueError("no previous accepted policy is available for rollback")
            current, previous = rows
            target = self.get(previous["policy_id"])
            if not self._checkpoint_matches(target):
                raise ValueError("rollback checkpoint is missing or changed")
            self._connection.execute("DELETE FROM active_stack WHERE position=?", (current["position"],))
            sequence = self._event("rollback", target.policy_id, {"previous_policy_id": current["policy_id"], "active_policy_id": target.policy_id, "reason": reason})
            return RollbackDecision(current["policy_id"], target.policy_id, reason, sequence)

    def history(self) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        return tuple({"sequence": row["sequence"], "kind": row["kind"], "policy_id": row["policy_id"], "payload": json.loads(row["payload"])} for row in rows)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CandidateRegistry:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


ModelRegistry = CandidateRegistry


__all__ = [
    "CandidateRecord", "CandidateRegistry", "EvaluationSummary", "EvaluationWindow", "ModelRegistry",
    "PromotionDecision", "PromotionEvidence", "PromotionGates", "RollbackDecision",
]
