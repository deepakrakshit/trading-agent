"""Reproducible local research orchestration; no network or real execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time as wall_time, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .config import make_env
from .data import CSVBarProvider, DataValidationError, MarketBar
from .data.calendar import KOLKATA, FIVE_MINUTES, SessionCalendar
from .data.persistence import load_snapshot, write_snapshot
from .evaluation import (
    CashBaseline, PerformanceMetrics, chronological_splits, compute_metrics,
    evaluate_baselines, evaluate_policy,
)
from .learning import DQNConfig, DQNTrainer, Transition
from .registry import (
    CandidateRegistry, EvaluationSummary, EvaluationWindow, PromotionEvidence, PromotionGates,
)
from .storage import SQLiteRepository


def jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not a report value: {type(value).__name__}")


def save_json(path: str | Path, payload: Any) -> None:
    """Atomic artifact update; never include provider objects/credentials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, default=jsonable, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, default=jsonable, sort_keys=True).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def code_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "source-archive"
    return result.stdout.strip() if result.returncode == 0 else "source-archive"


def versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "pandas", "pyarrow")}


def generate_synthetic(path: str | Path, *, sessions=24, seed=2026) -> dict:
    """Generate weekday artificial regimes; this is NOT Indian market history."""
    if not isinstance(sessions, int) or not 2 <= sessions <= 250:
        raise ValueError("synthetic sessions must be between 2 and 250")
    rng = np.random.default_rng(seed)
    calendar = SessionCalendar()
    day, price, bars = date(2026, 1, 5), 200.0, []
    for session in range(sessions):
        while calendar.session_bounds(day) is None:
            day += timedelta(days=1)
        opening, closing = calendar.session_bounds(day)
        regime = (session // 4) % 3
        drift = (0.00018, -0.00018, 0.0)[regime]
        stamp = opening
        while stamp < closing:
            open_price = price * math.exp(float(rng.normal(0, 0.0003)))
            close_price = open_price * math.exp(drift + float(rng.normal(0, 0.0018)))
            high = max(open_price, close_price) * (1 + float(rng.uniform(0.0001, 0.001)))
            low = min(open_price, close_price) * (1 - float(rng.uniform(0.0001, 0.001)))
            numbers = [round(value, 2) for value in (open_price, high, low, close_price)]
            bars.append(MarketBar("SYNTHETIC_EQ", stamp, *numbers, int(rng.integers(10000, 50000)), source="SYNTHETIC"))
            price, stamp = close_price, stamp + FIVE_MINUTES
        day += timedelta(days=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_snapshot(path, bars, source="LOCAL_SYNTHETIC_V1", provenance="SYNTHETIC", source_metadata={
        "dataset_version": f"synthetic-v1-seed-{seed}", "interval": "FIVE_MINUTE",
        "timestamp_convention": "BAR_START", "calendar_version": calendar.version,
    })
    return manifest.to_mapping()


def load_market(path: str | Path, config: Mapping[str, Any]) -> tuple[MarketBar, ...]:
    snapshot = load_snapshot(path)
    bars = snapshot.bars
    if len({bar.symbol for bar in bars}) != 1 or len(bars) < 3:
        raise DataValidationError("one instrument and at least three candles required")
    calendar = SessionCalendar(holidays=config["risk"]["closed_dates"])
    calendar.validate_intraday_bars(bars, bars[0].symbol, start=bars[0].timestamp, end=bars[-1].timestamp + FIVE_MINUTES, as_of=datetime.now(timezone.utc))
    for previous, current in zip(bars, bars[1:]):
        if abs(float(current.open / previous.close) - 1) > 0.20 or abs(float(current.close / current.open) - 1) > 0.20 or current.high / current.open > Decimal("1.20") or current.low / current.open < Decimal("0.80"):
            raise DataValidationError("20% price outlier/corporate-action discontinuity needs explicit review")
    return bars


def experiment_plan(bars, config: Mapping[str, Any]) -> dict:
    """Freeze all chronological bounds before seeing any evaluation score."""
    n = len(bars)
    settings = config["evaluation"]
    train_end = int(n * settings["train_fraction"])
    gap = settings["purge_bars"] + settings["embargo_bars"]
    validation_start = train_end + gap
    validation_end = validation_start + int(n * settings["validation_fraction"])
    holdout_start = n - int(n * settings["holdout_fraction"])
    if not 2 <= train_end < validation_start < validation_end < holdout_start - gap < n - 1:
        raise ValueError("dataset too small or split fractions/gaps are inconsistent")
    folds = chronological_splits(
        n, train_size=max(10, n // 3), validation_size=max(2, n // 12),
        test_size=max(2, n // 8), purge=settings["purge_bars"],
        embargo=settings["embargo_bars"], final_holdout_size=n - holdout_start,
    )
    return {
        "train": [0, train_end], "validation": [validation_start, validation_end],
        "test": [holdout_start, n], "folds": [asdict(fold) for fold in folds],
        "selection": "fixed seed 0; fixed config; no holdout-driven selection",
        "purge": settings["purge_bars"], "embargo": settings["embargo_bars"],
    }


def build_trainer(bars, config: Mapping[str, Any], *, seed: int, device: str, metadata=None):
    import torch
    # Tiny MLPs otherwise incur large OpenMP launch overhead on many-core CPUs.
    torch.set_num_threads(1)
    env = make_env(bars, config, training=True, policy_version=f"candidate-seed-{seed}")
    allowed = DQNConfig.__dataclass_fields__
    values = {key: value for key, value in config["training"].items() if key in allowed}
    trainer = DQNTrainer(DQNConfig(
        **values, observation_dim=env.observation_dim, n_actions=env.n_actions,
        seed=seed, device=device,
    ), metadata=metadata)
    return trainer, env


def train_experiment(data: str | Path, output: str | Path, config: Mapping[str, Any], *, steps=None, seed=0, device="cpu", resume=False, experience_db=None) -> dict:
    """Train a candidate only. The active registry/account is never modified."""
    output = Path(output)
    manifest_file = output / "experiment.json"
    bars = load_market(data, config)
    plan = experiment_plan(bars, config)
    plan["selection"] = f"fixed seed {seed}; fixed config; no holdout-driven selection"
    train_bars = bars[slice(*plan["train"])]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "candidate.pt"
    dataset_digest = digest([bar.to_mapping() for bar in bars])
    if manifest_file.exists():
        if not resume:
            raise FileExistsError("experiment exists; use --resume or a new --output")
        report = read_json(manifest_file)
        checkpoint = output / report["checkpoint"]
        if report["dataset_hash"] != dataset_digest or report["config"] != config or report["seed"] != seed:
            raise ValueError("resume dataset, configuration or seed mismatch")
        if report.get("checkpoint_sha256") != file_hash(checkpoint):
            raise ValueError("resume checkpoint hash mismatch")
        trainer = DQNTrainer.load_checkpoint(checkpoint, device=device)
        env = make_env(train_bars, config, training=True, policy_version=f"candidate-seed-{seed}")
        if (output / "evaluation.json").exists():
            raise ValueError("evaluated experiments are immutable; train a new candidate version")
    else:
        if resume:
            raise FileNotFoundError("no experiment to resume")
        metadata = {
            "dataset_hash": dataset_digest, "train_range": plan["train"],
            "feature_schema": list(make_env(train_bars, config).feature_schema),
            "feature_version": "paper-features-v1", "target_exposures": config["target_exposures"],
            "reward": "log(post_cost_equity_next/equity_current); no extra penalties",
            "config_hash": digest(config), "code_commit": code_revision(), "dependencies": versions(),
        }
        trainer, env = build_trainer(train_bars, config, seed=seed, device=device, metadata=metadata)
        report = {
            "id": output.name, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": os.path.relpath(Path(data).resolve(), output.resolve()),
            "dataset_hash": dataset_digest, "provenance": sorted({bar.source for bar in bars}),
            "plan": plan, "config": config, "seed": seed, "metadata": metadata,
            "checkpoint": "candidate.pt", "training_seconds": 0.0, "loss_history": [],
            "steps": 0, "updates": 0, "experience_transitions": 0,
        }
        if experience_db is not None:
            # Reject data after the training cutoff, never learn from this run's held-out demo.
            experience_spec = read_json(str(experience_db) + ".spec.json")
            if experience_spec["config"] != config or experience_spec["provenance"] != report["provenance"]:
                raise ValueError("paper experience provenance/action/risk/cost configuration mismatch")
            with SQLiteRepository(experience_db) as repo:
                rows = [json.loads(row["payload"]) for row in repo.transitions()]
            for row in rows:
                if datetime.fromisoformat(row["timestamp"]) > train_bars[-1].timestamp:
                    raise ValueError("paper experience overlaps validation/test/future training cutoff")
                if tuple(row["feature_schema"]) != env.feature_schema:
                    raise ValueError("paper experience feature schema mismatch")
                trainer.observe(Transition(**{key: row[key] for key in Transition.__dataclass_fields__}))
            report["experience_transitions"] = len(rows)
    budget = int(steps if steps is not None else config["training"]["steps"])
    if budget <= 0:
        raise ValueError("steps must be positive")
    if device.startswith("cuda"):
        trainer.torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    immutable_checkpoints = output / "checkpoints"
    immutable_checkpoints.mkdir(exist_ok=True)
    remaining = budget
    while remaining:
        chunk = min(remaining, config["training"]["checkpoint_every"])
        result = trainer.train(env, chunk)
        losses = result["losses"]
        report["loss_history"].append({"step": trainer.environment_steps, "mean_huber_loss": statistics.fmean(losses) if losses else None})
        report.update(steps=trainer.environment_steps, updates=trainer.gradient_steps, device=str(trainer.device))
        report["training_seconds"] += time.perf_counter() - start
        # Publish a new version before moving the manifest pointer. A crash can
        # leave an orphan new checkpoint, but never destroys the previous pair.
        checkpoint = immutable_checkpoints / f"step-{trainer.environment_steps:08d}.pt"
        if checkpoint.exists():
            raise FileExistsError("checkpoint version exists; inspect an interrupted run before retrying")
        trainer.save_checkpoint(checkpoint)
        report["checkpoint"] = os.path.relpath(checkpoint, output)
        report["checkpoint_sha256"] = file_hash(checkpoint)
        report["peak_vram_bytes"] = trainer.torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        save_json(manifest_file, report)
        remaining -= chunk
        start = time.perf_counter()
    # Reload as an independent integrity/inference check, not just a save claim.
    loaded = DQNTrainer.load_checkpoint(checkpoint, device="cpu")
    observation, _ = env.reset(seed=seed)
    report["reloaded_action"] = loaded.act(observation)
    # Convenience alias is never the resumable source of truth.
    trainer.save_checkpoint(output / "candidate.pt")
    save_json(manifest_file, report)
    return report


def _episode_report(episode) -> dict:
    return {
        "metrics": episode.metrics.to_dict(), "equity_curve": episode.equity_curve,
        "timestamps": episode.timestamps, "actions": episode.actions,
        "terminated": episode.terminated, "truncated": episode.truncated,
    }


def _aggregate_summary(policy_id: str, rows: list[dict], key: str, comparison_key: str, seeds) -> EvaluationSummary:
    """Aggregate disjoint window returns, never count seeds as extra sessions."""
    by_seed = {}
    windows = {}
    for row in rows:
        metric = row[key]["metrics"]
        by_seed.setdefault(row["seed"], []).append(metric)
        bounds = (row["split"]["test_start"], row["split"]["test_end"])
        windows.setdefault(bounds, []).append((metric["net_return"], row["regime"]))
    compound = [math.prod(1 + item["net_return"] for item in values) - 1 for values in by_seed.values()]
    flattened = [item for values in by_seed.values() for item in values]
    metrics = PerformanceMetrics(
        net_return=statistics.median(compound),
        max_drawdown=max(item["max_drawdown"] for item in flattened),
        sessions=min(sum(item["sessions"] for item in values) for values in by_seed.values()),
        trades=int(statistics.median(sum(item["trades"] for item in values) for values in by_seed.values())),
        expected_shortfall=max(item["expected_shortfall"] or 0 for item in flattened),
        total_fees=statistics.fmean(sum(item["total_fees"] for item in values) for values in by_seed.values()),
        turnover=statistics.fmean(sum(item["turnover"] for item in values) for values in by_seed.values()),
    )
    return EvaluationSummary(
        policy_id, comparison_key, metrics,
        tuple(EvaluationWindow(start, end, statistics.median(value[0] for value in values), values[0][1]) for (start, end), values in sorted(windows.items())),
        tuple(seeds),
    )


def evaluate_experiment(output: str | Path, *, registry_path="models/registry.sqlite", device="cpu") -> dict:
    """Fixed multi-seed walk-forward and once-only final holdout assessment."""
    output = Path(output)
    manifest = read_json(output / "experiment.json")
    config = manifest["config"]
    bars = load_market(output / manifest["dataset"], config)
    if manifest["dataset_hash"] != digest([bar.to_mapping() for bar in bars]):
        raise ValueError("evaluation dataset differs from training manifest")
    checkpoint = output / manifest["checkpoint"]
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint changed since training")
    destination = output / "evaluation.json"
    if destination.exists():
        return read_json(destination)  # Never tune/re-evaluate a consumed holdout.
    from .evaluation import WalkForwardSplit
    rows = []
    plan = manifest["plan"]
    seeds = config["training"]["seeds"]
    for fold_index, definition in enumerate(plan["folds"]):
        fold = WalkForwardSplit(**definition)
        training, validation, test = bars[fold.train], bars[fold.validation], bars[fold.test]
        for seed in seeds:
            model_dir = output / "walk_forward" / f"fold-{fold_index}-seed-{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / "candidate.pt"
            if model_file.exists():
                trainer = DQNTrainer.load_checkpoint(model_file, device=device)
                if trainer.metadata.get("fold") != definition or trainer.metadata.get("config_hash") != digest(config):
                    raise ValueError("walk-forward resume checkpoint mismatch")
                duration = read_json(model_dir / "training.json")["seconds"]
            else:
                trainer, env = build_trainer(training, config, seed=seed, device=device, metadata={"fold": definition, "config_hash": digest(config), "dataset_hash": manifest["dataset_hash"]})
                started = time.perf_counter()
                trainer.train(env, config["training"]["walk_forward_steps"])
                duration = time.perf_counter() - started
                trainer.save_checkpoint(model_file)
                save_json(model_dir / "training.json", {"seconds": duration, "steps": trainer.environment_steps, "checkpoint_sha256": file_hash(model_file)})
            factory = lambda block: make_env(block, config, policy_version=f"wf-{fold_index}-seed-{seed}")
            candidate = evaluate_policy(factory(test), trainer.act)
            validation_result = evaluate_policy(factory(validation), trainer.act)
            baselines = evaluate_baselines(lambda: factory(test), seed=seed)
            row = {
                "split": definition, "seed": seed, "training_seconds": duration,
                "regime": "rising-price-proxy" if test[-1].close >= test[0].close else "falling-price-proxy",
                "candidate": {"metrics": candidate.metrics.to_dict()},
                "validation": validation_result.metrics.to_dict(),
                **{name: {"metrics": result.metrics.to_dict()} for name, result in baselines.items()},
            }
            rows.append(row)
            save_json(output / "walk_forward_progress.json", rows)
    holdout = bars[slice(*plan["test"])]
    frozen = DQNTrainer.load_checkpoint(checkpoint, device=device)
    final_env = lambda: make_env(holdout, config, policy_version=manifest["id"])
    final = evaluate_policy(final_env(), frozen.act)
    validation = evaluate_policy(make_env(bars[slice(*plan["validation"])], config), frozen.act)
    baselines = evaluate_baselines(final_env, seed=manifest["seed"])
    comparison = digest({"data": manifest["dataset_hash"], "config": config, "folds": plan["folds"]})
    summary = _aggregate_summary(manifest["id"], rows, "candidate", comparison, seeds)
    baseline_summaries = {name: _aggregate_summary(name, rows, name, comparison, seeds) for name in baselines}
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with CandidateRegistry(registry_path, gates=PromotionGates(**config["promotion"])) as registry:
        registry.register(manifest["id"], checkpoint, metadata={"experiment_hash": digest(manifest), "provenance": manifest["provenance"]})
        evidence = PromotionEvidence(
            f"{manifest['id']}-walk-forward-v1", summary, baseline_summaries,
            # Offline mock/replay results are not an authorised live shadow trial.
            accounting_ok=True, data_integrity_ok=True, risk_ok=True,
            shadow_passed=False, recovery_passed=False,
        )
        decision = registry.consider(manifest["id"], evidence)
    report = {
        "experiment_id": manifest["id"], "provenance": manifest["provenance"],
        "dataset_hash": manifest["dataset_hash"], "evaluated_at": datetime.now(timezone.utc),
        "plan": plan, "walk_forward": rows, "walk_forward_summary": asdict(summary),
        "validation": _episode_report(validation), "test": _episode_report(final),
        "baselines": {name: _episode_report(result) for name, result in baselines.items()},
        "promotion": asdict(decision),
        "limitations": [
            "Synthetic data tests software, not a real market edge; no profit claim.",
            "No hyperparameter search or best-seed selection; final holdout assessed once.",
            "Walk-forward summary compounds fresh-account window returns per seed; drawdown is worst-window, not one continuous account.",
            "Few sessions; daily ratios/annualization are unstable and not forecasts.",
            "No verified live shadow evidence; promotion is deliberately fail-closed.",
        ],
    }
    save_json(destination, report)
    return read_json(destination)
