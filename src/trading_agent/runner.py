"""Restartable paper replay runner. No exchange execution is implemented."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from .attribution import attribute_round_trips
from .config import make_env
from .evaluation import CashBaseline, RuleBaseline, compute_metrics
from .execution import PaperBroker
from .learning import DQNTrainer
from .portfolio import PortfolioLedger
from .registry import CandidateRegistry
from .storage import SQLiteRepository
from .workflow import digest, file_hash, load_market, read_json, save_json


def spec_path(db: str | Path) -> Path:
    return Path(str(db) + ".spec.json")


def open_paper(db: str | Path):
    """Reconstruct immutable market/config/policy inputs, then explicit resume."""
    db = Path(db)
    if not db.is_file():
        raise FileNotFoundError("paper account does not exist")
    spec = read_json(spec_path(db))
    data = db.parent / spec["dataset"]
    bars = load_market(data, spec["config"])[slice(*spec["range"])]
    if digest([bar.to_mapping() for bar in bars]) != spec["slice_hash"]:
        raise ValueError("paper replay data changed")
    if spec.get("checkpoint") and file_hash(db.parent / spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("paper policy artifact changed")
    repo = SQLiteRepository(db)
    try:
        env = make_env(bars, spec["config"], repository=repo, policy_version=spec["policy_version"])
        observation, info = env.resume()
    except BaseException:
        repo.close()
        raise
    return spec, bars, repo, env, observation, info


def run_paper(experiment: str | Path | None, db: str | Path, *, strategy="active", shadow=False, resume=False, max_steps=None, registry_path="models/registry.sqlite") -> dict:
    """Replay a frozen policy without any training or automatic promotion.

    An unapproved candidate or rule baseline is allowed only in a separately
    labelled shadow account. The uninitialized active-policy fallback is cash.
    """
    db = Path(db)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if resume:
        spec, bars, repo, env, observation, info = open_paper(db)
    else:
        if db.exists() or spec_path(db).exists():
            raise FileExistsError("paper account exists; use --resume or a new --db")
        if experiment is None:
            raise ValueError("a new account requires --experiment")
        experiment = Path(experiment)
        manifest = read_json(experiment / "experiment.json")
        config = manifest["config"]
        data = (experiment / manifest["dataset"]).resolve()
        all_bars = load_market(data, config)
        bounds = manifest["plan"]["test"]
        bars = all_bars[slice(*bounds)]
        checkpoint = None
        policy_version = "cash-v1"
        if strategy == "candidate":
            if not shadow:
                raise ValueError("an unapproved candidate requires --shadow")
            checkpoint = (experiment / manifest["checkpoint"]).resolve()
            if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
                raise ValueError("candidate checkpoint hash mismatch")
            policy_version = manifest["id"]
        elif strategy == "rule":
            if not shadow:
                raise ValueError("the diagnostic rule baseline requires --shadow")
            policy_version = "rule-baseline-v1"
        elif strategy == "active":
            path = Path(registry_path)
            if path.exists():
                with CandidateRegistry(path) as registry:
                    active = registry.active
                    if active is not None:
                        checkpoint = (path.parent / active.checkpoint_path).resolve()
                        policy_version = active.policy_id
        elif strategy != "cash":
            raise ValueError("unknown paper strategy")
        db.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "format": "paper-replay-v1", "config": config,
            "dataset": os.path.relpath(data, db.parent.resolve()), "range": bounds,
            "slice_hash": digest([bar.to_mapping() for bar in bars]),
            "provenance": manifest["provenance"], "strategy": strategy,
            "shadow": bool(shadow), "policy_version": policy_version,
            "checkpoint": os.path.relpath(checkpoint, db.parent.resolve()) if checkpoint else None,
            "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        }
        save_json(spec_path(db), spec)
        repo = SQLiteRepository(db)
        env = make_env(bars, config, repository=repo, policy_version=policy_version)
        observation, info = env.reset(seed=manifest["seed"])
    try:
        if spec["checkpoint"]:
            policy = DQNTrainer.load_checkpoint(db.parent / spec["checkpoint"], device="cpu")
            if policy.metadata.get("config_hash") != digest(spec["config"]):
                raise ValueError("policy feature/action/cost/risk configuration mismatch")
            action_policy = policy.act
        elif spec["strategy"] == "rule":
            action_policy = RuleBaseline(long_action=len(env.target_exposures) - 1)
        else:
            action_policy = CashBaseline()
        count = 0
        while not info.get("terminated", False) and not info.get("truncated", False):
            if max_steps is not None and count >= max_steps:
                break
            action = action_policy(observation)
            observation, _, _, _, info = env.step(action, transition_id=f"paper:{env.current_index}")
            count += 1
        report = paper_report(spec, bars, env)
        save_json(db.with_suffix(".report.json"), report)
        return report
    finally:
        repo.close()


def paper_report(spec: dict, bars, env) -> dict:
    records = env.transition_records()
    curve = [spec["config"]["initial_cash"], *(row["info"]["equity"] for row in records)]
    stamps = [bars[0].timestamp, *(datetime.fromisoformat(row["timestamp"]) for row in records)]
    metrics = None
    if records:
        metrics = compute_metrics(
            curve, timestamps=stamps,
            total_fees=sum(row["info"]["realized_fees"] for row in records),
            turnover_notional=sum(row["info"]["turnover"] for row in records),
            exposures=[row["info"]["exposure"] for row in records],
            trade_count=sum(len(row["fills"]) for row in records),
            trade_pnls=[pnl for row in records for pnl in row["info"]["closed_trade_pnls"]],
        ).to_dict()
    account = env.ledger.snapshot().to_mapping()
    attribution = attribute_round_trips(records, bars)
    if not env.shares:
        realized = sum((Decimal(trade["net_pnl"]) for trade in attribution["closed_round_trips"]), Decimal("0"))
        if realized != env.ledger.realized_pnl:
            raise AssertionError("closed paper trade attribution disagrees with the ledger")
    return {
        "mode": "SHADOW_PAPER_REPLAY" if spec["shadow"] else "PAPER_REPLAY",
        "provenance": spec["provenance"], "policy_version": spec["policy_version"],
        "account": account, "metrics": metrics, "transitions": len(records),
        "cursor": env.current_index, "terminated": env.render().get("terminated", False),
        "kill_switch": env.risk_supervisor.kill_switch_active,
        "rejections": sum(len(row["info"]["rejections"]) for row in records),
        "liquidation_unfilled": env.render().get("liquidation_unfilled", False),
        "closed_round_trips": len(attribution["closed_round_trips"]),
        "attribution": attribution,
        "note": "Simulated fills only. Replay is not a genuinely streaming live trial.",
    }


def inspect_account(db: str | Path, *, view="summary") -> dict[str, Any] | list:
    db = Path(db)
    if not db.is_file():
        raise FileNotFoundError("paper account does not exist")
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if view == "summary":
            row = connection.execute("SELECT payload FROM states WHERE state_key=?", (PaperBroker.STATE_KEY,)).fetchone()
            if row is None:
                raise ValueError("paper broker checkpoint missing")
            state = json.loads(row["payload"])
            snapshot = PortfolioLedger.from_state(state["ledger"]).snapshot().to_mapping()
            return {"account": snapshot, "risk": state["risk"], "paper_only": True}
        table = {"orders": "orders", "fills": "fills", "equity": "account_snapshots", "transitions": "transitions", "events": "events"}.get(view)
        if table is None:
            raise ValueError("unknown account view")
        return [dict(row) | {"payload": json.loads(row["payload"])} for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def set_kill_switch(db: str | Path, *, enabled=True) -> dict:
    spec, bars, repo, env, _, _ = open_paper(db)
    try:
        env.set_kill_switch(enabled, reason="operator CLI kill switch")
        report = paper_report(spec, bars, env)
        save_json(Path(db).with_suffix(".report.json"), report)
        return {"kill_switch": env.risk_supervisor.kill_switch_active, "pending_orders": len(env.broker.pending_orders()), "holdings_retained": env.shares}
    finally:
        repo.close()
