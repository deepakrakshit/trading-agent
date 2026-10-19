"""Local command-line interface. All portfolio changes are simulated."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .config import load_config
from .data import CSVBarProvider
from .data.persistence import write_snapshot
from .diagnostics import benchmark, doctor, resolve_device
from .registry import CandidateRegistry
from .runner import inspect_account, run_paper, set_kill_switch
from .workflow import (
    evaluate_experiment, generate_synthetic, jsonable, read_json, save_json, train_experiment,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Cash-only PAPER trading research. No real order routes.")
    root.add_argument("--config", default="configs/default.json")
    sub = root.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("doctor", help="Verify Python/PyTorch and real CUDA tensor execution")
    probe.add_argument("--require-cuda", action="store_true")
    probe.add_argument("--output")
    generate = sub.add_parser("generate", help="Create explicitly SYNTHETIC five-minute candles")
    generate.add_argument("--output", default="data/raw/synthetic.parquet")
    generate.add_argument("--sessions", type=int, default=24)
    generate.add_argument("--seed", type=int, default=2026)
    imported = sub.add_parser("import-csv", help="Import user-authorized historical CSV; never fetch an exchange website")
    imported.add_argument("path")
    imported.add_argument("--output", required=True)
    imported.add_argument("--source", required=True)
    training = sub.add_parser("train", help="Train/checkpoint a candidate without changing the active policy")
    training.add_argument("--data", default="data/raw/synthetic.parquet")
    training.add_argument("--output", default="models/initial-v1")
    training.add_argument("--steps", type=int)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--experience-db")
    evaluation = sub.add_parser("evaluate", help="Fixed chronological baselines, walk-forward and promotion decision")
    evaluation.add_argument("--experiment", default="models/initial-v1")
    evaluation.add_argument("--registry", default="models/registry.sqlite")
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    paper = sub.add_parser("paper", help="Persistent offline replay; unapproved policies require --shadow")
    paper.add_argument("--experiment")
    paper.add_argument("--db", default="runs/paper/account.sqlite")
    paper.add_argument("--strategy", choices=("active", "candidate", "cash", "rule"), default="active")
    paper.add_argument("--registry", default="models/registry.sqlite")
    paper.add_argument("--shadow", action="store_true")
    paper.add_argument("--resume", action="store_true")
    paper.add_argument("--max-steps", type=int)
    account = sub.add_parser("account", help="Inspect durable account, equity, fills, orders or transitions")
    account.add_argument("--db", default="runs/demo/account.sqlite")
    account.add_argument("--view", choices=("summary", "orders", "fills", "equity", "transitions", "events"), default="summary")
    account.add_argument("--output", help="Optional JSON artifact path")
    kill = sub.add_parser("kill", help="Latch or explicitly clear the paper kill switch; retain holdings")
    kill.add_argument("--db", required=True)
    kill.add_argument("--clear", action="store_true")
    bench = sub.add_parser("benchmark", help="Same seeded training workload on CPU and CUDA when verified")
    bench.add_argument("--data", default="data/raw/synthetic.parquet")
    bench.add_argument("--steps", type=int, default=500)
    bench.add_argument("--output", default="runs/benchmark.json")
    demo = sub.add_parser("demo", help="Generate, train, evaluate, reload and restart a shadow paper account offline")
    demo.add_argument("--data", default="data/raw/synthetic.parquet")
    demo.add_argument("--experiment", default="models/initial-v1")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    demo.add_argument("--steps", type=int)
    registry = sub.add_parser("registry", help="Inspect immutable promotion/rollback history")
    registry.add_argument("--path", default="models/registry.sqlite")
    registry.add_argument("--rollback", action="store_true")
    registry.add_argument("--reason")
    for name in ("fetch-history", "quote"):
        provider = sub.add_parser(name, help="Optional Angel One READ-ONLY data; requires local auth and verified terms")
        provider.add_argument("--symbol", required=True)
        provider.add_argument("--instrument-token", required=True, help="Public instrument ID, NOT a secret session token")
        if name == "fetch-history":
            provider.add_argument("--start", required=True, help="ISO timestamp including UTC offset")
            provider.add_argument("--end", required=True)
            provider.add_argument("--output", required=True)
        else:
            provider.add_argument("--count", type=int, default=1)
    return root


def demo(args, config) -> dict:
    data, experiment, output = Path(args.data), Path(args.experiment), Path(args.output)
    if not data.exists():
        generate_synthetic(data)
    if not (experiment / "experiment.json").exists():
        train_experiment(data, experiment, config, steps=args.steps, device=resolve_device(args.device))
    evaluation = evaluate_experiment(experiment, device=resolve_device(args.device))
    db = output / "account.sqlite"
    receipt_path = output / "demo.json"
    restart_verified = False
    if not db.exists():
        partial = run_paper(experiment, db, strategy="candidate", shadow=True, max_steps=50)
        restart_verified = inspect_account(db)["account"] == partial["account"]
    elif receipt_path.exists():
        restart_verified = read_json(receipt_path).get("restart_verified", False)
    paper = run_paper(None, db, resume=True)
    rule_db = output / "rule_account.sqlite"
    rule = run_paper(experiment if not rule_db.exists() else None, rule_db, strategy="rule", shadow=True, resume=rule_db.exists())
    report = {
        "paper_only": True, "provenance": paper["provenance"], "experiment": str(experiment),
        "checkpoint": str(experiment / read_json(experiment / "experiment.json")["checkpoint"]),
        "paper_account": paper["account"], "paper_metrics": paper["metrics"],
        "rule_account": rule["account"], "rule_metrics": rule["metrics"],
        "test_metrics": evaluation["test"]["metrics"],
        "baselines": {name: item["metrics"] for name, item in evaluation["baselines"].items()},
        "promotion": evaluation["promotion"], "restart_verified": restart_verified,
        "note": "SYNTHETIC replay proves engineering behavior only; not historical/live profitability.",
    }
    save_json(receipt_path, report)
    return report


def provider_command(args, config):
    from .data.calendar import SessionCalendar
    from .data.smartapi import SmartAPIClient, SmartAPIHistoricalProvider, SmartAPIQuoteProvider
    token, key = os.environ.get("SMARTAPI_TOKEN"), os.environ.get("SMARTAPI_API_KEY")
    if not token or not key:
        raise ValueError("Set SMARTAPI_TOKEN and SMARTAPI_API_KEY locally; never paste secrets into chat or CLI arguments")
    client = SmartAPIClient(token=token, api_key=key, symbol_tokens={args.symbol: args.instrument_token})
    calendar = SessionCalendar(holidays=config["risk"]["closed_dates"])
    if args.command == "fetch-history":
        provider = SmartAPIHistoricalProvider(client, calendar=calendar)
        bars = provider.get_bars(args.symbol, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return write_snapshot(destination, bars, source=provider.source, provenance=provider.provenance, source_metadata=provider.source_metadata).to_mapping()
    provider = SmartAPIQuoteProvider(client, calendar=calendar)
    if not 1 <= args.count <= 100:
        raise ValueError("quote count must be in [1,100]")
    return [asdict(quote) | {"freshness_seconds": (quote.received_at - quote.timestamp).total_seconds()} for quote in provider.poll(args.symbol, count=args.count)]


def dispatch(args, config):
    if args.command == "doctor":
        report = doctor()
        if args.output:
            save_json(args.output, report)
        return report
    if args.command == "generate":
        return generate_synthetic(args.output, sessions=args.sessions, seed=args.seed)
    if args.command == "import-csv":
        bars = CSVBarProvider(args.path, provenance="HISTORICAL").all_bars()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_snapshot(output, bars, source=args.source, provenance="HISTORICAL", source_metadata={"timestamp_convention": "BAR_START", "interval": "FIVE_MINUTE"}).to_mapping()
    if args.command == "train":
        return train_experiment(args.data, args.output, config, steps=args.steps, seed=args.seed, device=resolve_device(args.device), resume=args.resume, experience_db=args.experience_db)
    if args.command == "evaluate":
        report = evaluate_experiment(args.experiment, registry_path=args.registry, device=resolve_device(args.device))
        return {"report": str(Path(args.experiment) / "evaluation.json"), "test": report["test"]["metrics"], "baselines": {name: row["metrics"] for name, row in report["baselines"].items()}, "walk_forward_summary": report["walk_forward_summary"], "promotion": report["promotion"], "provenance": report["provenance"]}
    if args.command == "paper":
        report = run_paper(args.experiment, args.db, strategy=args.strategy, shadow=args.shadow, resume=args.resume, max_steps=args.max_steps, registry_path=args.registry)
        return {key: value for key, value in report.items() if key != "attribution"} | {"detailed_report": str(Path(args.db).with_suffix(".report.json"))}
    if args.command == "account":
        result = inspect_account(args.db, view=args.view)
        if args.output:
            save_json(args.output, result)
        return result
    if args.command == "kill":
        return set_kill_switch(args.db, enabled=not args.clear)
    if args.command == "benchmark":
        return benchmark(args.data, config, args.output, steps=args.steps)
    if args.command == "demo":
        return demo(args, config)
    if args.command == "registry":
        if not Path(args.path).is_file():
            raise FileNotFoundError("registry does not exist; evaluate an experiment first")
        with CandidateRegistry(args.path) as registry:
            if args.rollback:
                if not args.reason:
                    raise ValueError("rollback requires --reason")
                return asdict(registry.rollback(reason=args.reason))
            return {"active_policy_id": registry.active_policy_id, "gates": asdict(registry.gates), "history": registry.history()}
    if args.command in ("fetch-history", "quote"):
        return provider_command(args, config)
    raise ValueError("unknown command")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args, load_config(args.config))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(json.dumps({"status": "error", "type": type(error).__name__, "message": str(error), "paper_only": True}), file=sys.stderr)
        return 1
    print(json.dumps(result, default=jsonable, sort_keys=True, indent=2, allow_nan=False))
    if args.command == "doctor" and args.require_cuda and not result["cuda_verified"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
