# Self-improving paper trader

**PAPER TRADING ONLY — no real brokerage order route exists.** This is a local research system, not financial advice or a profitability claim. Canonical specification: `self_improving_trading_agent_blueprint.md`.

The project starts from a configurable **₹10,000 virtual INR** account, validates completed five-minute NSE-style candles, trains a compact Double-DQN, evaluates frozen policies against baselines, and runs all decisions through an independent risk supervisor and durable local paper broker.

## Setup (Linux x86_64)

Use **CPython 3.14.4**, pinned in `.python-version`. Dependencies and their transitive versions are in `requirements.lock`. No system Python packages, global installations or root privileges are used.

```bash
bash scripts/setup.sh
```

The current lock uses the official CPU PyTorch wheel. This laptop's NVIDIA kernel module is missing for the running kernel. See `docs/environment_report.md`; GPU use must be reverified after driver repair and a compatible CUDA wheel is installed in a separate project venv using the official PyTorch selector.

## Offline checks

```bash
make test
```

This command requires no network, broker account, credentials, or GPU after setup.

## Containers

The Dockerfile installs system/runtime dependencies, installs the lock, and runs the full offline test suite while building. Docker is optional and is not currently installed on the audited laptop.

```bash
docker build -t paper-trader .
docker run --rm paper-trader
```

For durable local output, bind-mount a writable `runs/` directory (the container runs as UID 10001). This is a local batch/CLI deployment, not a hosted service or scheduler. Do not expose provider secrets in image layers.

## Safety and research boundaries

- Cash equities only; integer shares, no leverage or shorting.
- SYNTHETIC development fixtures are never passed off as historical/live data.
- Next-bar-open fills with adverse execution costs; no same-close look-ahead.
- Candidates remain separate from active policies, with fixed promotion gates and rollback.
- API credentials are not necessary for the offline workflow and must never be pasted into chat or committed.
- The official SmartAPI FAQ/docs were inaccessible during the audit. Read-only adapters are tested against mocks only until terms/account access and real responses are independently verified.

Full workflow commands and measured results are added below as integration is verified.
