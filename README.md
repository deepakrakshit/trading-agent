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

## One-command offline demonstration

```bash
make demo
```

On a fresh checkout this creates 24 **SYNTHETIC weekday** sessions, trains 3,000 Double-DQN interaction steps into the new `models/initial-v2/` experiment, saves/reloads checkpoints, runs three-seed walk-forward evaluation and cash/rule/random baselines, and restarts a SQLite shadow paper account under `runs/demo-v2/`. On subsequent runs it resumes that v2 account and reads its pinned evaluation receipt; it does **not** retune on that holdout. Preserved v1 artifacts are historical receipts and are rejected by the current loader rather than overwritten.

Actual first run: **₹9,502.08** final shadow equity, **311 paper fills / 29 closed inventory round trips**, and **₹188.32 explicit fees**. The candidate was rejected; the separate active-policy account remains **₹10,000 cash**. These synthetic results demonstrate the software, not a profitable strategy. Details: `docs/model_evaluation.md`.

## Individual workflow commands

Run from the repository root. The first `generate` command is for a fresh path: snapshots are immutable and refuse overwrite.

```bash
# Runtime/device evidence
.venv/bin/python -m trading_agent.cli doctor

# Explicitly synthetic data; no credentials or network
.venv/bin/python -m trading_agent.cli generate --sessions 24 --output data/raw/synthetic.parquet

# Candidate training only; choose a NEW output for each experiment
.venv/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/research-v1 --steps 3000 --device cpu

# Resume training BEFORE evaluating that version (adds another 500 steps)
.venv/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/research-v1 --steps 500 --device cpu --resume

# Fixed, chronological evaluation; does not automatically approve the candidate
.venv/bin/python -m trading_agent.cli evaluate --experiment models/research-v1 --device cpu

# Explicit shadow account for an UNAPPROVED model; stop after 50 steps
.venv/bin/python -m trading_agent.cli paper --experiment models/research-v1 --db runs/research/account.sqlite --strategy candidate --shadow --max-steps 50

# Resume with the exact stored dataset, policy hash, configuration and cursor
.venv/bin/python -m trading_agent.cli paper --db runs/research/account.sqlite --resume

# Active policy only: without an accepted candidate, this safely uses cash-v1
.venv/bin/python -m trading_agent.cli paper --experiment models/initial-v2 --db runs/new-active/account.sqlite --strategy active

# Durable account and audit records
.venv/bin/python -m trading_agent.cli account --db runs/demo/account.sqlite
.venv/bin/python -m trading_agent.cli account --db runs/demo/account.sqlite --view fills --output runs/demo/fills.json
.venv/bin/python -m trading_agent.cli account --db runs/demo/account.sqlite --view equity --output runs/demo/equity.json

# Kill switch cancels LOCAL pending intents and retains holdings; never invents a flatten fill
.venv/bin/python -m trading_agent.cli kill --db runs/research/account.sqlite
.venv/bin/python -m trading_agent.cli kill --db runs/research/account.sqlite --clear

# Version history; rollback requires a previous accepted version
.venv/bin/python -m trading_agent.cli registry
.venv/bin/python -m trading_agent.cli registry --rollback --reason "human-reviewed regression"

# Same workload on CPU and CUDA only when CUDA actually verifies
.venv/bin/python -m trading_agent.cli benchmark --steps 500
```

### Controlled paper-experience learning

`--experience-db` trains a **new candidate** from recorded numerical transitions, not from loss narratives and not by updating active weights after a trade. New paper accounts write a `paper-experience-v1` manifest beside the SQLite database. Import verifies the source dataset and slice hashes, instrument, feature/action/reward/execution/cost/risk versions, policy/account identity, contiguous history digest, and cutoff. It rejects observations after the new training cutoff or with different provenance/features/action/risk/cost settings. The earlier holdout may become training history only in a **new** chronological experiment with a later untouched test block.

This exact additional offline cycle was executed:

```bash
.venv-cuda/bin/python -m trading_agent.cli generate --output data/raw/synthetic-expanded.parquet --sessions 48
.venv-cuda/bin/python -m trading_agent.cli train --data data/raw/synthetic-expanded.parquet --output models/paper-replay-v2 --steps 1000 --experience-db runs/demo/account.sqlite --device auto
.venv-cuda/bin/python -m trading_agent.cli evaluate --experiment models/paper-replay-v2 --device auto
```

It imported 449 transitions, performed 1,000 additional environment interactions and 347 optimizer updates in total, and was rejected on a later test window. No performance improvement is claimed. Recurring/always-on retraining is not configured; this release uses deliberate CLI cycles.

### Checkpoint and environment compatibility

The supported financial simulation is `trading_agent.paper_env.PaperTradingEnv`.
Artificial `episode_length` boundaries are task-defined terminal transitions:
the configured terminal closeout is included in the reward and DQN does not
bootstrap through the account reset. Only an explicit trainer collection
budget is represented as a bootstrap-capable truncation. Checkpoint files now
use semantic schema v2 and are validated against the target environment's
ordered feature schema, exact action/exposure mapping, reward, execution,
cost, risk, environment hash, instrument, config version, and policy ID.
Walk-forward files also require a matching SHA-256 training receipt.

Existing v1 checkpoints and paper cursor artifacts are preserved as historical
receipts but are rejected by the current loader with an explicit migration or
retraining error; there is no silent compatibility fallback. See
`docs/t1_hardening.md` for the migration policy and regression evidence.

## GPU status and continuation

The optional `.venv-cuda` was actually installed with `requirements-cuda.lock` (**PyTorch 2.10.0+cu128**). Both CPU and CUDA-wheel environments pass the offline suite, but **CUDA tensor execution currently fails because the host kernel module is missing**. All actual training in this delivery ran on CPU. VRAM capacity/use could not be measured through a working driver.

Read **`docs/gpu_repair.md`** before system changes. Two simulated installs exposed incompatible NVIDIA package versions; no driver, kernel, Secure Boot setting, or system Python package was modified.

After a user-controlled Ubuntu driver/kernel repair and reboot:

```bash
.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda
make test PYTHON=.venv-cuda/bin/python
.venv-cuda/bin/python -m trading_agent.cli benchmark --steps 500 --output runs/benchmark-cuda.json
.venv-cuda/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/gpu-v1 --device cuda
```

An explicit `--device cuda` fails instead of silently using CPU. `auto` uses CUDA only after the tensor probe succeeds. A GPU wheel alone is not proof that the GPU works.

## Authorised historical/read-only data

See `docs/data_sources.md`. No real historical dataset, broker session, paid subscription, or genuinely live connection was available/used in this delivery.

Import your permitted **five-minute** OHLCV CSV with timezone-aware bar-start timestamps and these columns:

```text
symbol,timestamp,open,high,low,close,volume
```

```bash
.venv/bin/python -m trading_agent.cli import-csv permitted-history.csv --source MY_AUTHORISED_PROVIDER --output data/raw/history.parquet
.venv/bin/python -m trading_agent.cli train --data data/raw/history.parquet --output models/history-v1
```

Set authoritative holidays in `configs/default.json` before importing/evaluating real data. Missing/off-session/outlier bars fail closed; the loader never forward-fills a tradable price or silently substitutes synthetic data. Corporate actions and delivery-charge treatment require further source-specific work.

Optional SmartAPI commands consume **locally exported** `SMARTAPI_API_KEY` and `SMARTAPI_TOKEN` only. Obtain sessions through the official provider; this app has no login or real-order SDK. Verify eligibility, terms/costs, rate limits and token mapping locally first. Never put secrets in CLI arguments, prompts, Git, or logs.

```bash
.venv/bin/python -m trading_agent.cli fetch-history --symbol YOUR_EQUITY_SYMBOL --instrument-token YOUR_PUBLIC_INSTRUMENT_ID --start '2026-01-05T09:15:00+05:30' --end '2026-01-06T15:30:00+05:30' --output data/raw/authorised-smartapi.parquet
.venv/bin/python -m trading_agent.cli quote --symbol YOUR_EQUITY_SYMBOL --instrument-token YOUR_PUBLIC_INSTRUMENT_ID --count 3
```

Quote polling is labelled **READ_ONLY_POLL**, not verified LIVE. Stale last-trade timestamps are rejected even if a feed heartbeat is fresh. These adapters are mock-tested only; a continuously streaming live-paper runner is **not** claimed complete.

## Files and limitations

- `configs/default.json`: ₹10,000 balance; 25% single-name/gross cap; 2% daily loss lock; 10% drawdown lock; order-size/frequency guards; illustrative fee schedule.
- `models/initial-v1/`: preserved historical v1 checkpoint/evaluation receipt; current code rejects it without migration.
- `models/initial-v2/`: current v2 demo checkpoint, pinned snapshot manifest, compatibility contract, holdout receipt and walk-forward artifacts.
- `models/paper-replay-v2/`: subsequent replay-trained candidate and later evaluation.
- `models/registry.sqlite`: immutable registrations and rejection/promotion/rollback history.
- `runs/demo/`: preserved historical v1 SQLite account and receipt.
- `runs/demo-v2/`: current v2 SQLite account, detailed cost attribution, run receipt and diagnostic rule account.
- `runs/active/`: separate safe cash account, not an unapproved candidate.
- `docs/environment_report.md`, `docs/skills-audit.md`, `docs/test_report.md`, `docs/model_evaluation.md`, `docs/t1_hardening.md`: measured evidence, hardening changes, and gaps.

Checkpoints/data/logs and local virtual environments are ignored by Git; source/configuration/lockfiles/docs/tests are committed. No secrets or licensed datasets are committed. The main CLI uses `paper_env.PaperTradingEnv` and the canonical broker/ledger for training, evaluation and replay. `env.py` retains the small injected-execution reference environment for compatibility tests; it is not the CLI accounting engine.

Current limitations: one symbol per account; immutable replay datasets; full-fill-or-reject liquidity approximation; no verified exchange calendar/corporate actions/live-shadow history; no PPO comparison on meaningful real data; no dedicated statistical drift detector; no automatic schedule. Kill/loss locks can leave marked holdings when a safe exit is unavailable. They are not guarantees that gaps cannot exceed a loss threshold. No real execution path will be enabled automatically.
