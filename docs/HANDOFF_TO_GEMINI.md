# Handoff to Gemini Antigravity

## Metadata

- Workspace: `/home/deepakrakshit/trading_agent`
- Repository: `https://github.com/deepakrakshit/trading-agent.git`
- Branch: `main`
- Revision before this handoff file: `c1895733decf6de16eabd32adfbdf4ba739c28bb`
- Runtime: CPython `3.14.4`, pinned by `.python-version`
- Date: 20 October 2026
- Product boundary: **paper trading only**. No real order route, broker order endpoint, credentials, or financial execution exists.
- Current Git status before creating this file: clean. The three latest local commits are:
  - `53dbfb9 feat: add continuous paper replay runner`
  - `ef19f24 docs: record continuous paper handoff`
  - `c189573 docs: record final reproducibility receipt`
- The handoff was first committed as `1d5517e`, finalized in the documentation commits through `e0ab421`, and pushed to `origin/main`; verify the remote before starting new work.

## Executive Summary

The repository is an offline, reproducible, cash-only, long-only paper-trading
research system. The existing finite historical replay runner was kept intact;
the missing provider-independent completed-bar boundary has now been added as a
separate capability.

The new implementation provides:

- A narrow `CompletedBarFeed` protocol with explicit `ProviderIdentity`.
- Deterministic `ReplayBarFeed` and finite `RangeProviderFeed` adapters.
- `CompletedBarEvent` records containing bar time, completion/availability time,
  local receipt time, provenance and deterministic provider event identity.
- Bounded retry behavior that never substitutes cached or synthetic data for a
  failed provider.
- `ContinuousPaperRunner`, reusing the canonical `PaperBroker`,
  `PortfolioLedger`, `RiskSupervisor` and `SQLiteRepository`.
- Durable runner cursor/recovery, stale-feed health records, graceful stop and a
  persistent paper-only kill switch.
- A `paper-stream` CLI command for deterministic snapshot replay and resume.

This is verified offline/replay behavior only. It is **not** a verified live NSE
feed, a deployment service, or a profitability result. SmartAPI remains
mock-tested and no real provider entitlement or credential was used.

## What Exists

### Canonical trading path

The supported financial path is:

```text
validated MarketBar snapshot/provider
  -> CausalFeatureBuilder
  -> injectable policy/action
  -> RiskSupervisor
  -> PaperBroker
  -> PortfolioLedger
  -> SQLiteRepository/events/transitions/checkpoints
```

Important existing components:

- `src/trading_agent/data/bars.py` — validated completed OHLCV bars and
  provenance values (`SYNTHETIC`, `HISTORICAL`, `LIVE_READ_ONLY`).
- `src/trading_agent/data/providers.py` — read-only finite
  `MarketDataProvider`, in-memory provider, CSV provider and synthetic fixture.
- `src/trading_agent/data/persistence.py` — immutable CSV/Parquet snapshots,
  source metadata allowlist, manifest and SHA-256 checks.
- `src/trading_agent/features/causal.py` — completed-bar-only rolling features.
- `src/trading_agent/execution/paper.py` — the only portfolio-changing route;
  all fills are local paper fills.
- `src/trading_agent/portfolio/ledger.py` — Decimal cash, integer shares,
  fees, basis, realized/unrealized P&L and account invariants.
- `src/trading_agent/risk/supervisor.py` — stale-feed, session, exposure,
  frequency, loss/drawdown and kill-switch guards.
- `src/trading_agent/storage/sqlite.py` — transactional event, order, fill,
  transition and state persistence.
- `src/trading_agent/paper_env.py` — canonical Gymnasium training/evaluation
  environment.
- `src/trading_agent/runner.py` — existing finite restartable paper replay;
  do not replace it with the new continuous class.
- `src/trading_agent/learning.py`, `workflow.py`, `evaluation.py` and
  `registry.py` — DQN training, chronological evaluation and promotion gates.

### New continuous/replay path

- `src/trading_agent/data/stream.py`
  - `ProviderIdentity`
  - `CompletedBarEvent`
  - `CompletedBarFeed` protocol
  - `ReplayBarFeed`
  - `RangeProviderFeed`
  - `FeedError` / `FeedUnavailableError`
  - `RetryPolicy` / `fetch_with_retry`
- `src/trading_agent/continuous.py`
  - `ContinuousPaperRunner`
  - `ContinuousRunnerStatus`
  - event processing, polling, retry, stale health, cursor persistence,
    transition persistence, restart validation, graceful shutdown and kill latch.
- `src/trading_agent/cli.py`
  - `paper-stream --data ... --db ... --strategy cash|rule`
  - `--resume`, `--max-bars`, `--kill`, and `--clear-kill`.
- `tests/test_continuous.py` and `tests/test_cli.py`
  - replay/range feed behavior, completion/receipt clocks, monotonicity,
    provider mismatch, stale events, retries, rollback, restart, polling,
    persistent kill switch and CLI resume.

### Timing and causality

- Bar timestamps are UTC `BAR_START` timestamps.
- A five-minute bar starting at `T` is available at `T + 5 minutes`.
- The continuous runner makes its policy observation after the completed bar is
  observed and stores an action for the following event boundary.
- `PaperBroker.EXECUTION_VERSION` is now `causal-open-v4`.
- `process_open()` permits a stream order whose decision becomes available at
  the exact same timestamp as the eligible open event.
- The older `process_bar()` convenience path explicitly retains strict
  next-bar behavior for direct compatibility tests.
- Open execution sees only symbol, timestamp, open and explicit event-time
  liquidity. It never uses the arriving candle's future high, low, close or
  volume to decide an open fill. The default historical liquidity proxy is the
  previously observed candle's volume.

### Compatibility and artifacts

- The execution-version change intentionally invalidates pre-v4 environment,
  paper-account and checkpoint compatibility artifacts.
- `models/initial-v1/`, `models/paper-replay-v2/`, `models/initial-v2/` and
  prior `runs/` artifacts are preserved and must not be deleted or overwritten.
- The default compatible demo path is now `models/initial-v3/` and
  `runs/demo-v3/`. These generated artifacts are local/ignored; source Git
  archives do not include them.

## What Is Missing

The following are deliberately still incomplete or unverified:

1. **No verified live provider.** SmartAPI historical/quote adapters are
   narrow and mock-tested only. There is no WebSocket/live adapter, entitlement
   proof, live latency measurement or real shadow run.
2. **The CLI continuous command is finite snapshot replay.** The core runner
   has injectable `run_polling()`, clock, sleep, stop-event and feed interfaces,
   but no daemon, scheduler, service unit or verified live deployment is
   configured.
3. **The CLI exposes cash and rule policies.** A compatible callable can be
   injected into `ContinuousPaperRunner`; a dedicated DQN-checkpoint-to-stream
   CLI adapter is not implemented. If adding one, reuse the existing
   `PaperTradingEnv.checkpoint_compatibility()` and
   `DQNTrainer.load_checkpoint()` checks rather than bypassing them.
4. **No authorized historical dataset is present.** Existing data is explicitly
   synthetic and cannot support a market-performance claim.
5. **CUDA is not usable on the Linux host.** The PCI device is visible but the
   running kernel has no matching NVIDIA module.
6. **Docker is not installed on the audited host.** The Dockerfile is
   maintained but its build/run was not executed here.
7. Windows installation and GPU execution are not yet verified. The Linux lock
   files must not be assumed to be Windows lock files.
8. Research gaps remain: genuine multi-regime history, corporate actions,
   realistic provider-specific execution stress, PPO comparison, drift
   monitoring and statistically sufficient live-shadow evidence.

## Known Safe Architecture

Do not create a second ledger, broker, risk layer, feature implementation or
paper-account format. The safe extension point is:

```python
class CompletedBarFeed(Protocol):
    identity: ProviderIdentity

    def fetch(symbol, after, until, *, as_of) -> Sequence[CompletedBarEvent]: ...
```

Provider adapters may read candles/quotes, normalize them into validated
`MarketBar`/`CompletedBarEvent` objects and report failures. They may not place,
modify or cancel real orders.

`ContinuousPaperRunner` is the sole consumer of the stream contract. It:

- Checks provider identity, symbol, availability boundary, receipt-clock order,
  strict chronological order and same-session interval gaps.
- Persists each accepted stream bar, decision, transition, health record and
  cursor using the existing SQLite repository.
- Couples runner cursor changes with broker/ledger changes in one rollback unit.
- Rejects stale execution through the existing risk supervisor and suppresses a
  new policy action after a stale event until a fresh event arrives.
- Does not flatten holdings when the kill switch is engaged. It cancels local
  pending paper intents and retains holdings; any future safe exit remains
  subject to the existing broker/risk rules.
- Stores provider, provenance, policy, feature, reward, execution, cost and
  risk metadata on transitions.

The model must remain a policy/action source only. It cannot change risk limits,
accounting, provenance or execution semantics.

## Exact Commands to Run

Run from the repository root. Use project-local interpreters only.

### Linux CPU baseline

```bash
python3 -m venv .venv-repro
.venv-repro/bin/python -m pip install --cache-dir .cache/pip --timeout 30 --retries 2 -r requirements.lock
.venv-repro/bin/python -m pip install --no-deps --no-build-isolation -e .
make test PYTHON=.venv-repro/bin/python
.venv-repro/bin/python -m trading_agent.cli doctor
```

Expected current baseline: **259 passed**, compilation passed, `pip check`
passed, and the doctor command reports CPU success but no verified CUDA.

### Offline data, training and evaluation

```bash
.venv/bin/python -m trading_agent.cli generate --sessions 24 --output data/raw/synthetic.parquet
.venv/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/research-v1 --steps 3000 --device cpu
.venv/bin/python -m trading_agent.cli evaluate --experiment models/research-v1 --device cpu
```

Do not overwrite an existing immutable snapshot or model directory. Select a
new output directory for every new experiment.

### Current default demo

```bash
make demo
```

The current default creates/resumes `models/initial-v3/` and
`runs/demo-v3/`. The observed successful run completed restart verification and
rejected the synthetic candidate at ₹9,582.25 after 260 fills and ₹149.70
fees. A first attempt with the old v2 default failed because the pre-v4
checkpoint was correctly rejected; the CLI default was moved to v3 rather than
overwriting v2. The successful `make demo` run required a 300-second tool
timeout on this host.

### Continuous completed-bar replay

```bash
.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db runs/stream/account.sqlite \
  --strategy cash \
  --max-bars 100

.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db runs/stream/account.sqlite \
  --strategy cash \
  --resume \
  --max-bars 100
```

Use `--strategy rule` only as a diagnostic shadow policy. It is not an
approved learned model. To persistently block new local paper orders while
retaining holdings:

```bash
.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db runs/stream/account.sqlite \
  --strategy cash --resume --kill

.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db runs/stream/account.sqlite \
  --strategy cash --resume --clear-kill
```

The current CLI replays only events available in the supplied immutable
snapshot after the persisted cursor. It never silently falls back to a
different data source.

### Account and source-archive checks

```bash
.venv-repro/bin/python -m trading_agent.cli account --db runs/stream/account.sqlite
git diff --check
tmp=$(mktemp -d /tmp/omnirush/archive-check.XXXXXX)
git archive HEAD | tar -x -C "$tmp"
make -C "$tmp" test PYTHON=/home/deepakrakshit/trading_agent/.venv-repro/bin/python
```

The final source-archive check at revision `c189573` passed 259 tests,
compilation and `pip check`. After committing this handoff file, repeat the
archive check against the new HEAD and update this document only if the result
changes.

### Commands intentionally not run or not successful

```bash
docker --version
.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda
.venv-cuda/bin/python -m trading_agent.cli train --device cuda --steps 16
```

Docker is unavailable. The CUDA doctor command exits 2 and explicit CUDA
training exits before creating a run; these are expected environment blockers,
not successful GPU evidence.

## Training/Evaluation

All reported model results are synthetic engineering evidence and all learned
candidates were rejected:

| Artifact | Result | Status |
| --- | --- | --- |
| `models/initial-v1/` | ₹9,502.08 final equity, 311 fills, ₹188.32 fees, approximately -4.9792% holdout return | Rejected; preserved historical pre-v2 artifact |
| `models/paper-replay-v2/` | approximately -6.1238% later synthetic holdout, 449 imported transitions | Rejected; preserved historical artifact |
| `models/initial-v2/` | ₹9,698.67 final equity, 183 fills, ₹67.88 fees after the 256-step CPU audit | Rejected and now pre-v4 incompatible; preserved |
| `models/cpu-audit-v1/` | 64 CPU steps, 0.5981 seconds, approximately -2.2408% holdout return, 2.4074% drawdown, ₹16.33 fees, 21 fills | Rejected |
| `models/initial-v3/` | ₹9,582.25 final shadow equity, 260 fills, ₹149.70 fees; rule diagnostic account ₹9,255.21, 230 fills, ₹274.39 fees | Rejected; current execution-v4-compatible demo |

The active-policy fallback remains cash. No learned candidate has been
promoted. Keep the train/evaluate/promote separation: a training command must
write a new candidate and must not alter the active policy or risk limits.

For a learned policy in the continuous runner, the safe future integration is:

1. Load the checkpoint through `DQNTrainer.load_checkpoint()`.
2. Construct the exact compatible environment/config and call
   `PaperTradingEnv.checkpoint_compatibility(policy_id=...)`.
3. Reject any mismatch in observation dimension, ordered feature schema,
   action/exposure mapping, reward/execution/cost/risk versions, environment
   hash, instrument or config version.
4. Inject only `policy.act` plus a stable policy ID/metadata into
   `ContinuousPaperRunner`.
5. Keep the account labelled shadow/paper and do not change registry active
   state automatically.

## GPU Setup Requirements

### Observed Linux blocker

- PCI reports an NVIDIA GeForce RTX 2050 (GA107), but capacity/use is not
  runtime verified.
- Current kernel: `7.0.0-34-generic`.
- `nvidia-smi` exits 9 and cannot communicate with the driver.
- `modinfo nvidia` reports no module for the running kernel.
- PyTorch CUDA wheel: `2.10.0+cu128`; `torch.cuda.is_available()` is false and
  device count is zero.
- No driver, kernel, Secure Boot, reboot, `sudo`, or apt repair was attempted.

Do not attempt to fix this from the project or install system drivers from an
agent session. A user/admin must perform any signed driver/kernel repair and
reboot, then rerun:

```bash
nvidia-smi
.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda
make test PYTHON=.venv-cuda/bin/python
.venv-cuda/bin/python -m trading_agent.cli benchmark --steps 500 --output runs/benchmark-cuda.json
.venv-cuda/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/gpu-v1 --device cuda
```

Only report GPU training after both `nvidia-smi` and the real PyTorch tensor
probe pass. `--device cuda` must fail rather than silently fall back to CPU.

### Windows requirements

The Windows handoff must be project-local and reproducible:

1. Install/use CPython **3.14.4** and confirm `python --version`.
2. Create a repository-local `.venv` or `.venv-cuda`; do not install globally.
3. Do not assume `requirements.lock` or `requirements-cuda.lock` (Linux wheel
   selections) are Windows-compatible. Use the official PyTorch selector for
   a CUDA wheel compatible with the installed Windows driver, then record the
   successful package set in a Windows-specific project lock file if needed.
4. Run `nvidia-smi`, the doctor command and an actual CUDA tensor probe before
   training. The expected RTX 2050 is 4 GiB class hardware, but do not treat
   that as observed VRAM until the probe succeeds.
5. Run the full test suite in the Windows project environment. If a dependency
   or path differs, make the smallest project-local, pinned change and record
   it; do not weaken compatibility checks or add a global install.
6. Prefer small batch/replay sizes and CPU fallback for CPU verification. Keep
   GPU-specific output in a new artifact directory such as `models/windows-gpu-v1/`.

## Dataset/Provider Integration

### Authorized historical CSV

The supported no-network path is a user-authorized five-minute OHLCV CSV:

```text
symbol,timestamp,open,high,low,close,volume
```

Use timezone-aware `BAR_START` timestamps, declare the actual provider/source,
and import into a new immutable snapshot:

```bash
.venv/bin/python -m trading_agent.cli import-csv \
  permitted-history.csv \
  --source MY_AUTHORISED_PROVIDER \
  --output data/raw/authorised-history.parquet
```

Verify the generated manifest, source metadata, capture boundary, interval,
calendar version, instrument mapping and provenance before training. Do not
label CSV data historical unless the user is authorized to use it.

### Provider adapter contract

For a finite provider, implement `MarketDataProvider.get_bars(symbol, start,
end)` and wrap it with `RangeProviderFeed`. For an incremental provider,
implement `CompletedBarFeed.fetch(...)` and return only completed events. Each
event must include:

- stable provider ID and provenance;
- interval/timestamp convention/calendar/instrument mapping metadata;
- bar start time;
- `available_at = bar.timestamp + interval`;
- local `received_at >= available_at`;
- stable provider event ID;
- no credentials or raw provider payloads in persisted metadata.

Provider errors should become `FeedUnavailableError` only when bounded retry is
safe. Authentication, entitlement, schema or authorization failures should
stop rather than loop. Never substitute a synthetic bar, cached quote or
forward-filled price while calling the source live/historical.

The existing SmartAPI implementation is intentionally read-only and narrow.
It has no login or order route. Keep that boundary if real access is later
authorized; do not install the broad trading SDK or add order endpoints.

## Operational Caveats

- One SQLite file represents one paper account. Do not run two writers against
  the same account.
- A runner cursor and broker checkpoint must agree on revision, configuration,
  environment hash and ledger hash. Treat a mismatch as a recovery failure,
  not as a reason to guess or reset.
- A stale event is recorded as health state and does not receive a new policy
  action. The existing risk supervisor also rejects a stale fill attempt.
- Monotonicity is checked for bar timestamps and local receipt timestamps.
  Missing/irregular intraday bars inside one configured session fail closed.
- A kill switch blocks new exposure and cancels pending local paper intents; it
  does not secretly flatten holdings or create an invented exit fill.
- `run_polling()` is a library seam with injected clock/sleep/stop controls. It
  is not an always-on process manager, scheduler or proof of live reliability.
- `paper-stream` snapshot replay is finite and uses the snapshot's manifest
  provenance. It is not allowed to reinterpret `SYNTHETIC` as historical/live.
- Generated `data/`, `models/`, `runs/`, virtual environments and caches are
  ignored by Git. Preserve them locally; do not use `git clean`, `reset --hard`
  or broad deletion while investigating.
- Warnings in the 259-test run are known legacy-environment deprecations and
  advisory unbounded-Box warnings; they were not suppressed.
- No dedicated linter or static type checker is configured. Compilation and
  `pip check` pass, but do not report lint/typecheck coverage.
- Docker was not run because `docker` is not installed. A clean source archive
  and project-local venv were used for reproducibility evidence instead.

## What Not to Change

Do not:

- add real broker order placement, modification, cancellation, login or
  credential handling;
- paste or commit API keys, access tokens, cookies, client codes, PINs or raw
  provider payloads;
- bypass `RiskSupervisor`, `PaperBroker`, `PortfolioLedger`, compatibility
  validation, provenance validation or SQLite transactions;
- create another accounting engine for streaming;
- use completed-candle high/low/close/volume to fill at that candle's open;
- silently migrate pre-v4 checkpoints/accounts or change version strings to
  accept incompatible artifacts;
- overwrite `models/initial-v1/`, `models/paper-replay-v2/`,
  `models/initial-v2/`, `runs/demo/`, `runs/demo-v2/` or other preserved
  receipts;
- relabel synthetic data as historical/live, scrape exchange websites, or
  use a synthetic fallback during provider outages;
- install dependencies globally, use `sudo`, modify system drivers, alter
  Secure Boot, or add an OS scheduler as a substitute for a project feature;
- weaken or remove the offline socket-blocking tests and compatibility tests;
- claim a positive strategy, a live feed, a working GPU or a Docker build
  without the exact evidence.

## Checks

All commands below were run from `/home/deepakrakshit/trading_agent` unless a
clean archive path is shown.

### Tests — passed

- `make test` — **259 passed**, 65 known warnings; compileall passed; `pip check`
  passed.
- `make test PYTHON=.venv-cuda/bin/python` — **259 passed**; this exercised the
  CUDA-enabled wheel in CPU mode, not GPU execution.
- `make test PYTHON=.venv-repro/bin/python` — **259 passed**, compilation and
  `pip check` passed.
- `pytest -q tests/test_continuous.py tests/test_cli.py` — targeted stream/CLI
  tests passed during implementation.
- `git diff --check` — passed before each relevant commit/finalization.

### Build/package checks — passed or explicit gap

- `.venv/bin/python -m compileall -q src tests` — passed through `make test`.
- `.venv-cuda/bin/python -m compileall -q src tests` — passed through `make test`.
- `.venv-repro/bin/python -m compileall -q src tests` — passed through `make test`.
- `pip check` in all three project-local environments — passed.
- `make demo` — passed using `models/initial-v3/` and `runs/demo-v3/`; restart
  verification passed; synthetic promotion was rejected. The first attempt
  with a 120-second timeout was terminated before completion; the same command
  was rerun with a 300-second timeout and passed.
- Clean source archive at pre-handoff revision `c189573`:
  `git archive HEAD | tar -x -C <tmp>` followed by
  `make -C <tmp> test PYTHON=/home/deepakrakshit/trading_agent/.venv-repro/bin/python`
  — **259 passed**, compileall and `pip check` passed.
- The same clean source-archive check was repeated at handoff commit `1d5517e`
  and again passed **259 tests**, compilation and `pip check`.
- The last full source-archive check at `d37f398` also passed **259 tests**,
  compilation and `pip check`; the later `e0ab421` change was documentation-only
  and was rechecked with `make test`.
- Docker build/run — **not run**; `docker --version` is unavailable.
- Dedicated lint/typecheck — **not configured/not run**.

### Device checks — expected blocker

- `.venv-repro/bin/python -m trading_agent.cli doctor` — passed CPU tensor
  check, reported `cuda_verified: false`.
- `.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda` — exits 2;
  no usable NVIDIA driver/device.
- `.venv-cuda/bin/python -m trading_agent.cli train --device cuda ...` — exits
  before creating a run; no CPU fallback is claimed.

### Manual stream smoke — passed

```bash
.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db /tmp/omnirush/paper-stream.sqlite \
  --strategy rule --max-bars 5
.venv-repro/bin/python -m trading_agent.cli paper-stream \
  --data data/raw/synthetic.parquet \
  --db /tmp/omnirush/paper-stream.sqlite \
  --strategy rule --resume --max-bars 5
```

Observed result: first batch processed 5 events; second batch processed 5
additional events; durable transition count reached 9. The output explicitly
reported `paper_only: true` and `provenance: SYNTHETIC`.

## Recommended Next Steps

1. Verify `git status --short`, `git log --oneline -5`, and that `origin/main`
   contains `e0ab421` before changing anything.
2. On Windows, create project-local Python 3.14.4 environments and run the
   full offline suite before attempting any data/GPU work. Treat Windows
   dependency resolution as a separate pinned lockfile task.
3. If the user supplies authorized historical CSV data, import it into a new
   immutable snapshot, inspect the manifest and calendar metadata, and run the
   existing chronological train/evaluate path. Do not overwrite synthetic
   artifacts.
4. If live paper research is explicitly requested and authorized, implement a
   read-only provider adapter behind `CompletedBarFeed`, translate only safe
   transient failures to `FeedUnavailableError`, add mocked and replay
   regressions, and run a long shadow observation before drawing any research
   conclusion. Do not add order routes.
5. If a learned policy must run through the continuous runner, add the strict
   checkpoint compatibility adapter described in **Training/Evaluation** and
   keep the policy in a separate shadow account.
6. Keep `docs/test_report.md`, `docs/environment_report.md`,
   `docs/data_sources.md`, this handoff and the blueprint synchronized with
   actual commands/results. Never turn a planned or blocked check into a
   passed claim.

## Ready-to-Paste Antigravity Prompt

```text
You are taking over /home/deepakrakshit/trading_agent on Windows.

Read these files first, in this order:
1. docs/HANDOFF_TO_GEMINI.md
2. README.md
3. self_improving_trading_agent_blueprint.md
4. docs/test_report.md
5. docs/environment_report.md
6. src/trading_agent/continuous.py
7. src/trading_agent/data/stream.py
8. src/trading_agent/runner.py
9. src/trading_agent/execution/paper.py
10. src/trading_agent/risk/supervisor.py

Current local implementation goal:
- Preserve the existing finite replay runner.
- Use the new provider-independent CompletedBarFeed/CompletedBarEvent boundary
  and ContinuousPaperRunner for completed-bar paper replay only.
- Keep all portfolio changes inside PaperBroker -> PortfolioLedger and all
  independent safety decisions inside RiskSupervisor.
- Do not create a second ledger/accounting path.

Hard safety constraints:
- PAPER ONLY. Never add real broker order placement, modification, cancellation,
  login or credential handling.
- Never ask me to paste secrets into chat. Use local environment placeholders
  only and keep secrets out of Git/logs/reports.
- Never relabel synthetic data as historical/live and never substitute synthetic
  or cached data during a provider failure.
- Do not install globally, use sudo, change Secure Boot, repair drivers, or
  modify system settings from the agent.
- Preserve all existing models/, runs/ and data/ artifacts. Do not run git
  clean, reset --hard, broad deletion or overwrite an existing immutable path.
- Do not weaken compatibility/version checks. The current execution contract is
  causal-open-v4; pre-v4 v1/v2 artifacts are historical and intentionally
  rejected. Use new output directories.

First actions:
1. Run git status, git log --oneline -5 and verify the handoff paths.
2. Check python --version, nvidia-smi and the project-local environment. The
   expected host GPU/driver status is documented as blocked; do not infer CUDA
   from PCI detection.
3. Create only project-local Windows environments using Python 3.14.4. Do not
   assume the Linux lockfiles install on Windows; record any Windows-specific
   lockfile only after a reproducible successful install.
4. Run the offline suite before changing code:
   make test PYTHON=<windows-project-venv-python>
5. Run the doctor/tensor probe. Only report GPU success if nvidia-smi and the
   real PyTorch CUDA tensor probe both pass.

Provider/data work:
- Accept only authorized five-minute OHLCV BAR_START data with explicit source,
  provenance, calendar and instrument mapping metadata.
- Use import-csv/immutable snapshots for historical data.
- If a provider adapter is needed, implement only CompletedBarFeed or the
  existing read-only MarketDataProvider contract. Normalize into validated
  completed events with available_at, received_at and a stable event ID.
- Retry only FeedUnavailableError with bounded backoff. Stop on auth,
  entitlement or schema failures.
- Add tests for replay, provider identity, receipt/availability clocks,
  chronological/missing bars, stale events, retries, restart and no-network
  behavior before claiming the adapter works.

Policy work:
- Cash and rule are diagnostic policies only.
- If integrating a learned checkpoint, use
  PaperTradingEnv.checkpoint_compatibility() and DQNTrainer.load_checkpoint()
  to validate observation schema, actions/exposures, feature/reward/execution/
  cost/risk versions, environment hash, instrument, config and policy ID.
- Inject only policy.act and stable metadata into ContinuousPaperRunner. Never
  let the policy change risk limits or active registry state.

Verification and handoff:
1. Run targeted tests, then make test in every available project-local venv.
2. Run the finite paper-stream replay/resume smoke command and inspect the
   durable account/events; confirm paper_only and provenance labels.
3. Run make demo only with the current v3-compatible paths or a new directory.
4. Repeat a clean git archive test from the final commit. Docker is a gap if
   docker is unavailable.
5. Update docs only with actual results, including failed/blocked checks.
6. Before commit, inspect git status, git diff, git log and git diff --check.
7. Commit one logical verified step at a time with a conventional message. Push
   only the intended branch when explicitly authorized and verify the remote.

Stop and ask one targeted question if a missing user choice changes the data
license, provider entitlement, Windows dependency source, or whether the user
wants only offline replay versus a real read-only live feed. Never guess.
```

## Immediate Next Action

Verify that `origin/main` is at `e0ab421` and that the worktree is clean, then
start Windows environment verification. No real provider or GPU repair should
be attempted before that evidence exists.
