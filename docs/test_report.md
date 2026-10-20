# Test, build and execution report

## Final test state

The combined suite currently contains **259 tests: 259 passed, 0 failed, 0 skipped** in the project-local `.venv-repro` environment. The run emitted **65 warnings**: intentional deprecation notices for the legacy injected-execution compatibility environment and two advisory Gymnasium warnings for that legacy environment's unbounded observation space. The canonical CLI environment has finite float32 bounds and passes the checker. Warnings were not suppressed.

`tests/conftest.py` blocks socket connections automatically. The single offline test command is:

```bash
make test
```

It runs pytest, source/test compilation and `pip check`; it does not download data, contact a broker, or require credentials/GPU after installation.

## Executed environment checks

| Check | Actual result |
| --- | --- |
| `make test` in `.venv` | **259 passed**; compilation and dependency check passed |
| `make test PYTHON=.venv-cuda/bin/python` | **259 passed** with the CUDA-enabled wheel; not GPU execution evidence |
| `make test PYTHON=.venv-repro/bin/python` after the continuous-runner follow-up | **259 passed**, 0 skipped; compilation and dependency check passed |
| Canonical `Gymnasium check_env` | Passed in the test suite |
| `git diff --check` | Passed for working code steps |
| `.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda` | **Exit 2:** no usable CUDA device/driver |
| `make demo` | **Passed** using new execution-v4-compatible `models/initial-v3/` and `runs/demo-v3/`; restart verification passed and the synthetic candidate was rejected |
| `.venv-cuda/bin/python -m trading_agent.cli train --device cuda --output models/cuda-verification-only --steps 16` | **Exit 1:** refused CPU fallback for explicit CUDA request; no training run created |
| Two `apt-get --simulate install` attempts | Failed dependency resolution; no system modification |
| `docker --version` | Command not found; Docker build/run **not executed** |
| `git archive HEAD` at `c189573` into a clean temporary checkout + `make test PYTHON=/home/deepakrakshit/trading_agent/.venv-repro/bin/python` | **259 passed**; compilation and `pip check` passed; no ignored data/models/runs were present in the source archive |
| Dedicated linter/typechecker | Not configured/run; compilation is not claimed as static typing or lint coverage |

## Behaviors covered

- UTC/IST parsing, complete candles, invalid/duplicate/out-of-order/future/stale prices, missing intraday data, sessions/holidays/overrides.
- Immutable CSV/Parquet snapshots, metadata allowlist, tamper and manifest checks.
- Mocked read-only HTTP: route/origin allowlist, redirect refusal, 429/backoff, rate caps, disconnect/retry bounds, auth expiry, stale feed and stale last-trade timestamps.
- Causal rolling features, symbol isolation, next-open fills, future-volume/high/low/close invariance, completed-bar availability.
- Whole-share sizing, exact cash/basis/realized/unrealized P&L, equity/high-water mark, correct fee sides, brokerage cap/SEBI/IPFT/GST and slippage once.
- Independent cash/exposure/order/frequency/session/liquidity/risk limits, stale lockout, daily-loss/drawdown latches, kill switch and unfilled liquidation reporting.
- Gym observations/action bounds/reward/reset/termination/truncation, actual DQN optimization, bounded replay, deterministic CPU seeds, checkpoint save/load and RNG restoration.
- SQLite event/order/fill idempotency, transactional failure rollback, conflicting duplicates, stale concurrent writer detection, atomic cursor/transition/ledger recovery.
- Predeclared promotion acceptance/rejection, immutable evidence/artifact checks and rollback to an earlier accepted policy in isolated tests.
- No automatic unapproved candidate usage: active fallback is cash; explicit shadow is required.
- Full replay vs split-and-resumed replay, CLI parsing/doctor/data generation, loss attribution reconciliation and paper-experience training with chronological cutoff checks.
- Artificial episode-horizon liquidation is terminal and DQN-masked; collection-budget truncation remains bootstrap-capable.
- Explicit paper-experience manifests and row/database provenance checks reject incompatible datasets, instruments, feature semantics, execution versions, and tampering.
- DQN checkpoint v2 semantic compatibility, walk-forward SHA-256 receipts, strict cursor semantics, and hard versioned promotion receipts.
- Completed-bar clock monotonicity, fresh marks for every held symbol, reconciled ledger restore, strict CSV completion, and pinned snapshot-manifest reloads.
- Provider-independent completed-bar event identity, completion/receipt clocks, replay/range feed adapters, bounded retries without synthetic substitution, stale health records, cursor/broker rollback, restart continuation, graceful stop and persistent kill-switch behavior.

Passing tests do not establish a profitable strategy, an exchange-grade fill model, live-feed reliability or statistical sufficiency.

## Actual executed end-to-end commands

These commands were executed in the project root, not merely suggested:

```bash
# Project-local installation and meaningful test command
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy==2.4.2 gymnasium==1.2.3 pytest==8.4.2 pandas==3.0.0 pyarrow==23.0.1
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
make test

# Optional CUDA wheel, without modifying the host driver
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements-cuda.lock
.venv-cuda/bin/python -m pip install --no-deps --no-build-isolation -e .
make test PYTHON=.venv-cuda/bin/python

# Full actual synthetic demo, candidate training, frozen evaluation and restarted paper account
.venv/bin/python -m trading_agent.cli demo --device auto
.venv/bin/python -m trading_agent.cli demo --experiment models/initial-v3 --output runs/demo-v3 --steps 256 --device cpu
.venv-cuda/bin/python -m trading_agent.cli benchmark --steps 500 --output runs/benchmark.json
.venv/bin/python -m trading_agent.cli paper --db runs/demo-v3/account.sqlite --resume
.venv/bin/python -m trading_agent.cli account --db runs/demo-v3/account.sqlite

# Separate safe active-policy account and durable operator kill latch
.venv/bin/python -m trading_agent.cli paper --experiment models/initial-v1 --db runs/active/account.sqlite --strategy active
.venv-cuda/bin/python -m trading_agent.cli kill --db runs/active/account.sqlite

# Real follow-on learning from 449 persisted paper transitions
.venv-cuda/bin/python -m trading_agent.cli generate --output data/raw/synthetic-expanded.parquet --sessions 48
.venv-cuda/bin/python -m trading_agent.cli train --data data/raw/synthetic-expanded.parquet --output models/paper-replay-v2 --steps 1000 --experience-db runs/demo/account.sqlite --device auto
.venv-cuda/bin/python -m trading_agent.cli evaluate --experiment models/paper-replay-v2 --device auto

# Fresh committed-source CPU audit candidate
.venv/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/cpu-audit-v1 --steps 64 --device cpu
.venv/bin/python -m trading_agent.cli evaluate --experiment models/cpu-audit-v1 --device cpu

# Provider-independent completed-bar paper replay and durable resume
.venv-repro/bin/python -m trading_agent.cli paper-stream --data data/raw/synthetic.parquet --db /tmp/paper-stream.sqlite --strategy rule --max-bars 5
.venv-repro/bin/python -m trading_agent.cli paper-stream --data data/raw/synthetic.parquet --db /tmp/paper-stream.sqlite --strategy rule --resume --max-bars 5

# Empty-environment reproducibility
python3 -m venv .venv-repro
.venv-repro/bin/python -m pip install --cache-dir .cache/pip --timeout 30 --retries 2 -r requirements.lock
.venv-repro/bin/python -m pip install --no-deps --no-build-isolation -e .
make test PYTHON=.venv-repro/bin/python
```

`auto` selected **CPU** because the CUDA probe failed. No test output, GPU run, live connection or profitable result was fabricated. Setup/dependency installation needed public package-network access; subsequent tests were offline.

The continuation audit also verified a source-only `git archive` checkout using
the already-installed project-local `.venv-repro`; it did not claim a Docker
build or a fresh dependency download in that temporary checkout.

## Audit fixes made before handoff

- Integrated the independent branches with the canonical ledger/broker as the CLI's sole accounting path.
- Removed next-candle-volume leakage from open fills; added narrow open-event input and post-cost risk checks.
- Made broker execution + transition + cursor persistence atomic and idempotent, including failure/retry scenarios.
- Added explicit completed-candle availability and prevented an overlapping candle from filling too early.
- Included brokerage caps, SEBI/IPFT components and applicable GST; kept slippage separate and charged once.
- Rejected stale LTP even with a fresh quote heartbeat.
- Kept immutable per-step checkpoints so a crash cannot destroy the prior manifest/checkpoint pair.
- Removed exit-candle future high/low from holding-window MFE/MAE.
- Corrected walk-forward independent-session counting. Existing rejected receipts/scores remain archived, with a separate correction audit; see `docs/model_evaluation.md`.
- Enforced completed-bar availability and monotonic replay clocks, prevented stale omitted marks for held symbols, reconciled restored ledgers against fill records, and made CSV completion values strict.
- Pinned snapshot-manifest hashes into experiments/paper experience, reused the original snapshot capture boundary during reload, and hash-checked consumed evaluation receipts.

## T1 RL pipeline hardening

The seven follow-up findings from the RL/environment audit are implemented and
covered by targeted regressions. See `docs/t1_hardening.md` for the exact
termination policy, checkpoint v2 rejection/migration policy, provenance
manifest fields, changed files, and reproduction commands. Preserved v1
checkpoints and account artifacts are not rewritten; the current code rejects
them explicitly instead of silently guessing their semantics.

## Remaining work and precise blockers

1. **CUDA:** local Ubuntu administrator/user must repair the inconsistent signed kernel/driver package set and reboot, then run `doctor --require-cuda`. See `docs/gpu_repair.md`; no safe project-only package can load a missing kernel module.
2. **Real data:** user must choose authorised history or locally provide a read-only data session after verifying current provider eligibility, terms, costs and limits. No credentials belong in chat. Add authoritative calendar/instrument/corporate-action inputs.
3. **Live operations:** a provider-independent completed-bar replay runner, durable resume path and bounded retry/stale/kill controls are now implemented and offline-tested. Read-only provider adapters are still mocked; a real provider entitlement, continuously live shadow run, long enough live-shadow evidence and deployment monitoring must still be completed/validated before qualifying promotion.
4. **Research:** real chronological multi-regime validation, realistic fill/cost stress, DQN/PPO comparison, stronger uncertainty analysis and a dedicated drift detector remain open. No accepted learned model currently exists.
5. **Container:** Docker must be installed/run by an authorised user/environment to execute the documented build. The fresh-venv check does not replace this missing container check.

The repository is a working **offline research/paper system**, not a finished live-market service. Its value is auditable training, accounting, restart safety and rejection of unsupported candidates—not a claim that the policy has learned a profitable market edge.
