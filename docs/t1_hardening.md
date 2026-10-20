# RL pipeline hardening (T1 implementation)

This document records the implementation follow-up to the T1 audit. It is an
engineering correctness record, not a profitability claim. All examples and
the reproducibility run remain explicitly synthetic/offline.

## Fixed findings

1. **Episode termination and DQN targets** — `episode_length` is now a
   task-defined terminal horizon. A boundary transition includes the
   precommitted terminal liquidation (when enabled), returns
   `terminated=True, truncated=False`, and is masked from DQN bootstrapping.
   A collection-budget cutoff created by `DQNTrainer.train` remains a
   truncation and may bootstrap. Ordinary nonterminal transitions remain
   `(False, False)`.
2. **Experience provenance** — new paper accounts store an explicit
   `paper-experience-v1` manifest in `<account>.sqlite.spec.json`. Import
   validates full dataset identity, slice hash, instrument, environment hash,
   feature ordering/version, reward/execution/cost/risk versions, policy and
   account identity, source timestamps, contiguous episode rows, database
   cursor/history digest, and the new training cutoff. A later expanded
   dataset is allowed; an unverified source is not.
3. **Checkpoint compatibility** — DQN checkpoint schema v2 carries a strict
   semantic contract. Training resume, walk-forward reuse, final evaluation,
   paper replay, and the workflow's promotion preparation validate dimensions,
   ordered feature schema/version, exact exposure mapping, reward, execution,
   cost, risk, environment hash, config version, instrument, and policy ID.
   Tensor shape alone is insufficient.
4. **Walk-forward checkpoint integrity** — every saved fold has a receipt with
   SHA-256, fold/config/dataset/policy identity, dimensions, and step count.
   Modified, substituted, missing, or receipt-less checkpoints fail closed.
5. **Independent promotion evidence** — offline evaluation no longer asserts
   accounting/data/risk success. It records versioned, failed/unavailable
   safety receipts for accounting, data integrity, risk, shadow, and recovery.
   The registry requires all five versioned receipts even if legacy shadow or
   recovery gate switches are set false.
6. **Environment checkpoint semantics** — the paper cursor format is now v2.
   It validates types, bounds, episode options, done state, last-info index,
   RNG state, broker/ledger revision and hash, and the complete transition
   chain before resume. Earlier cursors are rejected rather than guessed.
7. **Environment contract** — `PaperTradingEnv` remains the only supported
   financial simulation for CLI training, evaluation, and replay. The old
   injected-execution `TradingEnv` remains import-compatible only and now
   emits a deprecation warning; it is documented as a compatibility fixture.

## Checkpoint and account migration policy

Existing v1 model/account artifacts are preserved unchanged. The current code
does **not** silently reinterpret them: DQN v1 checkpoints and paper cursor
v1 state are rejected with an explicit migration/retraining error. To use an
old experiment, start a new versioned experiment and retrain (or build an
independently reviewed migration tool that reconstructs the v2 compatibility
metadata and revalidates every transition). Existing holdout reports and
checksums remain historical receipts and are not rewritten.

New model checkpoints use:

```text
format: trading-agent-dqn
version: 2
metadata.checkpoint_compatibility.compatibility_version: paper-compat-v1
```

New paper specs use `paper-replay-v2` and include the explicit experience
manifest. The database cursor retains a v2 `format` marker under the existing
state key so a stale v1 account cannot be adopted accidentally.

## Changed implementation areas

- `src/trading_agent/paper_env.py` — terminal horizon, compatibility contract,
  provenance fields, and strict cursor/history validation.
- `src/trading_agent/learning.py` and new `src/trading_agent/compatibility.py`
  — checkpoint v2 and shared semantic validator.
- New `src/trading_agent/experience.py` — manifest, row, and history checks.
- `src/trading_agent/workflow.py` — validated experience import, source-data
  verification, fold checkpoint receipts, strict reloads, and fail-closed
  promotion evidence.
- `src/trading_agent/runner.py` — v2 paper specs, manifest creation, and
  action/checkpoint validation on replay.
- `src/trading_agent/registry.py` — versioned evidence receipts as a hard
  promotion boundary.
- `src/trading_agent/env.py` — explicit deprecation boundary for the legacy
  injected-execution environment.

## Regression coverage

Targeted tests cover open positions at artificial horizons, liquidation cash,
fees and realized/unrealized P&L, terminal-vs-truncated flags, DQN masking,
semantic checkpoint mismatch, provenance tampering, source cutoff, cursor
corruption, missing/substituted/modified fold checkpoints, and hard promotion
receipt requirements. The complete suite is run with:

```bash
.venv/bin/python -m pytest -q
make test
```

The authoritative result is reported in `docs/test_report.md`; no CUDA claim
is implied by these checks.

## Continuation audit fixes (T2/T4)

The independent accounting/data audits found and fixed additional unblocked
correctness issues:

- An explicit replay clock cannot receive a next-bar close/reward before
  `bar_start + interval`, and accepted clocks cannot move backwards. Open-event
  market-hours checks still use the actual eligible open while feed freshness
  uses the supplied observation clock.
- A later global mark now requires a fresh mark for every held symbol. Ledger
  restore validates fill/order idempotency sets, reconstructs the fill ledger,
  and rejects inconsistent cash, P&L, fees, positions or high-water marks.
- Direct canonical environments reject missing bars inside a local exchange
  session. CSV imports parse `complete` strictly, reject forming candles at an
  injected capture time, and reject unknown columns. Canonical CSV snapshots
  round-trip through the CSV provider.
- Experiments, paper accounts and experience manifests carry a pinned snapshot
  manifest digest. Historical reload validation uses the snapshot's immutable
  capture boundary rather than the current wall clock. Consumed evaluation
  receipts require a separate SHA-256 sidecar.
- Read-only quote responses are rejected when an optional returned
  `tradingSymbol` disagrees with the configured symbol/token mapping.

The current paper execution contract is `causal-open-v3`; old v1/v2 generated
artifacts remain preserved and are not silently migrated. The post-fix suite is
249 passing tests, and the exact CPU/demo evidence is in
`docs/model_evaluation.md` and `docs/test_report.md`.
