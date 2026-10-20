# Actual model training and paper evaluation

**Status:** the offline engineering pipeline works; **no profitable strategy, GPU training, genuine historical-data result or live NSE connection is established**. Both trained candidates were rejected. The active-policy fallback is `cash-v1`, not either unapproved model.

All data below is **SYNTHETIC**, generated locally with seed 2026. Dates label artificial weekday sessions; they are not evidence of real exchange trading or an authoritative holiday calendar. Reports were produced on **19 October 2026**.

## Initial candidate: `initial-v1`

### Frozen configuration and genuine training

- Data: `data/raw/synthetic.parquet`, 1,800 five-minute candles / 24 weekday sessions / one explicit fictitious instrument `SYNTHETIC_EQ`.
- Algorithm: **Double-DQN**, 9-element causal/account observation, 5 requested exposures **0%, 6.25%, 12.5%, 18.75%, 25%**. The independent supervisor can reject any request.
- Network: MLP **9 → 32 → 32 → 5**, 1,541 trainable parameters. Batch 32, replay capacity 5,000, gamma 0.99, Adam learning rate 0.0003, update frequency 4, fixed seed 0.
- Training episodes: randomized historical start/date, 149-transition windows, starting cash log-uniform over ₹5,000–₹10,000. The demonstration account itself always starts at the configured ₹10,000.
- Actual execution: **CPU**, PyTorch **2.10.0+cu128** in `.venv-cuda`; CUDA unavailable. **3,000 new interactions, 735 optimizer updates, 20.6236 seconds** measured training-loop time. Initial construction, evaluation and checkpoint I/O are not all included in that timing.
- Loss: mean Huber loss fell from approximately 0.003184 in the first 500-step chunk to 0.000005639 in the final chunk. **The account still lost money**, illustrating why low training loss is not a profitability test.
- Reward: `log(post_cost_equity_next / equity_current)`. No extra cost/drawdown bonus or double-counted slippage penalty. Invalid/nonpositive equity is terminal with a finite safety sentinel reward (-1), not `log(0)`.
- Training and paper replay use the same Decimal ledger, cost model, independent risk supervisor, target planner and next-open paper broker.

### Chronological protocol

Half-open source-row bounds were frozen before evaluation:

| Block | Rows | Synthetic bar-start range (Asia/Kolkata) |
| --- | --- | --- |
| Train | `[0, 900)` | 5 January 09:15 – 20 January 15:25 |
| Validation | `[906, 1131)` | 21 January 09:45 – 26 January 09:40 |
| Final holdout | `[1350, 1800)` | 29 January 09:15 – 5 February 15:25 |

There are five purge bars plus one embargo bar at specified fold boundaries. Extra unused dates remain between primary validation and final holdout. No fitted global scaler is used; features are causal ratios/log returns and trailing volume statistics. The seed/configuration were fixed, not selected by final-test performance.

Walk-forward folds: train `[0,600)`, validate `[606,756)`, test `[762,987)`; then train `[0,825)`, validate `[831,981)`, test `[987,1212)`. Each fold used seeds **0, 1, 2**, with 1,000 training interactions per fit. The final holdout was excluded from every fold and assessed once for the frozen primary model. Later CLI reads load the stored receipt rather than tuning/rerunning that holdout.

### Final holdout comparison: six synthetic sessions

Each strategy uses identical ₹10,000 starting funds, costs, maximum exposure and execution semantics. “Trades” here means **nonzero simulated fill events**, not independent closed round trips.

| Strategy | Net return | Final equity | Max drawdown | Explicit fees | Fill events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cash / no trade | 0.0000% | ₹10,000.00 | 0.0000% | ₹0.00 | 0 |
| Frozen Double-DQN | **-4.9792%** | **₹9,502.08** | **4.9792%** | **₹188.32** | **311** |
| Random exposure, seed 0 | -5.4446% | ₹9,455.54 | 5.4988% | ₹218.10 | 351 |
| Rule: positive completed-bar return → maximum target, otherwise cash | -7.4479% | ₹9,255.21 | 7.4662% | ₹274.39 | 230 |

The DQN lost less than these particular rule/random baselines, but **cash was superior**. This is not evidence of a positive market edge.

### Durable shadow account and loss attribution

`runs/demo/account.sqlite` contains **449 transitions**, **311 simulated fills**, and **29 inventory-flat round trips**. Replay stopped at 50 transitions, closed/reopened the repository/policy/environment, then resumed; later a separate CLI process also resumed the completed account without new fills.

Final state: cash/equity ₹9,502.08; holdings zero; realized P&L **-₹497.92**; unrealized P&L zero; no failed liquidation. Recorded costs reconcile as:

```text
Gross reference-price P&L        -₹86.01
Adverse execution-price effect -₹223.59
Explicit fees/taxes            -₹188.32
Net portfolio P&L              -₹497.92
```

The observed execution effect includes spread/slippage and adverse tick rounding. It is already in fill prices, not an additional cash debit. Trade reports include fee components, signal-to-fill gaps, MFE/MAE holding-window bounds and policy versions; market causes are explicitly `UNKNOWN`, not fabricated news explanations.

Separate `runs/active/account.sqlite`: **₹10,000.00 cash, zero fills**, no accepted model, and the operator kill switch was engaged as an additional safe final state. `runs/demo/rule_account.sqlite` is a separate diagnostic baseline account, not the active strategy.

## Subsequent candidate: `paper-replay-v2`

This additional controlled cycle was actually run, not just implemented:

1. Extend the same deterministic synthetic generator to 48 weekday sessions / 3,600 candles, through 11 March 2026.
2. Freeze a new earlier training block `[0,1800)` through 5 February. Only now are the first account's January/February observations eligible training history; attempting to import them into the earlier cutoff is rejected by a regression test.
3. Import **449 durable paper transitions** with matching source/feature/action/risk/cost configuration, then perform **1,000 new environment interactions**. The counter records **1,449 total observed transitions / 347 optimizer updates**; the measured **6.9430 seconds** is new-interaction training time, excluding the initial replay-import optimization/construction.
4. Evaluate the frozen candidate on new final holdout `[2700,3600)`: **24 February – 11 March**, 900 candles / **12 synthetic sessions**. Validation was `[1806,2256)`. Two expanding walk-forward folds, each with seeds 0/1/2, used earlier dates only.
5. Keep the active policy unchanged when the candidate fails its predeclared gates.

| Strategy, later holdout | Net return | Max drawdown | Explicit fees | Fill events |
| --- | ---: | ---: | ---: | ---: |
| Cash | 0.0000% | 0.0000% | ₹0.00 | 0 |
| Replay-trained Double-DQN | **-6.1238%** | **6.6090%** | **₹324.38** | **608** |
| Random exposure, seed 0 | -8.2292% | 8.3941% | ₹434.22 | 688 |
| Rule baseline | -9.0692% | 10.0774% | ₹452.97 | 374 |

The two candidate holdouts have different dates/lengths and training budgets; comparing -4.98% with -6.12% is **not** a controlled claim about improvement/deterioration. The second run proves that recorded experience can participate in a new candidate cycle without moving the old test into that same old experiment's training set. The walk-forward family fits train from historical environments afresh; they do not isolate the causal benefit of replay warm-starting. A fair warm-start ablation remains future research.

## Multi-seed dispersion and session-count audit

For each seed, the two fresh-account walk-forward window returns are compounded. This is not one uninterrupted portfolio curve; maximum drawdown in the summary is worst-window drawdown. Seed dispersion is algorithmic variability, **not** a statistical confidence interval and not extra independent market sessions.

| Experiment | Seed 0 compounded return | Seed 1 | Seed 2 | Median | Between-seed sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `initial-v1` | -0.8504% | -1.6431% | -4.1319% | **-1.6431%** | **1.7123 percentage points** |
| `paper-replay-v2` | -5.0130% | -5.0909% | -10.5854% | **-5.0909%** | **3.1950 percentage points** |

An audit found that disjoint intraday windows can share a boundary date. The first stored aggregate counted 8 window-sessions but only **7 unique dates**; the second counted 14 but only **13 unique dates**. The implementation was fixed and regression-tested to deduplicate sessions. The original financial scores and rejection receipts were deliberately not rewritten or retrained. Separate `evaluation_session_audit.json` artifacts record corrected dates/counts and all seed returns. Both corrected counts remain below the predeclared 20-session gate. The six-/twelve-session primary holdout counts above were not affected.

## Promotion decisions and confidence limits

Both registry decisions were **REJECTED**; active policy ID is `null` and the runner therefore uses versioned cash fallback. Reasons included negative walk-forward return, cash-baseline deterioration, too few sessions/positive windows, insufficient regime diversity, and no qualifying shadow/recovery evidence supplied to that decision.

The literal receipt codes `shadow_failed` / `recovery_failed` mean those independently qualifying receipts were **not supplied** at registry consideration. They do not mean pytest or the later offline restart demo failed. No live paper-shadow duration/performance evidence was fabricated, and the earlier immutable decision was not edited after the demo.

Important limits:

- Two related synthetic experiments, one fictitious instrument, and very few independent dates cannot support profitability, broad-market generalization, a reliable Sharpe ratio or a narrow confidence interval.
- Daily Sharpe/Sortino/annualized values remain available in machine reports but are unstable here. Extrapolating six days into an annual forecast would be misleading.
- Artificial regimes are not real market regimes. These evaluation windows did not satisfy the predefined multi-regime gate.
- Fees use an illustrative intraday schedule; order-book queues, true available liquidity, live latency, corporate actions, delivery/DP accounting and authoritative holidays are not modelled fully.
- The 2% daily/10% drawdown locks are order safeguards, not guarantees that gaps or retained positions cannot exceed thresholds. A failed exit is reported, never silently filled.
- No final-test-based hyperparameter search or best-seed selection was performed. Future research must use a new later holdout or a newly declared protocol; don't repeatedly optimize against these scores.
- PPO was not added merely to compare algorithms on fake data. A meaningful DQN/PPO comparison on permitted real history remains open.

## Actual artifact paths

| Artifact | Workspace-relative path |
| --- | --- |
| Initial trained checkpoint | `models/initial-v1/checkpoints/step-00003000.pt` |
| Initial training manifest and provenance | `models/initial-v1/experiment.json` |
| Initial evaluation and audit | `models/initial-v1/evaluation.json`, `models/initial-v1/evaluation_session_audit.json` |
| Replay-trained checkpoint | `models/paper-replay-v2/checkpoints/step-00001449.pt` |
| Replay experiment/evaluation | `models/paper-replay-v2/experiment.json`, `models/paper-replay-v2/evaluation.json` |
| Registry decisions | `models/registry.sqlite` |
| Paper account / detailed attribution | `runs/demo/account.sqlite`, `runs/demo/account.report.json` |
| End-to-end receipt | `runs/demo/demo.json` |
| CPU benchmark / failed GPU probe | `runs/benchmark.json`, `runs/gpu_verification.json` |

The initial checkpoint SHA-256 is `5874174842bf554031eae42b06680d324df92e7a53db7677ee7f22067ca6ed53`; its source commit is `f8d88e38d64d56e92b79e251f0b562a506efefcc`. The replay checkpoint SHA-256 is `047bf5f2d2ec76547747691b90caecb216b310c4df9095629bc56989eef77e75`; its training source commit is `b2448765bbc4338a8ba2954ae3d33eb98fb9cb03`. Manifests also store complete configuration, seed, dependency versions, dataset hash and feature/action/reward schema. Convenience `candidate.pt` copies are not the immutable resume source of truth.

The later T1 hardening work preserves these historical v1 artifacts and their
holdout receipts. The current loader requires checkpoint schema v2 semantic
compatibility metadata and rejects the old files rather than silently
reinterpreting them. Reuse requires a new versioned training cycle or an
independently reviewed migration; the historical scores above are not
rewritten.

## Continuation CPU audit: `cpu-audit-v1`

This is a fresh, deliberately small verification run from committed source
revision `2835cd2a04f17d7a929e35c9572f2d52dfbcc9bd`. It does not replace either
historical candidate.

- Data: the same explicitly **SYNTHETIC** `data/raw/synthetic.parquet`,
  dataset SHA-256 `1e3953f6ee637b8b92e15a60d937754ca845019b16bad4ff9d24731ea5850e11`,
  snapshot-manifest SHA-256
  `997ec1a2eaaaf481077a4fcd76410b2d46ebffdd278db5eb1fe7977f4fa0c58f`.
- Configuration: seed 0, 9-observation/5-action MLP, CPU PyTorch
  `2.10.0+cpu`, explicit `--device cpu`, 64 environment interactions, one
  optimizer update, and **0.5981 seconds** measured training time.
- Checkpoint: `models/cpu-audit-v1/checkpoints/step-00000064.pt`, SHA-256
  `7ef33a71442df97b7fdf344bba2a01b3a0aa9f8c80f757adc626e56a9c96a62e`.
- Reload evidence: checkpoint reload succeeded and produced action `4`, which
  is valid in the exact target action space. `peak_vram_bytes` was `null`.
- Frozen final holdout: **-2.2408%**, final equity approximately ₹9,775.92,
  2.4074% maximum drawdown, ₹16.33 fees and 21 fill events across six
  synthetic sessions.
- Walk-forward: median compounded seed return **-1.4465%**, 7 unique session
  dates, between-seed standard deviation 1.8143 percentage points. The
  candidate was rejected; no active policy changed.

Exact commands:

```bash
.venv/bin/python -m trading_agent.cli train --data data/raw/synthetic.parquet --output models/cpu-audit-v1 --steps 64 --device cpu
.venv/bin/python -m trading_agent.cli evaluate --experiment models/cpu-audit-v1 --device cpu
```

## Preserved pre-v4 v2 offline demo receipt

The recorded local v2 audit used
an explicit 256-step override: **3.5950 seconds**, checkpoint
`models/initial-v2/checkpoints/step-00000256.pt` with SHA-256
`af252036a16f07d0c5a48b74954f733ebe71f74eb6493e94a3902899a76d5b31`, final
shadow equity ₹9,698.67, 183 fills, ₹67.88 fees, and a rejected candidate.
`make demo` subsequently resumed this durable account and verified the pinned
evaluation receipt without adding fills. This remains synthetic engineering
evidence, not profitability evidence.

The execution contract is now `causal-open-v4` for the completed-bar runner.
The default CLI demo therefore uses new `models/initial-v3/` and
`runs/demo-v3/` paths; the pre-v4 v2 checkpoint/account is preserved and
intentionally rejected by the current compatibility loader rather than
silently migrated. Run `make demo` to create the v3-compatible receipt.

## Current execution-v4 v3 offline demo receipt

`make demo` was executed after the execution-v4 transition using the new
`models/initial-v3/` and `runs/demo-v3/` paths. The synthetic candidate ended
with shadow equity **₹9,582.25**, **260 fills**, **₹149.70 fees**, and a **rejected**
promotion decision. The diagnostic rule account ended at ₹9,255.21 with 230
fills and ₹274.39 fees. Restart verification passed. This is engineering and
accounting evidence only; it is not historical/live performance evidence.
