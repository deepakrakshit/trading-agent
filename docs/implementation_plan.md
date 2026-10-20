# Implementation plan and requirement checklist

The canonical technical specification is `self_improving_trading_agent_blueprint.md` (read in full before implementation). This document maps its required behavior to the first reproducible local implementation. The default mode is offline, cash-only, long-only paper trading; no real broker order route exists. Completed evidence and remaining limits are recorded below.

## Acceptance criteria

1. A clean project-local environment can install pinned dependencies and run one offline test command.
2. Synthetic candles are explicitly labelled `SYNTHETIC`; the provider boundary can later accept authorised historical or read-only live data without changing the paper broker.
3. Validation rejects malformed, duplicate, out-of-order, stale, non-causal, or impossible OHLCV records.
4. Features are causal and computed from completed bars only.
5. A Gymnasium-compatible environment exposes a small discrete target-exposure action space, next-bar fills, fee/slippage-aware equity, and log-equity rewards.
6. The deterministic risk supervisor enforces cash-only/long-only limits, stale-feed and loss locks, and a kill switch independently of the policy.
7. The paper broker and SQLite ledger durably record orders, fills, holdings, fees, equity, transitions, and restart-safe state.
8. A compact DQN-family policy really trains, saves, reloads, selects actions, and can be evaluated against cash, a rule baseline, and random actions on chronological splits.
9. Candidate promotion is deterministic, logged, and reversible; a failed candidate cannot replace the incumbent.
10. An offline end-to-end demo creates a ₹10,000 virtual account, runs paper fills, writes a report, and states the data provenance and limitations honestly.

## Requirement-to-component map

| Blueprint area | Implemented component | Evidence |
| --- | --- | --- |
| Data provider abstraction | `trading_agent.data` protocol, synthetic/CSV/Parquet providers, narrow SmartAPI historical/quote adapters | provider/validation/transport mocks; strict CSV completion and optional quote-symbol checks; no real feed claim |
| Data quality/freshness | `trading_agent.data.validation` | malformed/duplicate/stale/session tests |
| Causal features | `trading_agent.features` | no-look-ahead and split tests |
| Portfolio math/costs | `trading_agent.portfolio` | hand-calculated ledger and fee tests |
| Paper execution | `trading_agent.execution` | order/fill/idempotency/restart tests |
| Independent safety | `trading_agent.risk` | kill switch, stale, exposure/loss tests |
| RL environment | `trading_agent.paper_env.PaperTradingEnv`, canonical broker/ledger | Gymnasium checker, reward/action/causality/restart tests |
| Learning/checkpoints | `trading_agent.learning`, `trading_agent.compatibility` | deterministic training/reload tests and strict v2 semantic compatibility |
| Evaluation/baselines | `trading_agent.evaluation`, `trading_agent.workflow` | actual three-seed chronological runs, cash/rule/random baselines, session-count audit |
| Promotion/rollback | `trading_agent.registry` | gate/rejection/rollback tests |
| Durable records | SQLite repository plus JSON/Parquet run artifacts, `trading_agent.experience` | persistence/replay tests, SQLite integrity check, explicit provenance manifests and receipt hashes |
| User workflow | `trading_agent.cli`, `trading_agent.runner` and `Makefile` | actual doctor, generate, train, evaluate, demo, account, kill and resume commands |
| Reproducibility | `.python-version`, `pyproject.toml`, CPU/CUDA locks, `Dockerfile`, `.env.example` | empty-venv install/tests passed; Docker itself unavailable |

## Explicit scope boundaries

- Default market data is offline synthetic data for development. It is never described as live or historical.
- No website scraping, no real order endpoint, no order modification/cancellation endpoint, no shorting, leverage, margin, options, or futures.
- SmartAPI read-only historical/quote adapters are implemented and mock-tested; actual account/schema/terms/cost/rate-limit verification is still blocked. Zerodha/Upstox were researched, not silently installed or purchased. The offline system remains useful without credentials.
- Candidate training is separate from the active policy. A policy cannot change risk limits or approve itself.
- Performance numbers are research outputs with declared windows and costs, not profitability claims.

## Stage tracking

| Stage | Verified delivery status | Exit evidence |
| --- | --- | --- |
| A. Repository/environment audit | Complete | `docs/environment_report.md`, this checklist |
| B. Reproducible setup | CPU/CUDA-wheel venvs and fresh CPU rebuild verified; container not run | pinned install, 249 offline tests; Docker missing |
| C. Data/ledger/execution foundations | Offline implementation complete | validation, cost, execution, risk and persistence tests |
| D. RL environment/training | Actual CPU training/reload complete | two trained candidates and environment checker |
| E. Evaluation/self-improvement | Offline candidate cycle complete; both rejected | six fold/seed fits per experiment, later-cutoff replay learning and registry receipts |
| F. Offline end-to-end operation | Complete and restart-tested | ₹10,000 shadow account, durable fills/attribution, pinned manifests and v2 resume |
| G. GPU benchmark | CUDA blocked by kernel/driver; CPU measured | CUDA wheel installed, failed real availability probe, 500-step CPU benchmark |
| H. Real data/live paper | Not verified; live service not claimed complete | user-authorised data/session/calendar and operational shadow evidence required |

## Final acceptance boundary

The original ten offline acceptance criteria are exercised by the 249-test suite and actual CLI demo. The T1 RL hardening follow-up and subsequent timing/accounting/data-integrity fixes are recorded in `docs/t1_hardening.md` and `docs/test_report.md`. This is not completion of the blueprint's eventual live-market research/service phases. `docs/model_evaluation.md` records negative results, corrected independent session counts and uncertainty; `docs/test_report.md` records executed checks and genuine gaps. GPU system repair, authenticated provider access/terms and a Docker-capable environment require user-controlled setup. No paid resources, real financial actions, secrets or remote Git pushes were introduced.
