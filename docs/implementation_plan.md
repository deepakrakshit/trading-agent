# Implementation plan and requirement checklist

The canonical technical specification is `self_improving_trading_agent_blueprint.md` (read in full before implementation). This document maps its required behavior to the first reproducible local implementation. The default mode is offline, cash-only, long-only paper trading; no real broker order route will exist.

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

| Blueprint area | Planned implementation | Evidence |
| --- | --- | --- |
| Data provider abstraction | `trading_agent.data` provider protocol, synthetic fixture, CSV/Parquet loader, read-only provider stub boundary | provider and validation tests |
| Data quality/freshness | `trading_agent.data.validation` | malformed/duplicate/stale/session tests |
| Causal features | `trading_agent.features` | no-look-ahead and split tests |
| Portfolio math/costs | `trading_agent.portfolio` | hand-calculated ledger and fee tests |
| Paper execution | `trading_agent.execution` | order/fill/idempotency/restart tests |
| Independent safety | `trading_agent.risk` | kill switch, stale, exposure/loss tests |
| RL environment | `trading_agent.env` | Gymnasium checker, reward/action tests |
| Learning/checkpoints | `trading_agent.learning` | deterministic training/reload tests |
| Evaluation/baselines | `trading_agent.evaluation` | chronological walk-forward and metrics reports |
| Promotion/rollback | `trading_agent.registry` | gate/rejection/rollback tests |
| Durable records | SQLite repository plus JSON/Parquet run artifacts | persistence/replay tests |
| User workflow | `trading_agent.cli` and `Makefile` | doctor, generate, train, evaluate, demo, account commands |
| Reproducibility | `.python-version`, `pyproject.toml`, `requirements.lock`, `Dockerfile`, `.env.example` | clean install/test instructions |

## Explicit scope boundaries

- Default market data is offline synthetic data for development. It is never described as live or historical.
- No website scraping, no real order endpoint, no order modification/cancellation endpoint, no shorting, leverage, margin, options, or futures.
- Angel One SmartAPI/Zerodha/Upstox integration remains a read-only future adapter until official access, terms, authentication and rate limits are verified and the user supplies credentials locally. The code must remain useful without them.
- Candidate training is separate from the active policy. A policy cannot change risk limits or approve itself.
- Performance numbers are research outputs with declared windows and costs, not profitability claims.

## Stage tracking

| Stage | Status at audit | Exit evidence |
| --- | --- | --- |
| A. Repository/environment audit | Complete | `docs/environment_report.md`, this checklist |
| B. Reproducible setup | In progress | install, lock, Docker build, `make test` |
| C. Data/ledger/execution foundations | Pending | focused unit tests |
| D. RL environment/training | Pending | checkpoint reload and environment checker |
| E. Evaluation/self-improvement | Pending | walk-forward report and promotion decision |
| F. Offline end-to-end operation | Pending | demo artifacts and restart test |
| G. GPU benchmark | Blocked by current NVIDIA driver; CPU fallback proceeds | doctor output and benchmark report |
