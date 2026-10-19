"""Chronological paper-policy evaluation and deliberately simple baselines.

Split bounds are half-open and never shuffled.  Model fitting receives only
train/validation slices; each test policy and baseline uses a fresh environment
from the same factory, leaving execution/cost configuration in one place.
An optional final holdout is excluded from all generated folds.

Risk ratios use session/daily returns, never an assumed number of intraday
bars.  Supply exchange-local datetime timestamps or explicit session IDs for
overnight/timezone calendars.  Ratios with insufficient data or zero variance
are ``None``, rather than fictitious infinite Sharpe scores.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
import math
import random
import statistics
from typing import Any, Protocol


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    net_return: float
    max_drawdown: float
    sessions: int = 0
    trades: int = 0
    daily_sharpe: float | None = None
    daily_sortino: float | None = None
    annualized_return: float | None = None
    calmar_ratio: float | None = None
    expected_shortfall: float | None = None
    turnover: float = 0.0
    total_fees: float = 0.0
    mean_exposure: float | None = None
    time_in_cash: float | None = None
    win_rate: float | None = None
    expectancy: float | None = None
    profit_factor: float | None = None
    cost_fraction_of_gross_profit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session_id(timestamp: Any) -> str:
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    if isinstance(timestamp, date):
        return timestamp.isoformat()
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp).date().isoformat()
        except ValueError as exc:
            raise ValueError("timestamps must be ISO datetimes/dates or use explicit session_ids") from exc
    raise ValueError("daily metrics need datetime timestamps or explicit session_ids")


def session_returns(equity_curve: Sequence[float], session_ids: Sequence[Any]) -> tuple[float, ...]:
    """Aggregate to each session's last equity, retaining the initial capital."""

    equity = _finite(equity_curve, "equity_curve")
    if not equity or len(equity) != len(session_ids) or any(value <= 0 for value in equity):
        raise ValueError("equity/session_ids must have equal nonzero length and positive equity")
    seen: set[Any] = set()
    labels = []
    closes = []
    counts = []
    for value, label in zip(equity, session_ids):
        if not labels or label != labels[-1]:
            if label in seen:
                raise ValueError("session_ids must be chronological and contiguous")
            seen.add(label)
            labels.append(label)
            closes.append(value)
            counts.append(1)
        else:
            closes[-1] = value
            counts[-1] += 1
    previous = equity[0]
    result = []
    # The initial capital snapshot is context, not an extra zero-return
    # session.  Count the first session only when it contains a transition.
    first = 1 if counts[0] == 1 else 0
    for close in closes[first:]:
        result.append(close / previous - 1.0)
        previous = close
    return tuple(result)


def compute_metrics(
    equity_curve: Sequence[float],
    *,
    timestamps: Sequence[Any] | None = None,
    session_ids: Sequence[Any] | None = None,
    total_fees: float = 0.0,
    turnover_notional: float = 0.0,
    exposures: Sequence[float] = (),
    trade_pnls: Sequence[float] = (),
    trade_count: int | None = None,
    gross_profit: float | None = None,
    sessions_per_year: int = 252,
    tail_confidence: float = 0.95,
) -> PerformanceMetrics:
    """Compute post-cost curve metrics with explicit daily sampling.

    ``turnover`` is one-way traded notional divided by initial equity.
    ``trade_count`` may identify nonzero fill events when a ledger's closed
    trade count is unavailable; win/expectancy metrics require actual supplied
    closed-trade P&Ls.  ``gross_profit`` is explicit, never inferred from net
    equity.  Exposure and cash-time are bar-weighted, not clock-time weighted.
    Without timestamps/session IDs, daily ratios and tail loss remain unknown.
    """

    equity = _finite(equity_curve, "equity_curve")
    if len(equity) < 2 or any(value <= 0 for value in equity):
        raise ValueError("at least two positive equity observations are required")
    if timestamps is not None and session_ids is not None:
        raise ValueError("supply timestamps or session_ids, not both")
    if timestamps is not None:
        if len(timestamps) != len(equity):
            raise ValueError("timestamps must align with equity")
        if any(first > second for first, second in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be chronological")
        session_ids = tuple(_session_id(timestamp) for timestamp in timestamps)
    daily = session_returns(equity, session_ids) if session_ids is not None else ()
    if not isinstance(sessions_per_year, int) or sessions_per_year <= 0 or not 0 < tail_confidence < 1:
        raise ValueError("sessions_per_year/tail_confidence are invalid")
    if not all(math.isfinite(value) and value >= 0 for value in (total_fees, turnover_notional)):
        raise ValueError("fees and turnover_notional must be finite and non-negative")
    exposure = _finite(exposures, "exposures")
    if exposure and (len(exposure) not in (len(equity), len(equity) - 1) or any(not 0 <= value <= 1 + 1e-8 for value in exposure)):
        raise ValueError("long-only exposures must align with equity and lie in [0, 1]")
    pnls = _finite(trade_pnls, "trade_pnls")
    trades = len(pnls) if trade_count is None else trade_count
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < len(pnls):
        raise ValueError("trade_count must be an integer covering supplied P&Ls")
    if gross_profit is not None and (not math.isfinite(gross_profit) or gross_profit < 0):
        raise ValueError("gross_profit must be finite and non-negative")

    peak = equity[0]
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, 1.0 - value / peak)
    net_return = equity[-1] / equity[0] - 1.0
    sharpe = sortino = annualized = calmar = tail_loss = None
    if daily:
        mean = statistics.fmean(daily)
        deviation = statistics.stdev(daily) if len(daily) >= 2 else 0.0
        if deviation > 0:
            sharpe = mean / deviation * math.sqrt(sessions_per_year)
        downside_deviation = math.sqrt(statistics.fmean(min(value, 0.0) ** 2 for value in daily))
        if len(daily) >= 2 and downside_deviation > 0:
            sortino = mean / downside_deviation * math.sqrt(sessions_per_year)
        try:
            annualized = math.expm1(math.log1p(net_return) * sessions_per_year / len(daily))
        except OverflowError:
            annualized = None
        if annualized is not None and drawdown > 0:
            calmar = annualized / drawdown
        count = max(1, math.ceil((1 - tail_confidence) * len(daily) - 1e-12))
        tail_loss = max(0.0, -statistics.fmean(sorted(daily)[:count]))
    profits = sum(value for value in pnls if value > 0)
    losses = -sum(value for value in pnls if value < 0)
    return PerformanceMetrics(
        net_return=net_return, max_drawdown=drawdown, sessions=len(daily), trades=trades,
        daily_sharpe=sharpe, daily_sortino=sortino, annualized_return=annualized,
        calmar_ratio=calmar, expected_shortfall=tail_loss,
        turnover=turnover_notional / equity[0], total_fees=float(total_fees),
        mean_exposure=statistics.fmean(exposure) if exposure else None,
        time_in_cash=sum(value <= 1e-12 for value in exposure) / len(exposure) if exposure else None,
        win_rate=sum(value > 0 for value in pnls) / len(pnls) if pnls else None,
        expectancy=statistics.fmean(pnls) if pnls else None,
        profit_factor=profits / losses if losses > 0 else None,
        cost_fraction_of_gross_profit=total_fees / gross_profit if gross_profit else None,
    )


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    @property
    def train(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def validation(self) -> slice:
        return slice(self.validation_start, self.validation_end)

    @property
    def test(self) -> slice:
        return slice(self.test_start, self.test_end)

    def validate(self, n_samples: int) -> None:
        if not (0 <= self.train_start < self.train_end <= self.validation_start < self.validation_end <= self.test_start < self.test_end <= n_samples):
            raise ValueError("walk-forward splits must contain ordered, non-overlapping half-open windows")


def chronological_splits(
    n_samples: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
    purge: int = 0,
    embargo: int = 0,
    final_holdout_size: int = 0,
) -> tuple[WalkForwardSplit, ...]:
    """Generate walk-forward folds, with purge+embargo on both boundaries.

    Choose gaps to cover the feature lookback/holding horizon of the supplied
    data pipeline.  This function cannot infer those horizons.  Test windows
    never overlap, and the final holdout is never used by any fold.
    """

    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (n_samples, train_size, validation_size, test_size)):
        raise ValueError("sample/window sizes must be positive integers")
    if any(not isinstance(value, int) or value < 0 for value in (purge, embargo, final_holdout_size)):
        raise ValueError("purge, embargo, and final_holdout_size must be non-negative integers")
    step = test_size if step_size is None else step_size
    if not isinstance(step, int) or step < test_size:
        raise ValueError("step_size must be at least test_size to keep tests disjoint")
    limit = n_samples - final_holdout_size
    gap = purge + embargo
    folds = []
    offset = 0
    while True:
        train_end = offset + train_size
        validation_start = train_end + gap
        validation_end = validation_start + validation_size
        test_start = validation_end + gap
        test_end = test_start + test_size
        if test_end > limit:
            break
        split = WalkForwardSplit(0 if expanding else offset, train_end, validation_start, validation_end, test_start, test_end)
        split.validate(n_samples)
        folds.append(split)
        offset += step
    if not folds:
        raise ValueError("not enough samples for one complete chronological fold")
    return tuple(folds)


class ActionPolicy(Protocol):
    def __call__(self, observation: Sequence[float]) -> int:
        ...


@dataclass(frozen=True, slots=True)
class CashBaseline:
    cash_action: int = 0

    def __call__(self, observation: Sequence[float]) -> int:
        return self.cash_action


@dataclass(frozen=True, slots=True)
class RuleBaseline:
    """Long on positive completed-bar return, otherwise cash (no price access)."""

    long_action: int = 4
    cash_action: int = 0
    feature_index: int = 3
    threshold: float = 0.0

    def __call__(self, observation: Sequence[float]) -> int:
        return self.long_action if float(observation[self.feature_index]) > self.threshold else self.cash_action


class RandomBaseline:
    def __init__(self, n_actions: int, *, seed: int = 0) -> None:
        if not isinstance(n_actions, int) or n_actions <= 0:
            raise ValueError("n_actions must be a positive integer")
        self.n_actions = n_actions
        self._rng = random.Random(seed)

    def __call__(self, observation: Sequence[float]) -> int:
        return self._rng.randrange(self.n_actions)


cash_baseline = CashBaseline
rule_baseline = RuleBaseline
random_baseline = RandomBaseline


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    equity_curve: tuple[float, ...]
    timestamps: tuple[Any, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    metrics: PerformanceMetrics
    terminated: bool
    truncated: bool


def evaluate_policy(
    env: Any,
    policy: ActionPolicy,
    *,
    seed: int = 0,
    session_key: Callable[[Any], Any] | None = None,
    max_steps: int = 1_000_000,
) -> EpisodeResult:
    """Evaluate a frozen callable policy; neither fit nor update is invoked."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    observation, info = env.reset(seed=seed)
    equities = [float(info["equity"])]
    timestamps = [info["timestamp"]]
    actions, rewards, exposures, trade_pnls = [], [], [], []
    total_fees = turnover = 0.0
    trades = 0
    terminated = truncated = False
    for index in range(max_steps):
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        equities.append(float(info["equity"]))
        timestamps.append(info["timestamp"])
        actions.append(int(action))
        rewards.append(float(reward))
        exposures.append(float(info["exposure"]))
        fees = float(info["realized_fees"])
        notional = float(info["turnover"])
        total_fees += fees
        turnover += notional
        trades += int(notional > 0)
        trade_pnls.extend(info.get("closed_trade_pnls", ()))
        if terminated or truncated:
            break
    else:
        truncated = True
    if session_key is not None:
        metrics = compute_metrics(equities, session_ids=[session_key(timestamp) for timestamp in timestamps], total_fees=total_fees, turnover_notional=turnover, exposures=exposures, trade_pnls=trade_pnls, trade_count=max(trades, len(trade_pnls)))
    else:
        metrics = compute_metrics(equities, timestamps=timestamps, total_fees=total_fees, turnover_notional=turnover, exposures=exposures, trade_pnls=trade_pnls, trade_count=max(trades, len(trade_pnls)))
    return EpisodeResult(tuple(equities), tuple(timestamps), tuple(actions), tuple(rewards), metrics, bool(terminated), bool(truncated))


def evaluate_baselines(
    env_factory: Callable[[], Any],
    *,
    seed: int = 0,
    session_key: Callable[[Any], Any] | None = None,
    rule_feature_index: int = 3,
) -> dict[str, EpisodeResult]:
    result = {}
    for name in ("cash", "rule", "random"):
        env = env_factory()
        try:
            exposures = tuple(env.target_exposures)
            if 0.0 not in exposures:
                raise ValueError("the cash baseline requires a zero-exposure action")
            cash_action = exposures.index(0.0)
            policies = {
                "cash": CashBaseline(cash_action),
                "rule": RuleBaseline(len(exposures) - 1, cash_action, rule_feature_index),
                "random": RandomBaseline(len(exposures), seed=seed),
            }
            result[name] = evaluate_policy(env, policies[name], seed=seed, session_key=session_key)
        finally:
            env.close()
    return result


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    split: WalkForwardSplit
    seed: int
    candidate: EpisodeResult
    baselines: Mapping[str, EpisodeResult]


def walk_forward_evaluate(
    data: Sequence[Any],
    splits: Sequence[WalkForwardSplit],
    fit_policy: Callable[[Sequence[Any], Sequence[Any], int], ActionPolicy],
    env_factory: Callable[[Sequence[Any]], Any],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    session_key: Callable[[Any], Any] | None = None,
    rule_feature_index: int = 3,
) -> tuple[WalkForwardResult, ...]:
    """Fit independent candidates and score every fold/seed plus baselines.

    The first bar of each test slice supplies initial completed-bar context,
    so that slice produces ``len(test)-1`` transitions.  No test/final-holdout
    sample is supplied to ``fit_policy``.
    """

    if not splits or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide chronological splits and distinct evaluation seeds")
    previous_test_end = -1
    for split in splits:
        split.validate(len(data))
        if split.test_start < previous_test_end:
            raise ValueError("test windows must be chronological and disjoint")
        if split.test_end - split.test_start < 2:
            raise ValueError("test windows must contain context plus at least one fill bar")
        previous_test_end = split.test_end
    result = []
    for split in splits:
        for seed in seeds:
            policy = fit_policy(tuple(data[split.train]), tuple(data[split.validation]), seed)
            test_data = tuple(data[split.test])
            env = env_factory(test_data)
            try:
                candidate = evaluate_policy(env, policy, seed=seed, session_key=session_key)
            finally:
                env.close()
            baselines = evaluate_baselines(lambda: env_factory(test_data), seed=seed, session_key=session_key, rule_feature_index=rule_feature_index)
            result.append(WalkForwardResult(split, seed, candidate, baselines))
    return tuple(result)


def summarize_walk_forward(results: Sequence[WalkForwardResult]) -> dict[str, float | int]:
    """Publish all runs' median and dispersion instead of choosing a best seed."""

    if not results:
        raise ValueError("at least one evaluation result is required")
    returns = [row.candidate.metrics.net_return for row in results]
    return {
        "runs": len(results),
        "windows": len(set(row.split for row in results)),
        "seeds": len(set(row.seed for row in results)),
        "median_net_return": statistics.median(returns),
        "net_return_stdev": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "worst_net_return": min(returns),
        "worst_drawdown": max(row.candidate.metrics.max_drawdown for row in results),
    }


__all__ = [
    "ActionPolicy", "CashBaseline", "EpisodeResult", "PerformanceMetrics", "RandomBaseline",
    "RuleBaseline", "WalkForwardResult", "WalkForwardSplit", "cash_baseline", "chronological_splits",
    "compute_metrics", "evaluate_baselines", "evaluate_policy", "random_baseline", "rule_baseline",
    "session_returns", "summarize_walk_forward", "walk_forward_evaluate",
]
