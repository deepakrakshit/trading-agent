"""Canonical Gymnasium paper trading with atomic, resumable transitions.

All money and execution flow through PortfolioLedger/CostModel/PaperBroker.
Actions select a target after a completed bar; share sizing is deferred until
the next eligible open price is observable.  No next-bar high/low/close/volume
is made available to the open-fill engine.  The close is observed afterwards.

A supplied SQLiteRepository represents one persistent account, not a training
episode pool.  A reset cannot overwrite it: use resume(), or a fresh repository
for an independent account.  Ephemeral environments support seeded random
training starts, dates, lengths and initial balances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import hashlib
import json
import math
from numbers import Integral
from typing import Any
from zoneinfo import ZoneInfo

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .data import MarketBar, validate_bars
from .data.bars import decimal_value, utc_timestamp
from .execution import ExecutionReceipt, OrderStatus, PaperBroker, PaperOpenEvent
from .features import CausalFeatureBuilder
from .learning import Transition
from .portfolio import CostModel, PortfolioLedger
from .portfolio.costs import money
from .risk import RiskContext, RiskLimits, RiskSupervisor
from .storage import PersistenceError, SQLiteRepository


def _encode(value: Any) -> str:
    def encode_special(item: Any) -> Any:
        if isinstance(item, (date, time)):
            return item.isoformat()
        if isinstance(item, timedelta):
            return item.total_seconds()
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"unsupported checkpoint value: {type(item).__name__}")

    return json.dumps(value, default=encode_special, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_encode(value).encode("utf-8")).hexdigest()


def _positive_cash(value: Any) -> Decimal:
    amount = decimal_value(value, "initial_cash")
    if amount <= 0 or money(amount) <= 0:
        raise ValueError("initial_cash must be positive to integer-paise precision")
    return money(amount)


class PaperTradingEnv(gym.Env[np.ndarray, int]):
    """Single-instrument, whole-share, cash-only canonical paper environment.

    ``episode_length`` counts transitions, not observations; a shorter data
    window ends by truncation, while data exhaustion/insolvency is termination.
    Randomisation is for fresh training episodes only.
    ``step(..., transition_id=...)`` permits
    an explicit uncertain-response retry; the same ID/action returns the saved
    result without another fill, reward, transition, or cursor advance.

    Configured hours are an independent RiskSupervisor gate.  Default hours
    are NSE-style 09:15–15:30 Asia/Kolkata on weekdays; holiday dates must be
    supplied in RiskLimits.  Passing all-hours RiskLimits is an explicit offline
    research configuration.  A session/terminal closeout is precommitted for a
    later eligible open and can be rejected by risk or insufficient liquidity.
    It never fabricates a same-close fill or bypasses a kill switch.
    """

    metadata = {"render_modes": []}
    CHECKPOINT_KEY = "paper_environment_v1"
    FORMAT_VERSION = 1
    FEATURE_VERSION = "paper-features-v1"
    feature_schema = (
        "open_to_close", "high_to_close", "low_to_close", "log_return",
        "relative_volume", "cash_fraction", "exposure", "drawdown", "episode_progress",
    )

    def __init__(
        self,
        bars: Sequence[MarketBar],
        initial_cash: Decimal | str | float = 10_000,
        *,
        cost_model: CostModel | None = None,
        risk_limits: RiskLimits | None = None,
        repository: SQLiteRepository | None = None,
        policy_version: str = "untrained-v1",
        config_version: str = "paper-env-v1",
        terminal_liquidation: bool = True,
        session_liquidation: bool = True,
        target_exposures: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        participation_limit: Decimal | str | float = 1,
        bar_interval: timedelta = timedelta(minutes=5),
        feature_lookback: int = 5,
        start_index: int = 0,
        episode_length: int | None = None,
        random_start: bool = False,
        random_start_date: bool = False,
        initial_cash_range: tuple[Decimal | str | float, Decimal | str | float] | None = None,
    ) -> None:
        super().__init__()
        self.bars = validate_bars(bars)
        if len(self.bars) < 2:
            raise ValueError("at least two completed canonical MarketBar records are required")
        if len({bar.symbol for bar in self.bars}) != 1:
            raise ValueError("PaperTradingEnv accepts exactly one instrument per account")
        self.symbol = self.bars[0].symbol
        if not isinstance(start_index, int) or isinstance(start_index, bool) or not 0 <= start_index < len(self.bars) - 1:
            raise ValueError("start_index must leave a following fill event")
        self._check_length(episode_length)
        if episode_length is not None and start_index + episode_length >= len(self.bars):
            raise ValueError("episode_length exceeds available bars")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version must be non-empty")
        if not isinstance(config_version, str) or not config_version.strip():
            raise ValueError("config_version must be non-empty")
        if bar_interval <= timedelta(0):
            raise ValueError("bar_interval must be positive")
        exposures = tuple(float(value) for value in target_exposures)
        if (
            not exposures or exposures[0] != 0
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in exposures)
            or exposures != tuple(sorted(set(exposures)))
        ):
            raise ValueError("target_exposures must be sorted unique weights in [0, 1], including cash")
        self.initial_cash = _positive_cash(initial_cash)
        self.cost_model = cost_model or CostModel()
        self.risk_limits = risk_limits or RiskLimits(
            market_open=time(9, 15), market_close=time(15, 30),
            permitted_symbols=frozenset({self.symbol}),
        )
        self.repository = repository
        self._policy_version = policy_version.strip()
        self.config_version = config_version.strip()
        self.terminal_liquidation = bool(terminal_liquidation)
        self.session_liquidation = bool(session_liquidation)
        self.target_exposures = exposures
        self.participation_limit = decimal_value(participation_limit, "participation_limit")
        if not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        self.bar_interval = bar_interval
        self.feature_lookback = feature_lookback
        features = CausalFeatureBuilder(feature_lookback).transform(self.bars)
        self._market_features = tuple(
            (
                float(bar.open / bar.close - 1), float(bar.high / bar.close - 1),
                float(bar.low / bar.close - 1), row.log_return or 0.0,
                row.volume_ratio - 1 if row.volume_ratio is not None else 0.0,
            )
            for bar, row in zip(self.bars, features)
        )
        self._configured_start = start_index
        self.episode_length = episode_length
        self.random_start = bool(random_start)
        self.random_start_date = bool(random_start_date)
        self.initial_cash_range = None
        if initial_cash_range is not None:
            low, high = map(_positive_cash, initial_cash_range)
            if low > high or not all(math.isfinite(float(value)) for value in (low, high)):
                raise ValueError("initial_cash_range must be ordered, finite positive amounts")
            self.initial_cash_range = (low, high)
        self.dataset_hash = _hash([bar.to_mapping() for bar in self.bars])
        self.config_hash = _hash(self._configuration())
        self.feature_dim = len(self.feature_schema) - 4
        self.observation_dim = len(self.feature_schema)
        self.n_actions = len(exposures)
        self.action_space = spaces.Discrete(self.n_actions)
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-bound, bound, shape=(len(self.feature_schema),), dtype=np.float32)
        self.start_index, self.end_index = start_index, len(self.bars) - 1
        self._index = start_index
        self._steps = 0
        self._done = False
        self._broker: PaperBroker | None = None
        self._episode_id = ""
        self._history_hash = ""
        self._episode_cash = self.initial_cash
        self._episode_options: dict[str, Any] = {}
        self._position_pnl_start = Decimal("0")
        self._last_info: dict[str, Any] = {}
        self._records: list[dict[str, Any]] = []
        if repository is not None:
            checkpoint = repository.load_state(self.CHECKPOINT_KEY)
            if checkpoint is not None:
                self._validate_checkpoint(checkpoint)

    @property
    def policy_version(self) -> str:
        """The immutable policy identity attached to every transition."""

        return self._policy_version

    @property
    def broker(self) -> PaperBroker:
        """The sole execution engine; reset/resume is required before access."""

        if self._broker is None:
            raise RuntimeError("call reset() or resume() before accessing the paper account")
        return self._broker

    @property
    def ledger(self) -> PortfolioLedger:
        return self.broker.ledger

    @property
    def risk_supervisor(self) -> RiskSupervisor:
        return self.broker.risk_supervisor

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def cash(self) -> float:
        return float(self.ledger.cash)

    @property
    def shares(self) -> int:
        return self.ledger.position_quantity(self.symbol)

    @property
    def equity(self) -> float:
        return float(self.ledger.snapshot().equity)

    @property
    def high_water_mark(self) -> float:
        return float(self.ledger.high_water_mark)

    @property
    def drawdown(self) -> float:
        return float(self.ledger.snapshot().drawdown)

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode, or explicitly resume an existing account.

        Options: start_index OR start_date (exchange-local ISO date),
        episode_length, initial_cash, random_start, random_start_date, resume.
        Random cash is log-uniform over initial_cash_range and rounded to paise.
        """

        settings = dict(options or {})
        known = {"start_index", "start_date", "episode_length", "initial_cash", "random_start", "random_start_date", "resume"}
        if set(settings) - known:
            raise ValueError(f"unknown reset options: {sorted(set(settings) - known)}")
        if settings.pop("resume", False):
            if seed is not None or settings:
                raise ValueError("resume cannot reseed or change an existing paper episode")
            return self.resume()
        if self.repository is not None:
            if self.repository.load_state(self.CHECKPOINT_KEY) is not None:
                raise RuntimeError("persistent paper account exists; use resume() instead of reset")
            if self.repository.load_state(PaperBroker.STATE_KEY) is not None:
                raise ValueError("repository has an unmanaged broker account without an environment cursor")
        super().reset(seed=seed)
        start, end, capital = self._select_episode(settings)
        self.start_index, self.end_index, self._index = start, end, start
        self._steps, self._done = 0, False
        self._episode_cash = capital
        self._episode_options = {"seed": seed, "start_index": start, "end_index": end, "initial_cash": str(capital)}
        self._episode_id = _hash({"dataset": self.dataset_hash, "config": self.config_hash, **self._episode_options})[:32]
        self._history_hash = _hash({"episode": self._episode_id})
        self._position_pnl_start = Decimal("0")
        self._records = []
        self._last_info = {}
        try:
            with self.repository.transaction() if self.repository else nullcontext():
                # Recheck under the write lock so two initialisers cannot adopt
                # or overwrite each other's account/cursor.
                if self.repository and self.repository.load_state(PaperBroker.STATE_KEY) is not None:
                    raise PersistenceError("paper account was created by another environment")
                self._broker = PaperBroker(
                    PortfolioLedger(capital), cost_model=self.cost_model,
                    risk_supervisor=RiskSupervisor(self.risk_limits),
                    repository=self.repository, participation_limit=self.participation_limit,
                )
                self.broker.observe_bar(self.bars[start])
                self._last_info = self._info()
                if self.repository:
                    self.repository.append_event(
                        f"ENV_START:{self._episode_id}", "ENV_START", self._episode_options,
                        aggregate_id=self._episode_id, occurred_at=self.bars[start].timestamp,
                        idempotency_key=self._episode_id,
                    )
                    self._save_cursor()
        except BaseException:
            self._broker = None
            raise
        return self._observation(), deepcopy(self._last_info)

    def resume(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore the exact cursor/ledger/risk/transition boundary, without RNG reset."""

        if self.repository is None:
            raise RuntimeError("resume requires a SQLiteRepository")
        with self.repository.transaction():
            state = self.repository.load_state(self.CHECKPOINT_KEY)
            if state is None:
                raise ValueError("no saved paper environment to resume")
            self._validate_checkpoint(state)
            if self.repository.load_state(PaperBroker.STATE_KEY) is None:
                raise PersistenceError("environment cursor has no corresponding broker checkpoint")
            broker = PaperBroker(
                PortfolioLedger(0), cost_model=self.cost_model,
                risk_supervisor=RiskSupervisor(self.risk_limits),
                repository=self.repository, participation_limit=self.participation_limit,
            )
            if broker.revision != state["broker_revision"]:
                raise PersistenceError("broker and environment cursor revisions do not agree")
            if _hash(broker.ledger.export_state()) != state["ledger_hash"]:
                raise PersistenceError("broker ledger does not match the environment checkpoint")
            self._broker = broker
            self._restore_cursor(state)
            try:
                self._validate_history()
            except BaseException:
                self._broker = None
                raise
        return self._observation(), deepcopy(self._last_info)

    def step(
        self, action: int, *, transition_id: str | None = None, as_of: datetime | None = None
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one canonical paper step and atomically log its full transition.

        ``as_of`` is an optional explicit replay/feed clock.  A late clock
        causes normal stale-data risk rejection; it does not alter bar data.
        Reusing an explicit transition ID with a different action is an error.
        """

        self.broker  # Require reset/resume before any retry or action.
        if isinstance(action, bool) or not isinstance(action, Integral) or not self.action_space.contains(action):
            raise ValueError(f"invalid discrete target-exposure action: {action!r}")
        action = int(action)
        if _hash(self._configuration()) != self.config_hash:
            raise ValueError("environment configuration changed during the episode")
        if transition_id is not None:
            if not isinstance(transition_id, str) or not transition_id.strip():
                raise ValueError("transition_id must be non-empty")
            existing = self._find_transition(transition_id)
            if existing is not None:
                if existing["action"] != action or (as_of is not None and existing["as_of"] != utc_timestamp(as_of).isoformat()):
                    raise PersistenceError("conflicting duplicate transition request")
                return self._result(existing)
        if self._done:
            raise RuntimeError("step() called after the episode ended; use a fresh episode/account")
        current, following = self.bars[self._index], self.bars[self._index + 1]
        clock = utc_timestamp(as_of) if as_of is not None else following.timestamp
        if clock < following.timestamp:
            raise ValueError("step clock cannot precede the eligible next-open event")
        identifier = transition_id or f"{self._episode_id}:{self._index}"
        before_cursor = self._cursor()
        observation = self._observation()
        previous = self.ledger.snapshot()
        before_shares = self.shares
        terminal_exit = self.terminal_liquidation and self._index + 1 == self.end_index
        session_exit = self._session_exit_due(current, following)
        requested = self.target_exposures[action]
        target = 0.0 if terminal_exit or session_exit else requested
        receipts: list[ExecutionReceipt] = []
        record: dict[str, Any]
        try:
            with self.broker.atomic():
                self._assert_current_cursor()
                decision = self.risk_supervisor.evaluate(
                    {self.symbol: target}, self._decision_context(current, as_of)
                )
                event = PaperOpenEvent(self.symbol, following.timestamp, following.open)
                if decision.approved:
                    order = self.broker.plan_target_order(
                        f"{identifier}:order", target, decision_at=current.timestamp, event=event
                    )
                    if order is not None:
                        submitted = self.broker.submit_order(order)
                        if submitted.status != OrderStatus.ACCEPTED:
                            receipts.append(submitted)
                receipts.extend(self.broker.process_open(event, as_of=clock))
                self.broker.observe_bar(following, as_of=clock)
                self._index += 1
                self._steps += 1
                after = self.ledger.snapshot()
                invalid_equity = previous.equity <= 0 or after.equity <= 0
                reward = -1.0 if invalid_equity else math.log(float(after.equity / previous.equity))
                terminated = invalid_equity or self._index == len(self.bars) - 1
                truncated = not terminated and self._index >= self.end_index
                self._done = terminated or truncated
                fills = [receipt.fill for receipt in receipts if receipt.fill is not None]
                closed_pnls: list[float] = []
                if before_shares == 0 and self.shares:
                    self._position_pnl_start = previous.realized_pnl
                if before_shares and not self.shares:
                    closed_pnls.append(float(after.realized_pnl - self._position_pnl_start))
                    self._position_pnl_start = after.realized_pnl
                rejected = [receipt.reason for receipt in receipts if receipt.status == OrderStatus.REJECTED]
                if not decision.approved:
                    rejected.append(decision.reason)
                info = self._info(
                    action=action, target_exposure=requested, executed_target_exposure=target,
                    transition_id=identifier, decision_index=self._index - 1,
                    decision_timestamp=current.timestamp, fill_index=self._index,
                    fill_timestamp=following.timestamp, fill_open=float(following.open),
                    previous_equity=float(previous.equity),
                    realized_fees=float(sum((fill.fees for fill in fills), Decimal("0"))),
                    turnover=float(sum((fill.cost.execution_notional for fill in fills), Decimal("0"))),
                    execution_cost=float(sum((fill.cost.execution_cost for fill in fills), Decimal("0"))),
                    buy_price=next((float(fill.execution_price) for fill in fills if fill.side == "BUY"), None),
                    sell_price=next((float(fill.execution_price) for fill in fills if fill.side == "SELL"), None),
                    fills=[fill.to_mapping() for fill in fills],
                    orders=[receipt.to_mapping() for receipt in receipts],
                    rejections=rejected, risk_decision=json.loads(_encode(asdict(decision))),
                    risk_decision_phase="completed_bar_request",
                    terminal_liquidation=terminal_exit, session_liquidation=session_exit,
                    liquidation_unfilled=bool((terminal_exit or session_exit) and self.shares),
                    closed_trade_pnls=closed_pnls, invalid_equity=invalid_equity,
                    terminated=bool(terminated), truncated=bool(truncated),
                )
                self._last_info = info
                record = {
                    "transition_id": identifier, "episode_id": self._episode_id,
                    "dataset_hash": self.dataset_hash, "config_hash": self.config_hash,
                    "policy_version": self.policy_version, "config_version": self.config_version,
                    "feature_version": self.FEATURE_VERSION, "feature_schema": self.feature_schema,
                    "index": self._index - 1, "next_index": self._index,
                    "timestamp": following.timestamp.isoformat(), "as_of": clock.isoformat(),
                    "observation": observation.tolist(), "action": action, "reward": float(reward),
                    "next_observation": self._observation().tolist(),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "fills": info["fills"], "info": json.loads(_encode(info)),
                }
                self._history_hash = _hash({"previous": self._history_hash, "transition": record})
                if self.repository:
                    if not self.repository.record_transition(identifier, record, occurred_at=following.timestamp):
                        raise PersistenceError("transition already exists outside the saved cursor")
                    self.repository.append_event(
                        f"TRANSITION:{identifier}", "TRANSITION", record,
                        aggregate_id=self._episode_id, occurred_at=following.timestamp,
                        idempotency_key=identifier,
                    )
                    self._save_cursor()
        except BaseException:
            self._restore_cursor(before_cursor)
            raise
        if self.repository is None:
            self._records.append(deepcopy(record))
        return self._result(record)

    def checkpoint(self) -> None:
        """Atomically persist manual risk-latch changes and the unchanged cursor."""

        with self.broker.atomic():
            self._assert_current_cursor()
            self.broker.checkpoint()
            if self.repository:
                self._save_cursor()

    def set_kill_switch(self, enabled: bool, *, reason: str = "operator stop") -> None:
        """Operator-only control: cancel local pending intents, retain holdings.

        No flattening is attempted with a stale or missing quote. The broker and
        environment cursor remain one atomic restart boundary.
        """
        if not isinstance(enabled, bool):
            raise ValueError("kill switch state must be boolean")
        with self.broker.atomic():
            self._assert_current_cursor()
            if enabled:
                self.risk_supervisor.engage_kill_switch(reason)
                self.broker.cancel_pending("operator kill switch: local intent cancelled")
            else:
                self.risk_supervisor.clear_kill_switch()
            self.broker.checkpoint()
            if self.repository:
                self._save_cursor()

    def transition_records(self) -> tuple[dict[str, Any], ...]:
        """Return detached, policy-versioned numerical/audit replay records."""

        if self.repository is None:
            return tuple(deepcopy(self._records))
        return tuple(
            payload for row in self.repository.transitions()
            if (payload := json.loads(row["payload"])).get("episode_id") == self._episode_id
        )

    def replay_transitions(self) -> tuple[Transition, ...]:
        """Return the learner's Transition objects without recomputing any money."""

        names = ("observation", "action", "reward", "next_observation", "terminated", "truncated")
        return tuple(Transition(**{key: record[key] for key in names}) for record in self.transition_records())

    def render(self) -> dict[str, Any]:
        """Return a detached account/cursor snapshot; no external I/O occurs."""

        return deepcopy(self._last_info)

    def close(self) -> None:
        """Leave the caller-owned repository open; each step is already durable."""

    @staticmethod
    def _check_length(length: int | None) -> None:
        if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length <= 0):
            raise ValueError("episode_length must be a positive transition count")

    def _select_episode(self, options: Mapping[str, Any]) -> tuple[int, int, Decimal]:
        length = options.get("episode_length", self.episode_length)
        self._check_length(length)
        last_start = len(self.bars) - 1 - (length or 1)
        start = options.get("start_index", self._configured_start)
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= last_start:
            raise ValueError("episode start/length must leave the requested following bars")
        if "start_index" in options and "start_date" in options:
            raise ValueError("choose start_index or start_date, not both")
        candidates = list(range(start, last_start + 1))
        timezone = ZoneInfo(self.risk_limits.session_timezone)
        dates = {index: self.bars[index].timestamp.astimezone(timezone).date() for index in candidates}
        random_date = options.get("random_start_date", self.random_start_date)
        if "start_date" in options or random_date:
            value = options.get("start_date")
            selected = date.fromisoformat(value) if isinstance(value, str) else value
            if selected is None:
                unique_dates = sorted(set(dates.values()))
                selected = unique_dates[int(self.np_random.integers(len(unique_dates)))]
            candidates = [index for index in candidates if dates[index] == selected]
            if not candidates:
                raise ValueError("start_date has no eligible episode start")
        if options.get("random_start", self.random_start):
            start = candidates[int(self.np_random.integers(len(candidates)))]
        else:
            start = candidates[0]
        end = start + length if length is not None else len(self.bars) - 1
        capital = self.initial_cash
        if "initial_cash" in options:
            capital = _positive_cash(options["initial_cash"])
        elif self.initial_cash_range:
            low, high = self.initial_cash_range
            capital = money(math.exp(self.np_random.uniform(math.log(float(low)), math.log(float(high)))))
            capital = min(high, max(low, capital))
        return start, end, capital

    def _observation(self) -> np.ndarray:
        snapshot = self.ledger.snapshot()
        equity = snapshot.equity
        exposure = Decimal(self.shares) * self.bars[self._index].close / equity if equity > 0 else 0
        cash_fraction = snapshot.cash / equity if equity > 0 else 0
        progress = (self._index - self.start_index) / max(1, self.end_index - self.start_index)
        values = (*self._market_features[self._index], float(cash_fraction), float(exposure), float(snapshot.drawdown), progress)
        if any(not math.isfinite(value) or abs(value) > np.finfo(np.float32).max for value in values):
            raise ValueError("observation exceeds the finite float32 feature schema")
        return np.asarray(values, dtype=np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        snapshot = self.ledger.snapshot()
        exposure = Decimal(self.shares) * self.bars[self._index].close / snapshot.equity if snapshot.equity > 0 else 0
        return {
            "index": self._index, "timestamp": self.bars[self._index].timestamp,
            "cash": float(snapshot.cash), "shares": self.shares, "equity": float(snapshot.equity),
            "high_water_mark": float(snapshot.high_water_mark), "drawdown": float(snapshot.drawdown),
            "exposure": float(exposure), "realized_fees": 0.0, "turnover": 0.0,
            "policy_version": self.policy_version, "config_version": self.config_version,
            "dataset_hash": self.dataset_hash, "episode_id": self._episode_id,
            "feature_schema": list(self.feature_schema), "fees_paid_total": float(snapshot.fees_paid),
            **extra,
        }

    def _decision_context(self, bar: MarketBar, as_of: datetime | None) -> RiskContext:
        snapshot = self.ledger.snapshot()
        return RiskContext(
            timestamp=utc_timestamp(as_of) if as_of is not None else bar.timestamp,
            feed_timestamp=bar.timestamp, cash=snapshot.cash, equity=snapshot.equity,
            high_water_mark=snapshot.high_water_mark, session_start_equity=self.broker.session_start_equity,
            positions={symbol: position.quantity for symbol, position in snapshot.positions.items()},
            marks=self.ledger.last_prices,
        )

    def _session_exit_due(self, current: MarketBar, following: MarketBar) -> bool:
        if not self.session_liquidation or self.risk_limits.market_close is None:
            return False
        timezone = ZoneInfo(self.risk_limits.session_timezone)
        before, after = current.timestamp.astimezone(timezone), following.timestamp.astimezone(timezone)
        deadline = datetime.combine(after.date(), self.risk_limits.market_close, timezone)
        return (
            before.date() == after.date()
            and self.risk_supervisor.market_is_open(following.timestamp)
            and after + self.bar_interval >= deadline
        )

    def _configuration(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT_VERSION, "config_version": self.config_version,
            "policy_version": self.policy_version, "feature_version": self.FEATURE_VERSION,
            "feature_schema": self.feature_schema, "feature_lookback": self.feature_lookback,
            "cost_model": asdict(self.cost_model), "risk_limits": asdict(self.risk_limits),
            "execution_version": PaperBroker.EXECUTION_VERSION,
            "target_exposures": self.target_exposures, "participation_limit": self.participation_limit,
            "terminal_liquidation": self.terminal_liquidation, "session_liquidation": self.session_liquidation,
            "bar_interval": self.bar_interval, "initial_cash": self.initial_cash,
            "start_index": self._configured_start, "episode_length": self.episode_length,
            "random_start": self.random_start, "random_start_date": self.random_start_date,
            "initial_cash_range": self.initial_cash_range,
        }

    def _cursor(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT_VERSION, "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash, "policy_version": self.policy_version,
            "config_version": self.config_version, "episode_id": self._episode_id,
            "history_hash": self._history_hash,
            "start_index": self.start_index, "end_index": self.end_index,
            "index": self._index, "steps": self._steps, "done": self._done,
            "initial_cash": str(self._episode_cash), "episode_options": self._episode_options,
            "position_pnl_start": str(self._position_pnl_start),
            "last_info": json.loads(_encode(self._last_info)),
            "rng_state": deepcopy(self.np_random.bit_generator.state),
            "rng_seed": self.np_random_seed,
            "broker_revision": self.broker.revision,
            "ledger_hash": _hash(self.ledger.export_state()),
        }

    def _save_cursor(self) -> None:
        if self.repository:
            self.repository.save_state(self.CHECKPOINT_KEY, self._cursor())

    def _restore_cursor(self, state: Mapping[str, Any]) -> None:
        self.start_index, self.end_index = state["start_index"], state["end_index"]
        self._index, self._steps, self._done = state["index"], state["steps"], state["done"]
        self._episode_id = state["episode_id"]
        self._history_hash = state["history_hash"]
        self._episode_cash = Decimal(state["initial_cash"])
        self._episode_options = dict(state["episode_options"])
        self._position_pnl_start = Decimal(state["position_pnl_start"])
        self._last_info = self._restore_info(state["last_info"])
        self.np_random.bit_generator.state = deepcopy(state["rng_state"])
        self._np_random_seed = state["rng_seed"]

    def _validate_checkpoint(self, state: Mapping[str, Any]) -> None:
        for key, expected in (
            ("format", self.FORMAT_VERSION), ("dataset_hash", self.dataset_hash),
            ("policy_version", self.policy_version), ("config_version", self.config_version),
            ("config_hash", self.config_hash),
        ):
            if state.get(key) != expected:
                raise ValueError(f"paper environment checkpoint {key} mismatch")
        if not 0 <= state["start_index"] <= state["index"] <= state["end_index"] < len(self.bars):
            raise PersistenceError("persisted cursor is outside its episode/data bounds")
        if state["steps"] != state["index"] - state["start_index"]:
            raise PersistenceError("persisted transition count and cursor do not agree")

    def _assert_current_cursor(self) -> None:
        if self.repository:
            state = self.repository.load_state(self.CHECKPOINT_KEY)
            if state is None or state["index"] != self._index or state["broker_revision"] != self.broker.revision:
                raise PersistenceError("paper cursor changed in another runner; resume before advancing")

    def _validate_history(self) -> None:
        records = self.transition_records()
        if len(records) != self._steps:
            raise PersistenceError("saved transition history does not match the cursor count")
        digest = _hash({"episode": self._episode_id})
        for offset, record in enumerate(records):
            expected_index = self.start_index + offset
            if record["index"] != expected_index or record["next_index"] != expected_index + 1:
                raise PersistenceError("saved transition history is not a contiguous causal path")
            digest = _hash({"previous": digest, "transition": record})
        if digest != self._history_hash:
            raise PersistenceError("saved transition history digest does not match the checkpoint")
        if records and records[-1]["transition_id"] != self._last_info.get("transition_id"):
            raise PersistenceError("saved cursor does not identify its last transition")

    def _find_transition(self, identifier: str) -> dict[str, Any] | None:
        if self.repository:
            row = self.repository.get_transition(identifier)
            record = json.loads(row["payload"]) if row else None
        else:
            record = next((row for row in self._records if row["transition_id"] == identifier), None)
        if record is not None and record.get("episode_id") != self._episode_id:
            raise PersistenceError("transition ID belongs to a different episode")
        return record

    @staticmethod
    def _restore_info(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(payload))
        for name in ("timestamp", "decision_timestamp", "fill_timestamp"):
            if isinstance(result.get(name), str):
                result[name] = datetime.fromisoformat(result[name])
        return result

    @classmethod
    def _result(cls, record: Mapping[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return (
            np.asarray(record["next_observation"], dtype=np.float32), float(record["reward"]),
            bool(record["terminated"]), bool(record["truncated"]), cls._restore_info(record["info"]),
        )


__all__ = ["PaperTradingEnv"]
