"""Provider-independent continuous paper replay over the canonical broker.

This runner is deliberately separate from ``PaperTradingEnv``.  The Gym
environment owns an immutable finite dataset and remains the canonical
training/evaluation path.  ``ContinuousPaperRunner`` consumes completed-bar
events, persists a small incremental cursor, and reuses the existing
``PaperBroker``, ``PortfolioLedger``, ``RiskSupervisor`` and SQLite repository.
It has no real order route and does not claim a verified live provider.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import logging
import math
from numbers import Integral
from threading import Event
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .data import CompletedBarEvent, CompletedBarFeed, MarketBar, ProviderIdentity
from .data.bars import utc_timestamp
from .execution import OrderStatus, PaperBroker, PaperOpenEvent
from .features import CausalFeatureBuilder
from .paper_env import PaperTradingEnv
from .portfolio import CostModel, PortfolioLedger
from .risk import RiskLimits, RiskSupervisor
from .storage import PersistenceError, SQLiteRepository
from .data.stream import RetryPolicy, fetch_with_retry


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _bar_hash(bar: MarketBar) -> str:
    return _hash(bar.to_mapping())


@dataclass(frozen=True, slots=True)
class ContinuousRunnerStatus:
    """Detached operational status suitable for CLI/report output."""

    runner_format: str
    symbol: str
    provider_id: str
    provenance: str
    policy_id: str
    last_bar_timestamp: str | None
    last_received_at: str | None
    pending_action: int | None
    transitions: int
    kill_switch: bool
    feed_stale: bool
    account: Mapping[str, Any]


class ContinuousPaperRunner:
    """Incremental completed-bar runner backed by one durable paper account.

    A provider event is processed exactly once after its completed-bar
    availability time.  The policy action from the previous completed bar is
    persisted before the following open event is processed.  All portfolio
    changes go through ``PaperBroker``; the runner never mutates ledger money
    directly.
    """

    STATE_KEY = "continuous_paper_runner_v1"
    FORMAT = "continuous-paper-v1"
    EXECUTION_VERSION = PaperBroker.EXECUTION_VERSION

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        symbol: str,
        provider: ProviderIdentity,
        policy: Callable[[np.ndarray], int],
        policy_id: str = "cash-v1",
        initial_cash: Decimal | str | float = 10_000,
        cost_model: CostModel | None = None,
        risk_limits: RiskLimits | None = None,
        target_exposures: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        participation_limit: Decimal | str | float = 1,
        config_version: str = "continuous-paper-v1",
        feature_lookback: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if provider.interval <= timedelta(0):
            raise ValueError("provider interval must be positive")
        exposures = tuple(float(value) for value in target_exposures)
        if (
            not exposures
            or exposures[0] != 0
            or exposures != tuple(sorted(set(exposures)))
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in exposures)
        ):
            raise ValueError("target_exposures must be sorted unique weights including cash")
        if not isinstance(config_version, str) or not config_version.strip():
            raise ValueError("config_version must be non-empty")

        self.repository = repository
        self.symbol = symbol.strip()
        self.provider = provider
        self.policy = policy
        self.policy_id = policy_id.strip()
        self.cost_model = cost_model or CostModel()
        self.risk_limits = risk_limits or RiskLimits(
            permitted_symbols=frozenset({self.symbol}),
        )
        self.target_exposures = exposures
        self.participation_limit = participation_limit
        self.config_version = config_version.strip()
        self.feature_lookback = feature_lookback
        self.logger = logger or logging.getLogger(__name__)
        self._feature_schema = PaperTradingEnv.feature_schema
        self._environment_hash = _hash(self._contract())
        self._config_hash = _hash({**self._contract(), "policy_id": self.policy_id})
        self._bars: list[MarketBar] = []
        self._pending_action: int | None = None
        self._pending_observation: list[float] | None = None
        self._pending_decision_at: datetime | None = None
        self._previous_equity: Decimal | None = None
        self._last_bar_timestamp: datetime | None = None
        self._last_received_at: datetime | None = None
        self._last_event_id: str | None = None
        self._last_bar_hash: str | None = None
        self._history_hash = _hash({"runner": self.FORMAT, "symbol": self.symbol})
        self._transitions = 0
        self._bars_processed = 0
        self._feed_stale = False
        self._stopped = False

        persisted = repository.load_state(self.STATE_KEY)
        persisted_broker = repository.load_state(PaperBroker.STATE_KEY)
        if persisted is None and persisted_broker is not None:
            raise PersistenceError("repository contains an unmanaged paper broker account")
        if persisted is not None and persisted_broker is None:
            raise PersistenceError("continuous runner cursor has no paper broker checkpoint")
        self.broker = PaperBroker(
            PortfolioLedger(initial_cash),
            cost_model=self.cost_model,
            risk_supervisor=RiskSupervisor(self.risk_limits),
            repository=repository,
            participation_limit=participation_limit,
            bar_interval=provider.interval,
        )
        if persisted is None:
            self._save_state()
        else:
            self._restore_state(persisted)
            self._restore_bars()

    def _contract(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "config_version": self.config_version,
            "symbol": self.symbol,
            "provider": self.provider.to_mapping(),
            "feature_version": PaperTradingEnv.FEATURE_VERSION,
            "feature_schema": list(self._feature_schema),
            "reward_version": PaperTradingEnv.REWARD_VERSION,
            "execution_version": self.EXECUTION_VERSION,
            "cost_model_version": self.cost_model.version,
            "risk_version": self.risk_limits.version,
            "target_exposures": list(self.target_exposures),
            "feature_lookback": self.feature_lookback,
        }

    def _state(self) -> dict[str, Any]:
        snapshot = self.broker.ledger.snapshot()
        return {
            "format": self.FORMAT,
            "config_hash": self._config_hash,
            "environment_hash": self._environment_hash,
            "contract": self._contract(),
            "symbol": self.symbol,
            "policy_id": self.policy_id,
            "provider": self.provider.to_mapping(),
            "last_bar_timestamp": self._last_bar_timestamp.isoformat() if self._last_bar_timestamp else None,
            "last_received_at": self._last_received_at.isoformat() if self._last_received_at else None,
            "last_event_id": self._last_event_id,
            "last_bar_hash": self._last_bar_hash,
            "pending_action": self._pending_action,
            "pending_observation": self._pending_observation,
            "pending_decision_at": self._pending_decision_at.isoformat() if self._pending_decision_at else None,
            "previous_equity": str(self._previous_equity) if self._previous_equity is not None else None,
            "history_hash": self._history_hash,
            "transitions": self._transitions,
            "bars_processed": len(self._bars),
            "feed_stale": self._feed_stale,
            "broker_revision": self.broker.revision,
            "ledger_hash": _hash(self.broker.ledger.export_state()),
            "account": snapshot.to_mapping(),
        }

    def _save_state(self) -> None:
        self.repository.save_state(self.STATE_KEY, self._state())

    def _cursor_state(self) -> dict[str, Any]:
        """Return the mutable runner fields that must roll back with SQLite."""

        return {
            "bars": list(self._bars),
            "pending_action": self._pending_action,
            "pending_observation": deepcopy(self._pending_observation),
            "pending_decision_at": self._pending_decision_at,
            "previous_equity": self._previous_equity,
            "last_bar_timestamp": self._last_bar_timestamp,
            "last_received_at": self._last_received_at,
            "last_event_id": self._last_event_id,
            "last_bar_hash": self._last_bar_hash,
            "history_hash": self._history_hash,
            "transitions": self._transitions,
            "bars_processed": self._bars_processed,
            "feed_stale": self._feed_stale,
        }

    def _restore_cursor_state(self, state: Mapping[str, Any]) -> None:
        self._bars = list(state["bars"])
        self._pending_action = state["pending_action"]
        self._pending_observation = deepcopy(state["pending_observation"])
        self._pending_decision_at = state["pending_decision_at"]
        self._previous_equity = state["previous_equity"]
        self._last_bar_timestamp = state["last_bar_timestamp"]
        self._last_received_at = state["last_received_at"]
        self._last_event_id = state["last_event_id"]
        self._last_bar_hash = state["last_bar_hash"]
        self._history_hash = state["history_hash"]
        self._transitions = state["transitions"]
        self._bars_processed = state["bars_processed"]
        self._feed_stale = state["feed_stale"]

    @contextmanager
    def _atomic(self):
        """Make the runner cursor and broker checkpoint one rollback unit."""

        before = self._cursor_state()
        try:
            with self.broker.atomic():
                yield
        except BaseException:
            self._restore_cursor_state(before)
            raise

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("format") != self.FORMAT:
            raise PersistenceError("unsupported continuous runner state format")
        if state.get("config_hash") != self._config_hash or state.get("environment_hash") != self._environment_hash:
            raise PersistenceError("continuous runner configuration changed")
        if state.get("symbol") != self.symbol or state.get("policy_id") != self.policy_id:
            raise PersistenceError("continuous runner symbol or policy changed")
        if state.get("provider") != self.provider.to_mapping():
            raise PersistenceError("continuous runner provider identity changed")
        if not isinstance(state.get("transitions"), int) or state["transitions"] < 0:
            raise PersistenceError("continuous runner transition count is invalid")
        if (
            isinstance(state.get("bars_processed"), bool)
            or not isinstance(state.get("bars_processed"), int)
            or state["bars_processed"] < 0
        ):
            raise PersistenceError("continuous runner bar count is invalid")
        self._last_bar_timestamp = (
            utc_timestamp(datetime.fromisoformat(state["last_bar_timestamp"]))
            if state.get("last_bar_timestamp") else None
        )
        self._last_received_at = (
            utc_timestamp(datetime.fromisoformat(state["last_received_at"]))
            if state.get("last_received_at") else None
        )
        self._last_event_id = state.get("last_event_id")
        self._last_bar_hash = state.get("last_bar_hash")
        action = state.get("pending_action")
        if action is not None and (
            isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(self.target_exposures)
        ):
            raise PersistenceError("continuous runner pending action is invalid")
        self._pending_action = action
        observation = state.get("pending_observation")
        if observation is not None and (
            not isinstance(observation, list)
            or len(observation) != len(self._feature_schema)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in observation)
        ):
            raise PersistenceError("continuous runner pending observation is invalid")
        self._pending_observation = list(observation) if observation is not None else None
        self._pending_decision_at = (
            utc_timestamp(datetime.fromisoformat(state["pending_decision_at"]))
            if state.get("pending_decision_at") else None
        )
        self._previous_equity = Decimal(state["previous_equity"]) if state.get("previous_equity") is not None else None
        self._history_hash = state.get("history_hash", self._history_hash)
        self._transitions = state["transitions"]
        self._bars_processed = state["bars_processed"]
        self._feed_stale = bool(state.get("feed_stale", False))
        if self.broker.revision != state.get("broker_revision"):
            raise PersistenceError("continuous runner and broker revisions disagree")
        if _hash(self.broker.ledger.export_state()) != state.get("ledger_hash"):
            raise PersistenceError("continuous runner and broker ledger hashes disagree")

    def _restore_bars(self) -> None:
        self._bars = []
        for row in self.repository.events(event_type="STREAM_BAR"):
            payload = json.loads(row["payload"])
            event = payload.get("event")
            if not isinstance(event, Mapping) or not isinstance(event.get("bar"), Mapping):
                raise PersistenceError("continuous bar event payload is invalid")
            bar = MarketBar.from_mapping(event["bar"])
            if bar.symbol != self.symbol:
                raise PersistenceError("continuous bar event symbol changed")
            self._bars.append(bar)
        if len(self._bars) != self._bars_processed:
            raise PersistenceError("continuous bar history and cursor count disagree")

    @property
    def kill_switch(self) -> bool:
        return self.broker.risk_supervisor.kill_switch_active

    @property
    def current_index(self) -> int:
        return len(self._bars) - 1

    def status(self) -> ContinuousRunnerStatus:
        return ContinuousRunnerStatus(
            self.FORMAT,
            self.symbol,
            self.provider.provider_id,
            self.provider.provenance,
            self.policy_id,
            self._last_bar_timestamp.isoformat() if self._last_bar_timestamp else None,
            self._last_received_at.isoformat() if self._last_received_at else None,
            self._pending_action,
            self._transitions,
            self.kill_switch,
            self._feed_stale,
            self.broker.ledger.snapshot().to_mapping(),
        )

    def _validate_event(self, event: CompletedBarEvent) -> None:
        if event.provider.to_mapping() != self.provider.to_mapping():
            raise PersistenceError("continuous event provider identity does not match the account")
        if event.bar.symbol != self.symbol:
            raise ValueError("continuous event symbol does not match the account")
        if event.bar.timestamp + self.provider.interval != event.available_at:
            raise ValueError("continuous event completion boundary is invalid")
        if self._last_received_at is not None and event.received_at < self._last_received_at:
            raise ValueError("continuous event receipt clock moved backwards")
        if self._last_bar_timestamp is not None and event.bar.timestamp <= self._last_bar_timestamp:
            if event.provider_event_id == self._last_event_id and _bar_hash(event.bar) == self._last_bar_hash:
                raise PersistenceError("duplicate continuous event")
            raise ValueError("continuous bar timestamp moved backwards or duplicated")
        if self._bars:
            previous = self._bars[-1]
            gap = event.bar.timestamp - previous.timestamp
            if gap <= timedelta(0):
                raise ValueError("continuous bars must be strictly chronological")
            zone = ZoneInfo(self.risk_limits.session_timezone)
            if (
                previous.timestamp.astimezone(zone).date() == event.bar.timestamp.astimezone(zone).date()
                and gap != self.provider.interval
            ):
                raise ValueError("missing or irregular continuous bar inside a session")

    def _observation(self, bars: Sequence[MarketBar]) -> np.ndarray:
        features = CausalFeatureBuilder(self.feature_lookback).transform(bars)
        bar = bars[-1]
        row = features[-1]
        snapshot = self.broker.ledger.snapshot()
        equity = snapshot.equity
        mark = self.broker.ledger.last_prices.get(self.symbol, bar.close)
        exposure = Decimal(self.broker.ledger.position_quantity(self.symbol)) * mark / equity if equity > 0 else Decimal("0")
        cash_fraction = snapshot.cash / equity if equity > 0 else Decimal("0")
        values = (
            float(bar.open / bar.close - 1),
            float(bar.high / bar.close - 1),
            float(bar.low / bar.close - 1),
            row.log_return or 0.0,
            row.volume_ratio - 1 if row.volume_ratio is not None else 0.0,
            float(cash_fraction),
            float(exposure),
            float(snapshot.drawdown),
            0.0,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("continuous observation contains a non-finite value")
        return np.asarray(values, dtype=np.float32)

    def _policy_action(self, observation: np.ndarray) -> int:
        raw = self.policy(observation)
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise ValueError("continuous policy returned a non-integer action")
        action = int(raw)
        if not 0 <= action < len(self.target_exposures):
            raise ValueError("continuous policy returned an action outside the account action space")
        return action

    def _append_health(self, event_type: str, payload: Mapping[str, Any], *, occurred_at: datetime) -> None:
        self.repository.append_event(
            f"STREAM_HEALTH:{event_type}:{occurred_at.isoformat()}",
            "STREAM_HEALTH",
            {"status": event_type, **dict(payload), "provider": self.provider.to_mapping()},
            aggregate_id=self.symbol,
            occurred_at=occurred_at,
        )

    def process_event(self, event: CompletedBarEvent) -> dict[str, Any]:
        """Process one completed event exactly once and persist the cursor."""

        if event.provider.to_mapping() != self.provider.to_mapping():
            raise PersistenceError("continuous event provider identity does not match the account")
        if event.bar.symbol != self.symbol:
            raise ValueError("continuous event symbol does not match the account")
        if (
            self._last_event_id == event.provider_event_id
            and self._last_bar_hash == _bar_hash(event.bar)
        ):
            return {"event_id": event.provider_event_id, "duplicate": True, "status": asdict(self.status())}
        self._validate_event(event)
        stale = event.received_at - event.available_at > self.risk_limits.stale_after
        before_snapshot = self.broker.ledger.snapshot()
        receipts = ()
        transition: dict[str, Any] | None = None
        observation_before = self._pending_observation
        action_before = self._pending_action
        previous_equity = self._previous_equity
        next_bars = (*self._bars, event.bar)
        with self._atomic():
            if action_before is not None and not self.kill_switch:
                open_event = PaperOpenEvent(event.bar.symbol, event.bar.timestamp, event.bar.open)
                order = self.broker.plan_target_order(
                    f"stream:{event.provider_event_id}:order",
                    self.target_exposures[action_before],
                    decision_at=self._pending_decision_at or event.available_at,
                    event=open_event,
                )
                if order is not None:
                    self.broker.submit_order(order)
                receipts = self.broker.process_open(open_event, as_of=event.received_at)
            elif self.kill_switch:
                self.broker.cancel_pending("continuous paper kill switch")
            self.broker.observe_bar(event.bar, as_of=event.received_at)
            next_observation = self._observation(next_bars)
            after_snapshot = self.broker.ledger.snapshot()
            if action_before is not None and observation_before is not None and previous_equity is not None:
                invalid = previous_equity <= 0 or after_snapshot.equity <= 0
                reward = -1.0 if invalid else math.log(float(after_snapshot.equity / previous_equity))
                transition_id = f"stream:{event.provider_event_id}"
                info = {
                    "source": event.provider.to_mapping(),
                    "available_at": event.available_at.isoformat(),
                    "received_at": event.received_at.isoformat(),
                    "stale_feed": stale,
                    "orders": [receipt.to_mapping() for receipt in receipts],
                    "fills": [receipt.fill.to_mapping() for receipt in receipts if receipt.fill is not None],
                    "previous_equity": float(previous_equity),
                    "equity": float(after_snapshot.equity),
                    "cash": float(after_snapshot.cash),
                    "shares": self.broker.ledger.position_quantity(self.symbol),
                    "policy_id": self.policy_id,
                    "execution_version": self.EXECUTION_VERSION,
                }
                record = {
                    "transition_id": transition_id,
                    "episode_id": f"continuous:{self.symbol}",
                    "dataset_hash": None,
                    "source_dataset_hash": None,
                    "environment_hash": self._environment_hash,
                    "instrument": self.symbol,
                    "policy_version": self.policy_id,
                    "config_version": self.config_version,
                    "feature_version": PaperTradingEnv.FEATURE_VERSION,
                    "feature_schema": self._feature_schema,
                    "reward_version": PaperTradingEnv.REWARD_VERSION,
                    "execution_version": self.EXECUTION_VERSION,
                    "cost_model_version": self.cost_model.version,
                    "risk_version": self.risk_limits.version,
                    "index": self._transitions,
                    "next_index": self._transitions + 1,
                    "timestamp": event.bar.timestamp.isoformat(),
                    "as_of": event.received_at.isoformat(),
                    "provider_event_id": event.provider_event_id,
                    "provider": event.provider.to_mapping(),
                    "bar_hash": _bar_hash(event.bar),
                    "observation": list(observation_before),
                    "action": action_before,
                    "reward": float(reward),
                    "next_observation": next_observation.tolist(),
                    "terminated": False,
                    "truncated": False,
                    "fills": info["fills"],
                    "info": info,
                }
                self.repository.record_transition(transition_id, record, occurred_at=event.received_at)
                self._history_hash = _hash({"previous": self._history_hash, "transition": record})
                self._transitions += 1
                transition = record

            self.repository.append_event(
                f"STREAM_BAR:{event.provider_event_id}",
                "STREAM_BAR",
                {"event": event.to_mapping(), "stale_feed": stale},
                aggregate_id=self.symbol,
                occurred_at=event.received_at,
                idempotency_key=event.provider_event_id,
            )
            if stale:
                self._feed_stale = True
                self._append_health("STALE_FEED", {
                    "available_at": event.available_at.isoformat(),
                    "received_at": event.received_at.isoformat(),
                }, occurred_at=event.received_at)
                next_action = None
            elif self.kill_switch:
                next_action = None
            else:
                next_action = self._policy_action(next_observation)
                self.repository.append_event(
                    f"STREAM_DECISION:{event.provider_event_id}",
                    "STREAM_DECISION",
                    {
                        "action": next_action,
                        "policy_id": self.policy_id,
                        "available_at": event.available_at.isoformat(),
                        "provider_event_id": event.provider_event_id,
                    },
                    aggregate_id=self.symbol,
                    occurred_at=event.received_at,
                    idempotency_key=f"decision:{event.provider_event_id}",
                )
                self._feed_stale = False

            self._bars = list(next_bars)
            self._bars_processed = len(self._bars)
            self._pending_action = next_action
            self._pending_observation = next_observation.tolist() if next_action is not None else None
            self._pending_decision_at = event.available_at if next_action is not None else None
            self._previous_equity = after_snapshot.equity
            self._last_bar_timestamp = event.bar.timestamp
            self._last_received_at = event.received_at
            self._last_event_id = event.provider_event_id
            self._last_bar_hash = _bar_hash(event.bar)
            self._save_state()
        self.logger.info(
            "continuous paper event provider=%s provenance=%s symbol=%s bar=%s stale=%s transition=%s",
            event.provider.provider_id,
            event.provider.provenance,
            self.symbol,
            event.bar.timestamp.isoformat(),
            stale,
            transition is not None,
        )
        return {
            "event_id": event.provider_event_id,
            "transition": transition,
            "receipts": [receipt.to_mapping() for receipt in receipts],
            "status": asdict(self.status()),
            "previous_account": before_snapshot.to_mapping(),
        }

    def run_events(
        self,
        events: Iterable[CompletedBarEvent],
        *,
        max_events: int | None = None,
        stop_event: Event | None = None,
    ) -> list[dict[str, Any]]:
        """Run a finite deterministic sequence with graceful stop support."""

        if max_events is not None and (isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0):
            raise ValueError("max_events must be a positive integer")
        results: list[dict[str, Any]] = []
        for event in events:
            if stop_event is not None and stop_event.is_set():
                self.shutdown("stop requested")
                break
            if max_events is not None and len(results) >= max_events:
                break
            try:
                results.append(self.process_event(event))
            except PersistenceError as error:
                if str(error) == "duplicate continuous event":
                    continue
                raise
        return results

    def run_polling(
        self,
        feed: CompletedBarFeed,
        *,
        until: datetime,
        clock: Callable[[], datetime],
        poll_seconds: float = 1.0,
        max_cycles: int | None = None,
        retry_policy: RetryPolicy = RetryPolicy(),
        sleep: Callable[[float], None] = __import__("time").sleep,
        stop_event: Event | None = None,
    ) -> int:
        """Poll a provider with bounded retries and a stale-feed health path."""

        if not math.isfinite(poll_seconds) or poll_seconds < 0:
            raise ValueError("poll_seconds must be finite and non-negative")
        if max_cycles is not None and (isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles <= 0):
            raise ValueError("max_cycles must be a positive integer")
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            if stop_event is not None and stop_event.is_set():
                self.shutdown("stop requested")
                break
            now = utc_timestamp(clock())
            events = fetch_with_retry(
                feed,
                self.symbol,
                self._last_bar_timestamp,
                utc_timestamp(until),
                as_of=now,
                policy=retry_policy,
                sleep=sleep,
            )
            if any(event.received_at > now for event in events):
                raise ValueError("provider returned an event with a future receipt timestamp")
            for event in events:
                self.process_event(event)
            if not events and self._last_received_at is not None and now - self._last_received_at > self.risk_limits.stale_after:
                with self._atomic():
                    self._feed_stale = True
                    self._append_health("STALE_FEED", {"received_at": now.isoformat()}, occurred_at=now)
                    self._save_state()
            cycles += 1
            if poll_seconds:
                sleep(poll_seconds)
        return cycles

    def set_kill_switch(self, enabled: bool, *, reason: str = "operator stop") -> dict[str, Any]:
        """Persistently stop new continuous orders while retaining holdings."""

        if not isinstance(enabled, bool):
            raise ValueError("kill switch state must be boolean")
        occurred_at = self._last_received_at or datetime.now(timezone.utc)
        with self._atomic():
            if enabled:
                self.broker.risk_supervisor.engage_kill_switch(reason)
                self.broker.cancel_pending("continuous paper kill switch: local intent cancelled")
            else:
                self.broker.risk_supervisor.clear_kill_switch()
            self.broker.checkpoint()
            self.repository.append_event(
                f"STREAM_KILL:{self.symbol}:{int(enabled)}:{occurred_at.isoformat()}",
                "STREAM_KILL",
                {"enabled": enabled, "reason": reason, "provider": self.provider.to_mapping()},
                aggregate_id=self.symbol,
                occurred_at=occurred_at,
            )
            self._pending_action = None if enabled else self._pending_action
            self._pending_observation = None if enabled else self._pending_observation
            self._pending_decision_at = None if enabled else self._pending_decision_at
            self._save_state()
        return asdict(self.status())

    def shutdown(self, reason: str = "operator stop") -> None:
        """Record a graceful stop without changing the persistent kill latch."""

        if self._stopped:
            return
        occurred_at = self._last_received_at or datetime.now(timezone.utc)
        with self.broker.atomic():
            self.repository.append_event(
                f"STREAM_STOP:{self.symbol}:{occurred_at.isoformat()}:{reason}",
                "STREAM_STOP",
                {"reason": reason, "provider": self.provider.to_mapping()},
                aggregate_id=self.symbol,
                occurred_at=occurred_at,
            )
            self._save_state()
        self._stopped = True


__all__ = ["ContinuousPaperRunner", "ContinuousRunnerStatus"]
