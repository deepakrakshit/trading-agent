"""Provider-independent completed-bar paper runner contracts."""

from datetime import datetime, timedelta, timezone
import pytest

from trading_agent.continuous import ContinuousPaperRunner
from trading_agent.data import (
    CompletedBarEvent,
    FeedUnavailableError,
    InMemoryBarProvider,
    ProviderIdentity,
    RangeProviderFeed,
    ReplayBarFeed,
    RetryPolicy,
    fetch_with_retry,
    synthetic_bars,
)
from trading_agent.risk import RiskLimits
from trading_agent.storage import SQLiteRepository


START = datetime(2026, 1, 5, 4, 15, tzinfo=timezone.utc)


def bars():
    return synthetic_bars("ABC", START, (100, 101, 102, 103, 104, 105), volume=1000)


def identity(provider_id="fixture"):
    return ProviderIdentity(provider_id, "SYNTHETIC")


def limits():
    return RiskLimits(
        permitted_symbols=frozenset({"ABC"}),
        daily_loss_limit="0.99",
        max_drawdown="0.99",
    )


def open_runner(repository, provider=None, policy=None):
    provider = provider or identity()
    return ContinuousPaperRunner(
        repository,
        symbol="ABC",
        provider=provider,
        policy=policy or (lambda observation: 1),
        target_exposures=(0.0, 1.0),
        initial_cash=1000,
        risk_limits=limits(),
    )


def test_replay_runner_uses_completed_bar_boundary_and_persists_transitions(tmp_path):
    provider = identity()
    feed = ReplayBarFeed(bars(), provider)
    with SQLiteRepository(tmp_path / "stream.sqlite") as repository:
        runner = open_runner(repository, provider)
        results = runner.run_events(feed.events())

        assert len(results) == len(feed.events())
        assert runner.status().transitions == len(feed.events()) - 1
        assert runner.status().account["positions"]["ABC"]["quantity"] > 0
        assert len(repository.events(event_type="STREAM_BAR")) == len(feed.events())
        assert len(repository.transitions()) == len(feed.events()) - 1
        assert repository.events(event_type="STREAM_DECISION")
        assert not repository.events(event_type="STREAM_HEALTH")


def test_replay_runner_restart_matches_uninterrupted_account(tmp_path):
    provider = identity()
    feed = ReplayBarFeed(bars(), provider)

    with SQLiteRepository(tmp_path / "full.sqlite") as repository:
        full = open_runner(repository, provider)
        full.run_events(feed.events())
        expected = full.status()

    path = tmp_path / "partial.sqlite"
    with SQLiteRepository(path) as repository:
        partial = open_runner(repository, provider)
        partial.run_events(feed.events()[:3])

    with SQLiteRepository(path) as repository:
        resumed = open_runner(repository, provider)
        resumed.run_events(feed.events()[3:])
        assert resumed.status().account == expected.account
        assert resumed.status().transitions == expected.transitions
        assert resumed.status().last_bar_timestamp == expected.last_bar_timestamp
        duplicate = resumed.process_event(feed.events()[-1])
        assert duplicate["duplicate"] is True


def test_runner_rejects_provider_mismatch_and_non_monotonic_events(tmp_path):
    provider = identity()
    events = ReplayBarFeed(bars(), provider).events()
    with SQLiteRepository(tmp_path / "stream.sqlite") as repository:
        runner = open_runner(repository, provider)
        runner.process_event(events[0])
        with pytest.raises(ValueError, match="chronological|backwards"):
            runner.process_event(events[0].__class__(
                events[0].bar,
                events[0].available_at,
                events[0].received_at,
                "different-event-id",
                provider,
            ))
        other = identity("other-provider")
        mismatched = CompletedBarEvent(
            events[1].bar,
            events[1].available_at,
            events[1].received_at,
            "other-event",
            other,
        )
        with pytest.raises(RuntimeError, match="provider identity"):
            runner.process_event(mismatched)


def test_stale_event_stops_new_policy_actions_and_records_health(tmp_path):
    provider = identity()
    events = list(ReplayBarFeed(bars(), provider).events())
    stale = CompletedBarEvent(
        events[1].bar,
        events[1].available_at,
        events[1].available_at + timedelta(minutes=20),
        events[1].provider_event_id,
        provider,
    )
    with SQLiteRepository(tmp_path / "stale.sqlite") as repository:
        runner = open_runner(repository, provider)
        runner.process_event(events[0])
        result = runner.process_event(stale)

        assert result["receipts"]
        assert not result["receipts"][0]["fill"]
        assert runner.status().feed_stale
        assert runner.status().pending_action is None
        assert repository.events(event_type="STREAM_HEALTH")


def test_kill_switch_is_persistent_and_cancels_pending_intent(tmp_path):
    provider = identity()
    events = ReplayBarFeed(bars(), provider).events()
    path = tmp_path / "kill.sqlite"
    with SQLiteRepository(path) as repository:
        runner = open_runner(repository, provider)
        runner.process_event(events[0])
        assert runner.status().pending_action == 1
        status = runner.set_kill_switch(True, reason="test stop")
        assert status["kill_switch"]
        assert status["pending_action"] is None

    with SQLiteRepository(path) as repository:
        restored = open_runner(repository, provider)
        assert restored.kill_switch
        result = restored.process_event(events[1])
        assert result["receipts"] == []
        assert restored.status().account["positions"] == {}


def test_retry_policy_is_bounded_and_never_substitutes_data():
    provider = identity()
    event = ReplayBarFeed(bars(), provider).events()[0]

    class FlakyFeed:
        identity = provider

        def __init__(self):
            self.attempts = 0

        def fetch(self, symbol, after, until, *, as_of):
            self.attempts += 1
            if self.attempts < 3:
                raise FeedUnavailableError("temporary provider outage")
            return (event,)

    delays = []
    feed = FlakyFeed()
    result = fetch_with_retry(
        feed,
        "ABC",
        None,
        event.bar.timestamp + timedelta(minutes=10),
        as_of=event.received_at,
        policy=RetryPolicy(max_attempts=3, backoff_seconds=0.25, max_delay_seconds=1),
        sleep=delays.append,
    )
    assert result == (event,)
    assert feed.attempts == 3
    assert delays == [0.25, 0.5]

    class AlwaysDown(FlakyFeed):
        def fetch(self, symbol, after, until, *, as_of):
            raise FeedUnavailableError("provider remains unavailable")

    failing = AlwaysDown()
    with pytest.raises(FeedUnavailableError):
        fetch_with_retry(
            failing,
            "ABC",
            None,
            event.bar.timestamp + timedelta(minutes=10),
            as_of=event.received_at,
            policy=RetryPolicy(max_attempts=1),
            sleep=lambda _: None,
        )


def test_range_provider_adapter_exposes_only_completed_bars():
    provider = identity()
    source = InMemoryBarProvider(bars(), provenance="SYNTHETIC")
    feed = RangeProviderFeed(source, provider)
    all_events = feed.fetch(
        "ABC",
        None,
        bars()[-1].timestamp + provider.interval,
        as_of=bars()[-1].timestamp + provider.interval,
    )
    partial_events = feed.fetch(
        "ABC",
        None,
        bars()[-1].timestamp + provider.interval,
        as_of=bars()[-1].timestamp + provider.interval - timedelta(seconds=1),
    )
    assert len(all_events) == len(bars())
    assert len(partial_events) == len(bars()) - 1
    assert all(event.bar.source == "SYNTHETIC" for event in all_events)


def test_runner_cursor_rolls_back_when_the_durable_cursor_write_fails(tmp_path, monkeypatch):
    provider = identity()
    event = ReplayBarFeed(bars(), provider).events()[0]
    with SQLiteRepository(tmp_path / "rollback.sqlite") as repository:
        runner = open_runner(repository, provider)
        original_save_state = repository.save_state

        def fail_runner_cursor(key, payload):
            if key == runner.STATE_KEY:
                raise RuntimeError("injected cursor failure")
            return original_save_state(key, payload)

        monkeypatch.setattr(repository, "save_state", fail_runner_cursor)
        with pytest.raises(RuntimeError, match="injected cursor failure"):
            runner.process_event(event)
        assert runner.status().last_bar_timestamp is None
        assert runner.status().transitions == 0
        assert repository.load_state(runner.STATE_KEY)["last_bar_timestamp"] is None
        assert repository.load_state("paper_broker_v1")["ledger"]["cash"] == "1000.00"


def test_polling_uses_provider_events_and_marks_a_late_empty_poll(tmp_path):
    provider = identity()
    events = ReplayBarFeed(bars(), provider).events()

    class PollFeed:
        identity = provider

        def fetch(self, symbol, after, until, *, as_of):
            if after is not None:
                return ()
            return tuple(
                event for event in events
                if (after is None or event.bar.timestamp > after)
                and event.available_at <= as_of
                and event.bar.timestamp < until
            )[:1]

    clock_values = iter((events[0].received_at, events[0].received_at + timedelta(minutes=20)))
    with SQLiteRepository(tmp_path / "poll.sqlite") as repository:
        runner = open_runner(repository, provider)
        cycles = runner.run_polling(
            PollFeed(),
            until=events[-1].available_at,
            clock=lambda: next(clock_values),
            poll_seconds=0,
            max_cycles=2,
            sleep=lambda _: None,
        )
        assert cycles == 2
        assert runner.status().feed_stale
        assert repository.events(event_type="STREAM_HEALTH")
