from datetime import datetime, time, timedelta

import pytest

from trading_agent.data import DataValidationError, synthetic_bars
from trading_agent.data.calendar import KOLKATA, SessionCalendar, SessionHours


def local(day, value):
    return datetime.fromisoformat(f"{day}T{value}").replace(tzinfo=KOLKATA)


def test_weekday_session_bounds_and_timezone_edges():
    calendar = SessionCalendar()
    opening, closing = calendar.session_bounds("2026-01-05")
    assert opening.isoformat() == "2026-01-05T03:45:00+00:00"
    assert closing.isoformat() == "2026-01-05T10:00:00+00:00"
    assert calendar.is_open(local("2026-01-05", "09:15"))
    assert calendar.is_open(local("2026-01-05", "15:29:59"))
    assert not calendar.is_open(local("2026-01-05", "09:14:59"))
    assert not calendar.is_open(local("2026-01-05", "15:30"))
    assert not calendar.is_open(local("2026-01-04", "10:00"))
    with pytest.raises(DataValidationError, match="timezone-aware"):
        calendar.is_open(datetime(2026, 1, 5, 10))


def test_explicit_holidays_and_exceptional_sessions_change_fingerprint():
    calendar = SessionCalendar(holidays=["2026-01-26"])
    assert calendar.session_bounds("2026-01-26") is None
    # No inference that a weekday is an official exchange holiday.
    assert SessionCalendar().session_bounds("2026-01-26") is not None
    special = SessionCalendar(holidays=["2026-01-26"], overrides={"2026-01-03": SessionHours(time(10), time(11)), "2026-01-26": None})
    assert special.is_open(local("2026-01-03", "10:05"))
    assert special.session_bounds("2026-01-26") is None
    assert special.version != calendar.version
    assert calendar.version == SessionCalendar(holidays=["2026-01-26"]).version
    with pytest.raises(TypeError):
        special.overrides[local("2026-01-03", "10:00").date()] = None


def test_completed_bar_starts_use_as_of_and_exclude_close():
    calendar = SessionCalendar()
    starts = calendar.expected_bar_starts(local("2026-01-05", "09:15"), local("2026-01-05", "10:00"), as_of=local("2026-01-05", "09:27"))
    assert starts == tuple(local("2026-01-05", value).astimezone(starts[0].tzinfo) for value in ("09:15", "09:20"))
    whole_day = calendar.expected_bar_starts(local("2026-01-05", "09:15"), local("2026-01-05", "15:35"))
    assert len(whole_day) == 75
    assert whole_day[-1] == local("2026-01-05", "15:25")
    assert calendar.expected_bar_starts(local("2026-01-05", "09:15"), local("2026-01-05", "10:00"), as_of=local("2026-01-05", "09:19:59")) == ()


def test_overnight_weekend_and_declared_holiday_are_legitimate_gaps():
    calendar = SessionCalendar(holidays=["2026-01-26"])
    start, end = local("2026-01-23", "15:20"), local("2026-01-27", "09:25")
    times = [local("2026-01-23", value) for value in ("15:20", "15:25")] + [local("2026-01-27", value) for value in ("09:15", "09:20")]
    bars = tuple(synthetic_bars("ABC", timestamp, [100])[0] for timestamp in times)
    assert calendar.validate_intraday_bars(bars, "ABC", start=start, end=end, as_of=end) == bars


@pytest.mark.parametrize("indices", [(), (1, 2), (0, 2), (0, 1)])
def test_empty_and_leading_interior_trailing_gaps_fail_closed(indices):
    calendar = SessionCalendar()
    start, end = local("2026-01-05", "09:15"), local("2026-01-05", "09:30")
    bars = synthetic_bars("ABC", start, [100, 101, 102])
    with pytest.raises(DataValidationError, match="missing intraday"):
        calendar.validate_intraday_bars(tuple(bars[index] for index in indices), "ABC", start=start, end=end, as_of=end)


def test_off_grid_forming_wrong_symbol_and_duplicates_are_rejected():
    calendar = SessionCalendar()
    start, end = local("2026-01-05", "09:15"), local("2026-01-05", "09:30")
    bars = synthetic_bars("ABC", start, [100, 101, 102])
    for invalid in (bars + (bars[-1],), tuple(reversed(bars)), synthetic_bars("OTHER", start, [100, 101, 102]), synthetic_bars("ABC", start + timedelta(minutes=1), [100, 101])):
        with pytest.raises(DataValidationError):
            calendar.validate_intraday_bars(invalid, "ABC", start=start, end=end, as_of=end)
    with pytest.raises(DataValidationError, match="forming"):
        calendar.validate_intraday_bars(bars, "ABC", start=start, end=end, as_of=end - timedelta(minutes=1))


def test_non_grid_request_start_does_not_demand_an_unrequested_bar():
    calendar = SessionCalendar()
    expected = calendar.expected_bar_starts(local("2026-01-05", "09:16"), local("2026-01-05", "09:26"), as_of=local("2026-01-05", "09:30"))
    assert expected == tuple(local("2026-01-05", value) for value in ("09:20", "09:25"))
