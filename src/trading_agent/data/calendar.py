"""Explicit NSE-style weekday sessions, not an official exchange calendar.

The default is Asia/Kolkata Monday-Friday, 09:15 inclusive to 15:30 exclusive.
Supply authoritative holidays and any exceptional sessions yourself.  Candle
timestamps identify their opening instant; a five-minute candle is available
only once its entire interval has completed.  Missing scheduled candles are a
data error, never a request to forward-fill or manufacture observations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
from types import MappingProxyType
from zoneinfo import ZoneInfo

from .bars import DataValidationError, MarketBar, utc_timestamp
from .validation import validate_bars

KOLKATA = ZoneInfo("Asia/Kolkata")
FIVE_MINUTES = timedelta(minutes=5)


def _date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise ValueError("calendar dates must be dates, not datetimes")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("calendar dates must be dates or ISO date strings")


@dataclass(frozen=True, slots=True)
class SessionHours:
    open: time = time(9, 15)
    close: time = time(15, 30)

    def __post_init__(self) -> None:
        if not isinstance(self.open, time) or not isinstance(self.close, time):
            raise ValueError("session hours must be times")
        if self.open.tzinfo is not None or self.close.tzinfo is not None or self.open >= self.close:
            raise ValueError("session hours must be increasing, timezone-free local times")
        if any(value.second or value.microsecond for value in (self.open, self.close)):
            raise ValueError("session hours must use whole minutes")


class SessionCalendar:
    """Weekday schedule with immutable, explicit date overrides.

    ``holidays`` closes specified dates. ``overrides`` takes precedence: a
    ``SessionHours`` value can open an exceptional weekend/short session, and
    ``None`` explicitly closes a date.  No holidays are guessed or downloaded.
    """

    timezone = KOLKATA

    def __init__(
        self,
        *,
        holidays: Iterable[date | str] = (),
        overrides: Mapping[date | str, SessionHours | None] | None = None,
        hours: SessionHours = SessionHours(),
    ) -> None:
        if not isinstance(hours, SessionHours):
            raise ValueError("hours must be SessionHours")
        self.hours = hours
        self.holidays = frozenset(_date(day) for day in holidays)
        configured = {_date(day): value for day, value in (overrides or {}).items()}
        if any(value is not None and not isinstance(value, SessionHours) for value in configured.values()):
            raise ValueError("session overrides must be SessionHours or None")
        self.overrides = MappingProxyType(configured)

    @property
    def version(self) -> str:
        """Fingerprint the complete local schedule for source manifests."""

        def encoded(hours: SessionHours | None):
            return None if hours is None else [hours.open.isoformat(), hours.close.isoformat()]

        payload = {
            "timezone": "Asia/Kolkata", "weekdays": [0, 1, 2, 3, 4],
            "hours": encoded(self.hours), "holidays": sorted(day.isoformat() for day in self.holidays),
            "overrides": {day.isoformat(): encoded(value) for day, value in sorted(self.overrides.items())},
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"weekday-ist-v1:{digest}"

    def session_bounds(self, day: date | str) -> tuple[datetime, datetime] | None:
        """Return UTC session bounds, or None for a closed local date."""

        day = _date(day)
        if day in self.overrides:
            hours = self.overrides[day]
        elif day.weekday() >= 5 or day in self.holidays:
            hours = None
        else:
            hours = self.hours
        if hours is None:
            return None
        return (
            utc_timestamp(datetime.combine(day, hours.open, self.timezone)),
            utc_timestamp(datetime.combine(day, hours.close, self.timezone)),
        )

    def is_open(self, timestamp: datetime) -> bool:
        instant = utc_timestamp(timestamp)
        bounds = self.session_bounds(instant.astimezone(self.timezone).date())
        return bounds is not None and bounds[0] <= instant < bounds[1]

    def expected_bar_starts(
        self,
        start: datetime,
        end: datetime,
        *,
        as_of: datetime | None = None,
        interval: timedelta = FIVE_MINUTES,
    ) -> tuple[datetime, ...]:
        """Scheduled starts in [start, end), excluding forming/session-tail bars."""

        start, end = utc_timestamp(start), utc_timestamp(end)
        available = utc_timestamp(as_of) if as_of is not None else None
        if end <= start or not isinstance(interval, timedelta) or interval <= timedelta(0):
            raise ValueError("end must follow start and interval must be positive")
        first = start.astimezone(self.timezone).date()
        last = (min(end, available) if available is not None else end).astimezone(self.timezone).date()
        result = []
        day = first
        while day <= last:
            bounds = self.session_bounds(day)
            if bounds is not None:
                opening, closing = bounds
                # Align to the session grid, including a non-grid request start.
                skip = max(0, (start - opening) // interval)
                cursor = opening + skip * interval
                if cursor < start:
                    cursor += interval
                while cursor < end and cursor + interval <= closing:
                    if available is not None and cursor + interval > available:
                        break
                    result.append(cursor)
                    cursor += interval
            day += timedelta(days=1)
        return tuple(result)

    def validate_intraday_bars(
        self,
        bars: Iterable[MarketBar],
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        as_of: datetime,
        interval: timedelta = FIVE_MINUTES,
    ) -> tuple[MarketBar, ...]:
        """Fail closed on missing leading, interior, trailing or empty data.

        Closed sessions are legitimate gaps.  Duplicate, out-of-session,
        off-grid, forming, future, or wrong-symbol bars are rejected as well.
        Input order is preserved and is never silently repaired.
        """

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be explicit and non-empty")
        expected = self.expected_bar_starts(start, end, as_of=as_of, interval=interval)
        materialised = validate_bars(bars, as_of=as_of)
        if any(bar.symbol != symbol for bar in materialised):
            raise DataValidationError("unexpected instrument in intraday data")
        actual = tuple(bar.timestamp for bar in materialised)
        expected_set, actual_set = set(expected), set(actual)
        if actual_set - expected_set:
            raise DataValidationError("unexpected/off-session/forming intraday bars")
        if expected_set - actual_set:
            raise DataValidationError("missing intraday bars; feed is incomplete")
        if actual != expected:
            raise DataValidationError("intraday bars are not in scheduled order")
        return materialised


__all__ = ["FIVE_MINUTES", "KOLKATA", "SessionCalendar", "SessionHours"]
