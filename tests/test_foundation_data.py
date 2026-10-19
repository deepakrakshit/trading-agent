"""Provider/schema boundary tests; also runnable with stdlib unittest."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trading_agent.data import (
    CSVBarProvider,
    DataValidationError,
    InMemoryBarProvider,
    MarketBar,
    MarketDataProvider,
    SyntheticBarProvider,
    synthetic_bars,
    validate_bars,
)

START = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)


class MarketDataTests(unittest.TestCase):
    def test_schema_rejects_missing_and_impossible_values(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "missing.*volume"):
            MarketBar.from_mapping({"symbol": "ABC", "timestamp": START, "open": 10, "high": 11, "low": 9, "close": 10})
        for field, value in (("open", 0), ("open", "1e1000"), ("open", "1e-1000"), ("high", 8), ("low", 12), ("close", "NaN"), ("volume", -1), ("volume", "1e1000"), ("complete", False)):
            with self.subTest(field=field), self.assertRaises(DataValidationError):
                values = dict(symbol="ABC", timestamp=START, open=10, high=11, low=9, close=10, volume=100)
                values[field] = value
                MarketBar(**values)

    def test_timestamps_are_aware_and_normalised_to_utc(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "timezone-aware"):
            synthetic_bars("ABC", START.replace(tzinfo=None), [10])
        bar = MarketBar.from_mapping({"symbol": "ABC", "timestamp": "2026-01-05T09:30:00+05:30", "open": "10", "high": "11", "low": "9", "close": "10", "volume": "100"})
        self.assertEqual(bar.timestamp, START)
        self.assertEqual(MarketBar.from_mapping(bar.to_mapping()), bar)

    def test_duplicates_out_of_order_gaps_and_future_data_fail(self) -> None:
        bars = synthetic_bars("ABC", START, [10, 11, 12])
        for invalid, kwargs, message in (
            ((bars[0], bars[0]), {}, "duplicate"),
            ((bars[1], bars[0]), {}, "out-of-order"),
            ((bars[0], bars[2]), {"expected_interval": timedelta(minutes=5)}, "missing/irregular"),
            (bars, {"as_of": bars[1].timestamp}, "future/unavailable"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(DataValidationError, message):
                validate_bars(invalid, **kwargs)

    def test_stale_and_missing_feeds_fail_with_injected_clock(self) -> None:
        bars = synthetic_bars("ABC", START, [10])
        self.assertEqual(validate_bars(bars, stale_after=timedelta(minutes=5), as_of=START + timedelta(minutes=5)), bars)
        with self.assertRaisesRegex(DataValidationError, "stale feed"):
            validate_bars(bars, stale_after=timedelta(minutes=5), as_of=START + timedelta(minutes=6))
        with self.assertRaisesRegex(DataValidationError, "missing market feed"):
            validate_bars((), stale_after=timedelta(minutes=5), as_of=START)

    def test_read_only_provider_labels_and_filters_fixture(self) -> None:
        bars = synthetic_bars("ABC", START, [10, 11, 12])
        provider = SyntheticBarProvider(bars)
        self.assertIsInstance(provider, MarketDataProvider)
        self.assertEqual(provider.provenance, "SYNTHETIC")
        self.assertEqual(provider.get_bars("ABC", bars[1].timestamp, bars[2].timestamp), (bars[1],))
        with self.assertRaisesRegex(ValueError, "provenance"):
            InMemoryBarProvider(bars, provenance="HISTORICAL")
        zero_volume = synthetic_bars("ABC", START, [10], volume=0)
        self.assertEqual(validate_bars(zero_volume), zero_volume)
        with self.assertRaisesRegex(DataValidationError, "volume"):
            validate_bars(zero_volume, require_positive_volume=True)

    def test_csv_provider_round_trip_and_duplicate_rejection(self) -> None:
        with TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "bars.csv"
            row = "ABC,2026-01-05T09:30:00+05:30,10,11,9,10,100\n"
            header = "symbol,timestamp,open,high,low,close,volume\n"
            path.write_text(header + row, encoding="utf-8")
            provider = CSVBarProvider(path)
            self.assertEqual(provider.provenance, "HISTORICAL")
            self.assertEqual(provider.all_bars()[0].timestamp, START)
            path.write_text(header + row + row, encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "duplicate"):
                CSVBarProvider(path)
