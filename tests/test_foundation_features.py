"""No-look-ahead and cross-symbol feature tests."""

from datetime import datetime, timezone
from math import log
import unittest

from trading_agent.data import synthetic_bars
from trading_agent.features import CausalFeatureBuilder

START = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)


class CausalFeatureTests(unittest.TestCase):
    def test_future_rows_cannot_change_a_feature_prefix(self) -> None:
        builder = CausalFeatureBuilder(lookback=3)
        bars = synthetic_bars("ABC", START, [100, 110, 90, 120, 50, 1000])
        full = builder.transform(bars)
        for end in range(1, len(bars) + 1):
            self.assertEqual(builder.transform(bars[:end]), full[:end])
        changed_future = bars[:3] + synthetic_bars("ABC", bars[3].timestamp, [5000, 200, 1])
        self.assertEqual(builder.transform(changed_future)[:3], full[:3])

    def test_history_does_not_cross_symbols(self) -> None:
        abc = synthetic_bars("ABC", START, [100, 110, 121])
        xyz = synthetic_bars("XYZ", START, [1, 2, 4])
        mixed = tuple(item for pair in zip(abc, xyz) for item in pair)
        rows = CausalFeatureBuilder(2).transform(mixed)
        self.assertIsNone(rows[0].log_return)
        self.assertIsNone(rows[1].log_return)
        self.assertAlmostEqual(rows[2].log_return, log(1.1))
        self.assertAlmostEqual(rows[3].log_return, log(2))
        self.assertEqual(rows[2].volume_ratio, 1)
        self.assertAlmostEqual(rows[4].rolling_volatility, 0)

    def test_empty_and_zero_volume_have_explicit_warmup_values(self) -> None:
        builder = CausalFeatureBuilder()
        self.assertEqual(builder.transform(()), ())
        rows = builder.transform(synthetic_bars("ABC", START, [10, 10], volume=0))
        self.assertIsNone(rows[0].rolling_volatility)
        self.assertIsNone(rows[1].volume_ratio)
        self.assertEqual(rows[1].log_return, 0)
