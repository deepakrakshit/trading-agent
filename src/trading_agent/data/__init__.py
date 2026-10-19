"""Validated, read-only market-bar data boundaries."""

from .bars import DataProvenance, DataValidationError, MarketBar
from .providers import (
    CSVBarProvider,
    InMemoryBarProvider,
    MarketDataProvider,
    SyntheticBarProvider,
    synthetic_bars,
)
from .validation import validate_bars

__all__ = [
    "DataProvenance",
    "CSVBarProvider",
    "DataValidationError",
    "InMemoryBarProvider",
    "MarketBar",
    "MarketDataProvider",
    "SyntheticBarProvider",
    "synthetic_bars",
    "validate_bars",
]
