"""Offline, paper-only trading primitives.

The package deliberately contains no broker order client and no credential
handling.  Market data enters through the read-only provider protocol and all
portfolio-changing actions terminate at :class:`PaperBroker`.
"""

from .data import (
    CSVBarProvider,
    DataProvenance,
    DataValidationError,
    InMemoryBarProvider,
    MarketBar,
    MarketDataProvider,
    SyntheticBarProvider,
    synthetic_bars,
    validate_bars,
)
from .execution import ExecutionReceipt, OrderRequest, PaperBroker
from .features import CausalFeatureBuilder, FeatureRow
from .portfolio import (
    AccountSnapshot,
    CostBreakdown,
    CostModel,
    LedgerFill,
    PortfolioLedger,
    Position,
)
from .risk import RiskContext, RiskDecision, RiskLimits, RiskSupervisor
from .storage import SQLiteRepository

__all__ = [
    "AccountSnapshot",
    "CausalFeatureBuilder",
    "CostBreakdown",
    "CostModel",
    "CSVBarProvider",
    "DataProvenance",
    "DataValidationError",
    "ExecutionReceipt",
    "FeatureRow",
    "InMemoryBarProvider",
    "LedgerFill",
    "MarketBar",
    "MarketDataProvider",
    "OrderRequest",
    "PaperBroker",
    "PortfolioLedger",
    "Position",
    "RiskContext",
    "RiskDecision",
    "RiskLimits",
    "RiskSupervisor",
    "SQLiteRepository",
    "SyntheticBarProvider",
    "synthetic_bars",
    "validate_bars",
]
