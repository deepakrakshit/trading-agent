"""Deterministic fee and adverse execution-cost calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Literal

from ..data.bars import decimal_value

Side = Literal["BUY", "SELL"]
_BPS = Decimal("10000")
PAISE = Decimal("0.01")


def money(value: Decimal | str | float) -> Decimal:
    """Round a monetary amount to integer paise using commercial rounding."""

    return decimal_value(value, "money").quantize(PAISE, rounding=ROUND_HALF_UP)


def normalise_side(side: str) -> Side:
    """Return a canonical side or reject unsupported order directions."""

    value = side.upper() if isinstance(side, str) else ""
    if value not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """One complete cost calculation for one simulated fill.

    Slippage and half-spread are represented once in ``execution_price`` and
    ``execution_cost``.  Fee components contain charges only; no fee component
    includes slippage again.
    """

    side: Side
    quantity: int
    reference_price: Decimal
    execution_price: Decimal
    reference_notional: Decimal
    execution_notional: Decimal
    execution_cost: Decimal
    fee_components: tuple[tuple[str, Decimal], ...]
    cash_effect: Decimal

    @property
    def fees_total(self) -> Decimal:
        """Total explicit fees and taxes, excluding execution impact."""

        return sum((amount for _, amount in self.fee_components), Decimal("0"))

    @property
    def total_cost(self) -> Decimal:
        """Total cost relative to an unadjusted reference fill."""

        return self.execution_cost + self.fees_total

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for an event record."""

        return {
            "side": self.side,
            "quantity": self.quantity,
            "reference_price": str(self.reference_price),
            "execution_price": str(self.execution_price),
            "reference_notional": str(self.reference_notional),
            "execution_notional": str(self.execution_notional),
            "execution_cost": str(self.execution_cost),
            "fee_components": {name: str(amount) for name, amount in self.fee_components},
            "fees_total": str(self.fees_total),
            "total_cost": str(self.total_cost),
            "cash_effect": str(self.cash_effect),
        }


@dataclass(frozen=True, slots=True)
class CostModel:
    """Versioned, cash-equity cost schedule for a paper-only fill.

    Rates are decimal fractions (``0.001`` is 10 basis points).  ``spread_bps``
    is the full spread, so each side pays half of it.  The model intentionally
    uses one adverse execution price and does not add a second slippage fee.
    """

    version: str = "paper-v1"
    effective_date: date | None = None
    brokerage_rate: Decimal | str | float = Decimal("0")
    transaction_fee_rate: Decimal | str | float = Decimal("0")
    buy_tax_rate: Decimal | str | float = Decimal("0")
    sell_tax_rate: Decimal | str | float = Decimal("0")
    gst_rate: Decimal | str | float = Decimal("0")
    stamp_duty_buy_rate: Decimal | str | float = Decimal("0")
    spread_bps: Decimal | str | float = Decimal("0")
    slippage_bps: Decimal | str | float = Decimal("0")
    price_tick: Decimal | str | float = PAISE

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("cost-model version must be non-empty")
        for field_name in (
            "brokerage_rate",
            "transaction_fee_rate",
            "buy_tax_rate",
            "sell_tax_rate",
            "gst_rate",
            "stamp_duty_buy_rate",
            "spread_bps",
            "slippage_bps",
            "price_tick",
        ):
            value = decimal_value(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "version", self.version.strip())
        if self.price_tick <= 0:
            raise ValueError("price_tick must be positive")
        if self.slippage_bps + self.spread_bps / 2 >= _BPS:
            raise ValueError("spread/slippage must leave a positive sell price")

    def estimate(self, side: str, quantity: int, reference_price: Decimal | str | float) -> CostBreakdown:
        """Calculate one deterministic fill without mutating account state."""

        canonical_side = normalise_side(side)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        reference = decimal_value(reference_price, "reference_price")
        if reference <= 0:
            raise ValueError("reference_price must be positive")

        adverse_bps = self.slippage_bps + self.spread_bps / Decimal("2")
        impact_fraction = adverse_bps / _BPS
        if canonical_side == "BUY":
            execution_price = reference * (Decimal("1") + impact_fraction)
            rounding = ROUND_CEILING
        else:
            execution_price = reference * (Decimal("1") - impact_fraction)
            rounding = ROUND_FLOOR
        execution_price = (
            (execution_price / self.price_tick).to_integral_value(rounding=rounding)
            * self.price_tick
        )
        if execution_price <= 0:
            raise ValueError("rounded execution price must be positive")
        reference_notional = money(reference * quantity)
        execution_notional = money(execution_price * quantity)
        execution_cost = abs(execution_notional - reference_notional)

        rate = self.buy_tax_rate if canonical_side == "BUY" else self.sell_tax_rate
        stamp = self.stamp_duty_buy_rate if canonical_side == "BUY" else Decimal("0")
        brokerage = money(execution_notional * self.brokerage_rate)
        transaction = money(execution_notional * self.transaction_fee_rate)
        tax = money(execution_notional * rate)
        gst = money((brokerage + transaction) * self.gst_rate)
        stamp_duty = money(execution_notional * stamp)
        components = (
            ("brokerage", brokerage),
            ("transaction_fee", transaction),
            ("tax", tax),
            ("gst", gst),
            ("stamp_duty", stamp_duty),
        )
        fees_total = sum((amount for _, amount in components), Decimal("0"))
        cash_effect = (
            -(execution_notional + fees_total)
            if canonical_side == "BUY"
            else execution_notional - fees_total
        )
        return CostBreakdown(
            side=canonical_side,
            quantity=quantity,
            reference_price=reference,
            execution_price=execution_price,
            reference_notional=reference_notional,
            execution_notional=execution_notional,
            execution_cost=execution_cost,
            fee_components=components,
            cash_effect=cash_effect,
        )
