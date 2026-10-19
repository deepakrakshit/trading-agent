"""Observable paper-cost attribution, never speculative market explanations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from .data import MarketBar


def attribute_round_trips(records: Sequence[dict], bars: Sequence[MarketBar]) -> dict[str, Any]:
    """Group inventory-flat round trips and reconcile reference P&L minus costs.

    Partial reductions remain in the current inventory episode; no per-fill win
    label or invented causal news is produced. MFE/MAE are descriptive completed
    candle bounds, not claims that fills at those extrema were achievable.
    """
    quantity = 0
    gross = fees = execution = Decimal("0")
    entry_index = None
    initial_price = None
    closed = []
    fill_rows = []
    for record in records:
        for fill in record["fills"]:
            cost = fill["cost"]
            reference = Decimal(cost["reference_notional"])
            impact = Decimal(cost["execution_cost"])
            paid = Decimal(cost["fees_total"])
            if quantity == 0:
                entry_index = record["next_index"]
                initial_price = Decimal(cost["reference_price"])
            buying = fill["side"] == "BUY"
            quantity += fill["quantity"] if buying else -fill["quantity"]
            gross += -reference if buying else reference
            fees += paid
            execution += impact
            signal = bars[record["index"]].close
            fill_rows.append({
                "fill_id": fill["fill_id"], "policy_version": record["policy_version"],
                "side": fill["side"], "quantity": fill["quantity"],
                "reference_price": cost["reference_price"], "execution_price": cost["execution_price"],
                "gap_from_signal_close": str(Decimal(cost["reference_price"]) - signal),
                "execution_cost": str(impact), "fee_components": cost["fee_components"],
                "simulated": True,
            })
            if quantity == 0:
                path = bars[entry_index:record["next_index"] + 1]
                closed.append({
                    "entry_timestamp": bars[entry_index].timestamp.isoformat(),
                    "exit_timestamp": fill["occurred_at"],
                    "gross_reference_pnl": str(gross), "execution_effect": str(-execution),
                    "fees": str(fees), "net_pnl": str(gross - execution - fees),
                    "mfe_fraction": float(max(bar.high for bar in path) / initial_price - 1),
                    "mae_fraction": float(min(bar.low for bar in path) / initial_price - 1),
                    "holding_bars": record["next_index"] - entry_index,
                    "market_cause": "UNKNOWN: no trustworthy event/news feed supplied",
                })
                gross = fees = execution = Decimal("0")
    return {
        "closed_round_trips": closed, "fills": fill_rows, "open_quantity": quantity,
        "definition": "Inventory-flat round trips; counterfactual price extrema are not executable fills",
    }
