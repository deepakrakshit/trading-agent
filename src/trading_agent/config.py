"""Versioned research configuration; never mutable by the numerical policy."""

from __future__ import annotations

from datetime import date, time, timedelta
import json
from pathlib import Path
from typing import Any, Mapping

from .paper_env import PaperTradingEnv
from .portfolio import CostModel
from .risk import RiskLimits


def load_config(path: str | Path = "configs/default.json") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not config.get("version"):
        raise ValueError("configuration must be a versioned JSON object")
    # Instantiate guards now: invalid limits/costs fail before any account opens.
    cost_model(config)
    risk_limits(config, "CONFIG_VALIDATION")
    return config


def cost_model(config: Mapping[str, Any]) -> CostModel:
    return CostModel(**config["costs"])


def risk_limits(config: Mapping[str, Any], symbol: str) -> RiskLimits:
    values = dict(config["risk"])
    values["stale_after"] = timedelta(seconds=values.pop("stale_seconds"))
    values["min_order_interval"] = timedelta(seconds=values.pop("min_order_seconds"))
    values["market_open"] = time.fromisoformat(values["market_open"])
    values["market_close"] = time.fromisoformat(values["market_close"])
    values["closed_dates"] = frozenset(date.fromisoformat(day) for day in values["closed_dates"])
    return RiskLimits(**values, permitted_symbols=frozenset({symbol}))


def make_env(bars, config: Mapping[str, Any], *, training=False, **kwargs) -> PaperTradingEnv:
    options: dict[str, Any] = {}
    if training:
        options.update(
            episode_length=min(config["training"]["episode_length"], len(bars) - 1),
            random_start=True, random_start_date=True, initial_cash_range=(5000, 10000),
        )
    options.update(kwargs)
    return PaperTradingEnv(
        bars, initial_cash=config["initial_cash"], cost_model=cost_model(config),
        risk_limits=risk_limits(config, bars[0].symbol),
        target_exposures=config["target_exposures"],
        participation_limit=config["participation_limit"],
        config_version=config["version"], **options,
    )
