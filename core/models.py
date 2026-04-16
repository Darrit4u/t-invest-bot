"""Backward-compatible exports for shared domain models."""

from __future__ import annotations

from domain.models import (
    IndicatorSnapshot,
    Instrument as InstrumentMeta,
    MarketRegime,
    MarketRegimeState,
    Portfolio,
    Position,
    Signal as StrategySignal,
    SignalDecision,
    SignalDirection,
    StrategyContext,
    Trade,
)

__all__ = [
    "IndicatorSnapshot",
    "InstrumentMeta",
    "MarketRegime",
    "MarketRegimeState",
    "Portfolio",
    "Position",
    "SignalDecision",
    "SignalDirection",
    "StrategyContext",
    "StrategySignal",
    "Trade",
]
