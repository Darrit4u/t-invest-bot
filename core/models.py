"""Shared domain models for Stage 2 signal processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.instrument_registry import InstrumentMeta
from core.market_data import Candle


class MarketRegime(str, Enum):
    """Market state used by strategy routing."""

    TREND = "TREND"
    COMPRESSION = "COMPRESSION"
    BALANCE = "BALANCE"
    NEUTRAL = "NEUTRAL"


class SignalDirection(str, Enum):
    """Signal side."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Latest indicator values for one instrument/timeframe stream."""

    timestamp: datetime
    close: float
    vwap: float
    ema_fast: float
    ema_slow: float
    atr: float
    rolling_volume_avg: float
    vwap_slope: float
    ema_fast_slope: float
    ema_slow_slope: float
    ema_distance: float
    crossing_count: int
    range_width: float
    overlap_ratio: float
    swing_high: float
    swing_low: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Structured signal returned by strategies and accepted by the pipeline."""

    signal_id: str
    instrument: str
    strategy: str
    regime: MarketRegime
    direction: SignalDirection
    timestamp: datetime
    entry_mode: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Input context supplied to each strategy."""

    instrument: InstrumentMeta
    timeframe: str
    candles: list[Candle]
    indicators: IndicatorSnapshot
    regime: MarketRegime
    session_active: bool
    blackout_active: bool
    blackout_reason: str | None
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Centralized filter output."""

    accepted: bool
    reason: str
