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
class MarketRegimeState:
    """Score-based regime state with dominant label and reason codes."""

    dominant: MarketRegime
    trend_score: float
    compression_score: float
    balance_score: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def score_for(self, regime: MarketRegime) -> float:
        if regime == MarketRegime.TREND:
            return float(self.trend_score)
        if regime == MarketRegime.COMPRESSION:
            return float(self.compression_score)
        if regime == MarketRegime.BALANCE:
            return float(self.balance_score)
        return 0.0

    def score_for_strategy(self, strategy: str) -> float:
        normalized = strategy.strip().lower()
        if normalized == "trend_pullback_vwap_ema":
            return float(self.trend_score)
        if normalized == "compression_breakout":
            return float(self.compression_score)
        if normalized == "liquidity_sweep_reversal":
            return float(self.balance_score)
        return 0.0


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
    regime_state: MarketRegimeState | None = None


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Centralized filter output."""

    accepted: bool
    reason: str
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    signal_quality_score: float = 0.0
    expected_fill_price: float | None = None
    post_fill_rr: float | None = None
    expected_edge_after_fees: float | None = None
    enriched_metadata: dict[str, Any] = field(default_factory=dict)
