"""Canonical domain entities used across trading modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
class Instrument:
    """Resolved instrument metadata and strategy permissions."""

    symbol: str
    enabled: bool
    uid: str | None
    figi: str | None
    ticker: str
    class_code: str | None
    tick_size: float
    tick_value: float
    lot: int
    sessions: tuple[Any, ...]
    allowed_strategies: tuple[str, ...]


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
class Signal:
    """Strategy output separated from execution concerns."""

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
    timeframe: str | None = None
    confidence: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def strategy_id(self) -> str:
        return self.strategy

    @property
    def side(self) -> SignalDirection:
        return self.direction

    @property
    def entry_price(self) -> float:
        return self.entry

    @property
    def take_profit(self) -> float:
        return self.tp1


StrategySignal = Signal


@dataclass(frozen=True, slots=True)
class Position:
    """Opened position snapshot."""

    instrument: str
    side: SignalDirection
    entry_price: float
    size: float
    opened_at: datetime
    stop_loss: float | None
    take_profit: float | None
    strategy_id: str
    status: str
    position_id: str | None = None
    signal_id: str | None = None
    timeframe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> SignalDirection:
        return self.side


@dataclass(frozen=True, slots=True)
class Trade:
    """Completed or tracked trade lifecycle snapshot."""

    instrument: str
    side: SignalDirection
    entry_price: float
    exit_price: float | None
    size: float
    opened_at: datetime
    closed_at: datetime | None
    pnl: float
    pnl_pct: float | None
    strategy_id: str
    trade_id: str | None = None
    signal_id: str | None = None
    status: str | None = None
    timeframe: str | None = None
    created_at: datetime | None = None
    activated_at: datetime | None = None
    gross_pnl: float | None = None
    fees_paid: float | None = None
    r_multiple: float | None = None
    exit_reason: str | None = None
    entry_fill_price: float | None = None
    remaining_qty: float | None = None
    bars_waiting: int | None = None
    bars_in_trade: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> SignalDirection:
        return self.side


@dataclass(frozen=True, slots=True)
class Portfolio:
    """Minimal portfolio placeholder for future allocation/risk phases."""

    portfolio_id: str = "default"
    positions: tuple[Position, ...] = field(default_factory=tuple)
    trades: tuple[Trade, ...] = field(default_factory=tuple)
    cash: float = 0.0
    equity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Input context supplied to each strategy."""

    instrument: Instrument
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
