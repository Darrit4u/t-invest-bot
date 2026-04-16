"""Shared fixtures/helpers for unit and integration tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from core.config_loader import ConfigLoader, SessionRule
from core.instrument_registry import InstrumentMeta, InstrumentRegistry
from core.market_data import Candle
from core.models import (
    IndicatorSnapshot,
    MarketRegime,
    MarketRegimeState,
    SignalDirection,
    StrategyContext,
    StrategySignal,
)


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def load_registry() -> InstrumentRegistry:
    cfg = ConfigLoader(config_dir()).load()
    return InstrumentRegistry.from_config(cfg)


def dt_at(index: int, base: datetime | None = None) -> datetime:
    start = base or datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    return start + timedelta(minutes=index)


def make_candle(
    index: int,
    *,
    open_: float,
    close: float,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    instrument: str = "ES",
    timeframe: str = "1min",
    base: datetime | None = None,
) -> Candle:
    _high = high if high is not None else max(open_, close) + 0.2
    _low = low if low is not None else min(open_, close) - 0.2
    return Candle.validated(
        dt=dt_at(index, base=base),
        open_=open_,
        high=_high,
        low=_low,
        close=close,
        volume=volume,
        instrument=instrument,
        timeframe=timeframe,
    )


def build_signal(
    *,
    instrument: str = "ES",
    strategy: str = "trend_pullback_vwap_ema",
    regime: MarketRegime = MarketRegime.TREND,
    direction: SignalDirection = SignalDirection.LONG,
    timestamp: datetime | None = None,
    entry: float = 100.0,
    stop_loss: float = 99.0,
    tp1: float = 101.0,
    tp2: float = 102.0,
    entry_mode: str = "NEXT_BAR_OPEN",
) -> StrategySignal:
    return StrategySignal(
        signal_id=str(uuid4()),
        instrument=instrument,
        strategy=strategy,
        regime=regime,
        direction=direction,
        timestamp=timestamp or dt_at(0),
        entry_mode=entry_mode,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        metadata={"source": "test"},
    )


def build_indicator(
    *,
    timestamp: datetime | None = None,
    close: float = 100.0,
    vwap: float = 99.5,
    ema_fast: float = 100.2,
    ema_slow: float = 99.9,
    atr: float = 1.0,
    rolling_volume_avg: float = 1000.0,
    vwap_slope: float = 0.05,
    ema_fast_slope: float = 0.04,
    ema_slow_slope: float = 0.03,
    ema_distance: float = 0.3,
    crossing_count: int = 2,
    range_width: float = 1.8,
    overlap_ratio: float = 0.65,
    swing_high: float = 101.0,
    swing_low: float = 98.0,
    extra: dict | None = None,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=timestamp or dt_at(0),
        close=close,
        vwap=vwap,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr=atr,
        rolling_volume_avg=rolling_volume_avg,
        vwap_slope=vwap_slope,
        ema_fast_slope=ema_fast_slope,
        ema_slow_slope=ema_slow_slope,
        ema_distance=ema_distance,
        crossing_count=crossing_count,
        range_width=range_width,
        overlap_ratio=overlap_ratio,
        swing_high=swing_high,
        swing_low=swing_low,
        extra=dict(extra or {}),
    )


def build_instrument_meta(
    *,
    symbol: str = "ES",
    enabled: bool = True,
    strategies: tuple[str, ...] = (
        "trend_pullback_vwap_ema",
        "compression_breakout",
        "liquidity_sweep_reversal",
    ),
) -> InstrumentMeta:
    session = SessionRule(name="US_DAY", start="09:30", end="16:00", timezone="America/New_York")
    return InstrumentMeta(
        symbol=symbol,
        enabled=enabled,
        uid=None,
        figi=None,
        ticker=symbol,
        class_code="SPBFUT",
        tick_size=0.25,
        tick_value=12.5,
        lot=1,
        sessions=(session,),
        allowed_strategies=strategies,
    )


def build_context(
    *,
    candles: list[Candle],
    regime: MarketRegime,
    instrument: InstrumentMeta | None = None,
    indicators: IndicatorSnapshot | None = None,
    session_active: bool = True,
    blackout_active: bool = False,
    params: dict | None = None,
    regime_state: MarketRegimeState | None = None,
) -> StrategyContext:
    meta = instrument or build_instrument_meta(symbol=candles[-1].instrument)
    snapshot = indicators or build_indicator(timestamp=candles[-1].datetime)
    return StrategyContext(
        instrument=meta,
        timeframe=candles[-1].timeframe,
        candles=candles,
        indicators=snapshot,
        regime=regime,
        session_active=session_active,
        blackout_active=blackout_active,
        blackout_reason="test" if blackout_active else None,
        params=params or {},
        regime_state=regime_state,
    )


def build_trend_sequence() -> list[Candle]:
    """Deterministic bullish swing pullback confirmation with enough bars for EMA200."""
    candles: list[Candle] = []
    price = 100.0
    for i in range(240):
        if i < 220:
            open_, close = price, price + 0.12
        elif i < 233:
            open_, close = price, price + 0.24
        elif i < 237:
            open_, close = price, price - 0.45
        elif i < 239:
            open_, close = price, price - 0.15
        else:
            open_, close = price, price + 0.95
        candle = make_candle(i, open_=open_, close=close, high=max(open_, close) + 0.3, low=min(open_, close) - 0.3, volume=1200)
        candles.append(candle)
        price = close
    return candles


def build_bear_trend_sequence() -> list[Candle]:
    """Deterministic bearish swing pullback confirmation with enough bars for EMA200."""
    candles: list[Candle] = []
    price = 140.0
    for i in range(240):
        if i < 220:
            open_, close = price, price - 0.12
        elif i < 233:
            open_, close = price, price - 0.24
        elif i < 237:
            open_, close = price, price + 0.45
        elif i < 239:
            open_, close = price, price + 0.15
        else:
            open_, close = price, price - 0.95
        candle = make_candle(i, open_=open_, close=close, high=max(open_, close) + 0.3, low=min(open_, close) - 0.3, volume=1200)
        candles.append(candle)
        price = close
    return candles
