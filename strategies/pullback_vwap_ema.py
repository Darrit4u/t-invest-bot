"""Trend Pullback to VWAP/EMA strategy."""

from __future__ import annotations

from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy
from strategies.mtf import mtf_alignment


class TrendPullbackVWAPEMAStrategy(BaseStrategy):
    """Trend continuation entry after controlled pullback and confirmation."""

    name = "trend_pullback_vwap_ema"
    allowed_regime = MarketRegime.TREND

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        if context.regime != self.allowed_regime:
            return None

        candles = context.candles
        impulse_bars = self._int("impulse_bars", 3)
        min_required = impulse_bars + 2
        if len(candles) < min_required:
            return None

        atr = context.indicators.atr
        if atr <= 0:
            return None

        long_bias = (
            context.indicators.close > context.indicators.vwap
            and context.indicators.ema_fast > context.indicators.ema_slow
        )
        short_bias = (
            context.indicators.close < context.indicators.vwap
            and context.indicators.ema_fast < context.indicators.ema_slow
        )

        direction: SignalDirection | None
        if long_bias:
            direction = SignalDirection.LONG
        elif short_bias:
            direction = SignalDirection.SHORT
        else:
            return None

        mtf_ok, mtf_meta = mtf_alignment(
            enabled=self._bool("use_mtf_filter", False),
            candles=candles,
            source_timeframe=context.timeframe,
            direction=direction,
            trend_timeframe=self._str("trend_timeframe", "1hour"),
            setup_timeframe=self._str("setup_timeframe", "15min"),
            fast_ema=max(2, self._int("mtf_fast_ema", 8)),
            slow_ema=max(3, self._int("mtf_slow_ema", 21)),
            slope_bars=max(1, self._int("mtf_slope_bars", 2)),
        )
        if not mtf_ok:
            return None

        impulse = candles[-(impulse_bars + 2) : -2]
        pullback = candles[-2]
        confirm = candles[-1]
        impulse_vwap, pullback_vwap = _select_impulse_and_pullback_vwap(
            candles=candles,
            impulse_bars=impulse_bars,
            timezone_name=_strategy_timezone(context),
        )
        ema_fast_series = _ema_series(
            [item.close for item in candles],
            period=_indicator_ema_fast_period(context),
        )
        pullback_ema_fast = ema_fast_series[-2]

        if direction == SignalDirection.LONG:
            return self._evaluate_long(
                context=context,
                impulse=impulse,
                pullback=pullback,
                confirm=confirm,
                impulse_vwap=impulse_vwap,
                pullback_vwap=pullback_vwap,
                pullback_ema_fast=pullback_ema_fast,
                mtf_meta=mtf_meta,
            )
        return self._evaluate_short(
            context=context,
            impulse=impulse,
            pullback=pullback,
            confirm=confirm,
            impulse_vwap=impulse_vwap,
            pullback_vwap=pullback_vwap,
            pullback_ema_fast=pullback_ema_fast,
            mtf_meta=mtf_meta,
        )

    def _evaluate_long(
        self,
        context: StrategyContext,
        impulse: list,
        pullback,
        confirm,
        impulse_vwap: list[float],
        pullback_vwap: float,
        pullback_ema_fast: float,
        mtf_meta: dict[str, Any],
    ) -> StrategySignal | None:
        atr = context.indicators.atr
        impulse_atr_mult = self._float("impulse_atr_mult", 0.6)
        move = impulse[-1].close - impulse[0].open
        if move < impulse_atr_mult * atr:
            return None

        min_bullish = self._int("min_bullish_bars_in_impulse", 1)
        bullish_count = sum(1 for item in impulse if item.close > item.open)
        if bullish_count < min_bullish:
            return None

        avg_impulse_volume = mean(item.volume for item in impulse)
        volume_ratio = avg_impulse_volume / max(context.indicators.rolling_volume_avg, 1e-9)
        if volume_ratio < self._float("volume_impulse_mult", 0.9):
            return None

        max_extension = max(item.high - context.indicators.vwap for item in impulse)
        min_ext = self._float("min_vwap_extension_atr", 0.0) * atr
        max_ext = self._float("max_vwap_extension_atr", 3.0) * atr
        if not (min_ext <= max_extension <= max_ext):
            return None

        impulse_high = max(item.high for item in impulse)
        pullback_depth = impulse_high - pullback.low
        if pullback_depth < self._float("pullback_min_atr", 0.08) * atr:
            return None
        if pullback_depth > self._float("pullback_max_atr", 1.8) * atr:
            return None

        zone_mode = self._str("pullback_location_mode", "ANY")
        touched_vwap = pullback.low <= pullback_vwap <= pullback.high
        touched_ema = pullback.low <= pullback_ema_fast <= pullback.high
        if zone_mode == "VWAP_ONLY" and not touched_vwap:
            return None
        if zone_mode == "EMA_FAST_ONLY" and not touched_ema:
            return None
        if zone_mode in {"VWAP_OR_EMA_FAST", "VWAP_EMA_ZONE"} and not (touched_vwap or touched_ema):
            return None

        confirm_body = abs(confirm.close - confirm.open)
        if confirm_body < self._float("confirmation_body_min_atr", 0.05) * atr:
            return None

        if confirm.close <= confirm.open:
            return None
        if confirm.close <= pullback.close:
            return None
        confirmation_delta = self._float("confirmation_close_delta_atr", 0.02) * atr
        if (confirm.close - pullback.close) < confirmation_delta:
            return None
        confirmation_break_mode = self._str("confirmation_break_mode", "BODY").strip().upper()
        if confirmation_break_mode == "EXTREME" and confirm.close <= pullback.high:
            return None

        stop = pullback.low - self._float("stop_buffer_atr", 0.15) * atr
        entry = confirm.close
        risk = entry - stop
        if risk <= 0:
            return None

        tp1 = entry + self._float("tp1_r", 1.0) * risk
        tp2 = entry + self._float("tp2_r", 2.0) * risk
        return self.build_signal(
            context=context,
            direction=SignalDirection.LONG,
            entry_mode=self._str("entry_timing_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "impulse_size_atr": move / atr,
                "pullback_depth_atr": pullback_depth / atr,
                "volume_ratio": volume_ratio,
                "pullback_vwap": pullback_vwap,
                "pullback_ema_fast": pullback_ema_fast,
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
                **mtf_meta,
            },
        )

    def _evaluate_short(
        self,
        context: StrategyContext,
        impulse: list,
        pullback,
        confirm,
        impulse_vwap: list[float],
        pullback_vwap: float,
        pullback_ema_fast: float,
        mtf_meta: dict[str, Any],
    ) -> StrategySignal | None:
        atr = context.indicators.atr
        impulse_atr_mult = self._float("impulse_atr_mult", 0.6)
        move = impulse[0].open - impulse[-1].close
        if move < impulse_atr_mult * atr:
            return None

        min_bearish = self._int("min_bearish_bars_in_impulse", 1)
        bearish_count = sum(1 for item in impulse if item.close < item.open)
        if bearish_count < min_bearish:
            return None

        avg_impulse_volume = mean(item.volume for item in impulse)
        volume_ratio = avg_impulse_volume / max(context.indicators.rolling_volume_avg, 1e-9)
        if volume_ratio < self._float("volume_impulse_mult", 0.9):
            return None

        max_extension = max(context.indicators.vwap - item.low for item in impulse)
        min_ext = self._float("min_vwap_extension_atr", 0.0) * atr
        max_ext = self._float("max_vwap_extension_atr", 3.0) * atr
        if not (min_ext <= max_extension <= max_ext):
            return None

        impulse_low = min(item.low for item in impulse)
        pullback_depth = pullback.high - impulse_low
        if pullback_depth < self._float("pullback_min_atr", 0.08) * atr:
            return None
        if pullback_depth > self._float("pullback_max_atr", 1.8) * atr:
            return None

        zone_mode = self._str("pullback_location_mode", "ANY")
        touched_vwap = pullback.low <= pullback_vwap <= pullback.high
        touched_ema = pullback.low <= pullback_ema_fast <= pullback.high
        if zone_mode == "VWAP_ONLY" and not touched_vwap:
            return None
        if zone_mode == "EMA_FAST_ONLY" and not touched_ema:
            return None
        if zone_mode in {"VWAP_OR_EMA_FAST", "VWAP_EMA_ZONE"} and not (touched_vwap or touched_ema):
            return None

        confirm_body = abs(confirm.close - confirm.open)
        if confirm_body < self._float("confirmation_body_min_atr", 0.05) * atr:
            return None

        if confirm.close >= confirm.open:
            return None
        if confirm.close >= pullback.close:
            return None
        confirmation_delta = self._float("confirmation_close_delta_atr", 0.02) * atr
        if (pullback.close - confirm.close) < confirmation_delta:
            return None
        confirmation_break_mode = self._str("confirmation_break_mode", "BODY").strip().upper()
        if confirmation_break_mode == "EXTREME" and confirm.close >= pullback.low:
            return None

        stop = pullback.high + self._float("stop_buffer_atr", 0.15) * atr
        entry = confirm.close
        risk = stop - entry
        if risk <= 0:
            return None

        tp1 = entry - self._float("tp1_r", 1.0) * risk
        tp2 = entry - self._float("tp2_r", 2.0) * risk
        return self.build_signal(
            context=context,
            direction=SignalDirection.SHORT,
            entry_mode=self._str("entry_timing_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "impulse_size_atr": move / atr,
                "pullback_depth_atr": pullback_depth / atr,
                "volume_ratio": volume_ratio,
                "pullback_vwap": pullback_vwap,
                "pullback_ema_fast": pullback_ema_fast,
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
                **mtf_meta,
            },
        )


def _touched_zone_label(*, touched_vwap: bool, touched_ema: bool) -> str:
    if touched_vwap and touched_ema:
        return "VWAP_AND_EMA_FAST"
    if touched_vwap:
        return "VWAP"
    if touched_ema:
        return "EMA_FAST"
    return "NONE"


def _indicator_ema_fast_period(context: StrategyContext) -> int:
    indicator_cfg = context.params.get("indicator_engine", {})
    if not isinstance(indicator_cfg, dict):
        return 20
    return max(2, int(indicator_cfg.get("ema_fast", 20)))


def _strategy_timezone(context: StrategyContext) -> str:
    if context.instrument.sessions:
        return context.instrument.sessions[0].timezone
    return "UTC"


def _session_vwap_series(candles: list, *, timezone_name: str) -> list[float]:
    zone = ZoneInfo(timezone_name)
    result: list[float] = []

    current_session = None
    cumulative_tpv = 0.0
    cumulative_vol = 0.0

    for candle in candles:
        session_key = candle.datetime.astimezone(zone).date()
        if session_key != current_session:
            current_session = session_key
            cumulative_tpv = 0.0
            cumulative_vol = 0.0

        typical = (candle.high + candle.low + candle.close) / 3.0
        cumulative_tpv += typical * candle.volume
        cumulative_vol += candle.volume
        if cumulative_vol <= 0:
            result.append(candle.close)
        else:
            result.append(cumulative_tpv / cumulative_vol)

    return result


def _select_impulse_and_pullback_vwap(
    *,
    candles: list,
    impulse_bars: int,
    timezone_name: str,
) -> tuple[list[float], float]:
    vwap_series = _session_vwap_series(candles, timezone_name=timezone_name)
    impulse = vwap_series[-(impulse_bars + 2) : -2]
    pullback_vwap = vwap_series[-2]
    return impulse, pullback_vwap


def _ema_series(values: list[float], *, period: int) -> list[float]:
    alpha = 2.0 / (max(period, 2) + 1.0)
    result: list[float] = []
    ema_value = values[0]
    result.append(ema_value)
    for value in values[1:]:
        ema_value = (alpha * value) + ((1.0 - alpha) * ema_value)
        result.append(ema_value)
    return result
