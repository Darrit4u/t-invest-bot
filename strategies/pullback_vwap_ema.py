"""Swing trend pullback strategy based on EMA20/50/200 structure."""

from __future__ import annotations

from typing import Any

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy
from strategies.mtf import mtf_alignment


class TrendPullbackVWAPEMAStrategy(BaseStrategy):
    """Classic swing pullback setup: trend by EMA50/200, pullback to EMA20-50 zone."""

    name = "trend_pullback_vwap_ema"
    allowed_regime = MarketRegime.TREND

    def generate_signals(self, context: StrategyContext) -> list[StrategySignal]:
        signal = self._evaluate_one(context)
        if signal is None:
            return []
        return [signal]

    def _evaluate_one(self, context: StrategyContext) -> StrategySignal | None:
        context_score = _strategy_context_score(
            context=context,
            strategy_name=self.name,
            fallback_regime=self.allowed_regime,
        )
        if context_score < self._float("strategy_context_score_min", 0.0):
            return None

        candles = context.candles
        atr = float(context.indicators.atr)
        if atr <= 0:
            return None

        pullback_period = max(2, self._int("pullback_ema_period", 20))
        trend_period = max(pullback_period + 1, self._int("trend_ema_period", 50))
        anchor_period = max(trend_period + 1, self._int("anchor_ema_period", 200))
        pullback_lookback = max(2, self._int("pullback_lookback_bars", 6))
        stop_lookback = max(2, self._int("stop_lookback_bars", 8))
        trend_slope_bars = max(1, self._int("trend_slope_bars", 3))
        min_required = max(anchor_period + trend_slope_bars + 2, pullback_lookback + 3, stop_lookback + 2)
        if len(candles) < min_required:
            return None

        closes = [item.close for item in candles]
        ema_pullback = _ema_series(closes, period=pullback_period)
        ema_trend = _ema_series(closes, period=trend_period)
        ema_anchor = _ema_series(closes, period=anchor_period)

        direction = _trend_direction(
            close=closes[-1],
            ema_trend=ema_trend,
            ema_anchor=ema_anchor,
            atr=atr,
            slope_bars=trend_slope_bars,
            min_trend_slope_atr=self._float("min_trend_slope_atr", 0.02),
        )
        if direction is None:
            return None

        mtf_ok, mtf_meta = mtf_alignment(
            enabled=self._bool("use_mtf_filter", False),
            candles=candles,
            source_timeframe=context.timeframe,
            direction=direction,
            trend_timeframe=self._str("trend_timeframe", "4hour"),
            setup_timeframe=self._str("setup_timeframe", "1hour"),
            fast_ema=max(2, self._int("mtf_fast_ema", 20)),
            slow_ema=max(3, self._int("mtf_slow_ema", 50)),
            slope_bars=max(1, self._int("mtf_slope_bars", 2)),
        )
        if not mtf_ok:
            return None

        if direction == SignalDirection.LONG:
            return self._evaluate_long(
                context=context,
                context_score=context_score,
                ema_pullback=ema_pullback,
                ema_trend=ema_trend,
                ema_anchor=ema_anchor,
                pullback_lookback=pullback_lookback,
                stop_lookback=stop_lookback,
                mtf_meta=mtf_meta,
            )
        return self._evaluate_short(
            context=context,
            context_score=context_score,
            ema_pullback=ema_pullback,
            ema_trend=ema_trend,
            ema_anchor=ema_anchor,
            pullback_lookback=pullback_lookback,
            stop_lookback=stop_lookback,
            mtf_meta=mtf_meta,
        )

    def _evaluate_long(
        self,
        *,
        context: StrategyContext,
        context_score: float,
        ema_pullback: list[float],
        ema_trend: list[float],
        ema_anchor: list[float],
        pullback_lookback: int,
        stop_lookback: int,
        mtf_meta: dict[str, Any],
    ) -> StrategySignal | None:
        candles = context.candles
        atr = float(context.indicators.atr)
        pullback = candles[-2]
        confirm = candles[-1]

        zone_tolerance = self._float("pullback_zone_tolerance_atr", 0.20) * atr
        ema20_pull = ema_pullback[-2]
        ema50_pull = ema_trend[-2]
        ema200_pull = ema_anchor[-2]
        zone_low = min(ema20_pull, ema50_pull) - zone_tolerance
        zone_high = max(ema20_pull, ema50_pull) + zone_tolerance

        if pullback.high < zone_low or pullback.low > zone_high:
            return None
        if pullback.low < (ema200_pull - self._float("anchor_breach_tolerance_atr", 0.25) * atr):
            return None

        recent_segment = candles[-(pullback_lookback + 2) : -2]
        if not recent_segment:
            return None
        recent_high = max(item.high for item in recent_segment)
        counter_trend_bars = sum(1 for item in candles[-(pullback_lookback + 1) : -1] if item.close < item.open)
        if counter_trend_bars < self._int("min_counter_trend_bars", 1):
            return None

        pullback_depth = recent_high - pullback.low
        min_depth = self._float("pullback_min_depth_atr", 0.25) * atr
        max_depth = self._float("pullback_max_depth_atr", 3.25) * atr
        if pullback_depth < min_depth or pullback_depth > max_depth:
            return None

        confirm_body = confirm.close - confirm.open
        if confirm_body <= self._float("confirmation_body_min_atr", 0.10) * atr:
            return None
        if confirm.close <= max(ema_pullback[-1], ema_trend[-1]):
            return None
        break_buffer = self._float("confirmation_break_buffer_atr", 0.05) * atr
        if confirm.close <= (pullback.high + break_buffer):
            return None

        entry = confirm.close
        stop_extreme = min(item.low for item in candles[-(stop_lookback + 1) : -1])
        atr_stop = entry - self._float("stop_atr_mult", 1.25) * atr
        stop = min(stop_extreme - self._float("stop_buffer_atr", 0.10) * atr, atr_stop)
        risk = entry - stop
        if risk <= 0:
            return None

        tp1_r = max(2.0, self._float("tp1_r", 2.0))
        tp2_r = max(self._float("tp2_r", 3.0), tp1_r)
        tp1 = entry + tp1_r * risk
        tp2 = entry + tp2_r * risk

        trend_strength = _clamp01((ema_trend[-1] - ema_anchor[-1]) / max(atr * self._float("trend_strength_atr", 1.5), 1e-9))
        zone_quality = _zone_quality(
            pullback_low=pullback.low,
            pullback_high=pullback.high,
            zone_low=zone_low,
            zone_high=zone_high,
            atr=atr,
        )
        confirm_quality = _clamp01(confirm_body / max(self._float("confirmation_body_min_atr", 0.10) * atr, 1e-9))
        setup_quality = _clamp01(0.40 * trend_strength + 0.30 * zone_quality + 0.30 * confirm_quality)

        return self.build_signal(
            context=context,
            direction=SignalDirection.LONG,
            entry_mode=self._str("entry_timing_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "structure_valid": True,
                "setup_quality_score": setup_quality,
                "trend_strength_score": trend_strength,
                "pullback_quality_score": zone_quality,
                "confirm_quality_score": confirm_quality,
                "strategy_context_score": context_score,
                "pullback_depth_atr": pullback_depth / max(atr, 1e-9),
                "pullback_zone_low": zone_low,
                "pullback_zone_high": zone_high,
                "ema_pullback": ema_pullback[-1],
                "ema_trend": ema_trend[-1],
                "ema_anchor": ema_anchor[-1],
                "rr_primary": tp1_r,
                "rr_secondary": tp2_r,
                "reason_codes": ["trend_filter_pass", "pullback_zone_touch", "bullish_confirmation"],
                **mtf_meta,
            },
        )

    def _evaluate_short(
        self,
        *,
        context: StrategyContext,
        context_score: float,
        ema_pullback: list[float],
        ema_trend: list[float],
        ema_anchor: list[float],
        pullback_lookback: int,
        stop_lookback: int,
        mtf_meta: dict[str, Any],
    ) -> StrategySignal | None:
        candles = context.candles
        atr = float(context.indicators.atr)
        pullback = candles[-2]
        confirm = candles[-1]

        zone_tolerance = self._float("pullback_zone_tolerance_atr", 0.20) * atr
        ema20_pull = ema_pullback[-2]
        ema50_pull = ema_trend[-2]
        ema200_pull = ema_anchor[-2]
        zone_low = min(ema20_pull, ema50_pull) - zone_tolerance
        zone_high = max(ema20_pull, ema50_pull) + zone_tolerance

        if pullback.high < zone_low or pullback.low > zone_high:
            return None
        if pullback.high > (ema200_pull + self._float("anchor_breach_tolerance_atr", 0.25) * atr):
            return None

        recent_segment = candles[-(pullback_lookback + 2) : -2]
        if not recent_segment:
            return None
        recent_low = min(item.low for item in recent_segment)
        counter_trend_bars = sum(1 for item in candles[-(pullback_lookback + 1) : -1] if item.close > item.open)
        if counter_trend_bars < self._int("min_counter_trend_bars", 1):
            return None

        pullback_depth = pullback.high - recent_low
        min_depth = self._float("pullback_min_depth_atr", 0.25) * atr
        max_depth = self._float("pullback_max_depth_atr", 3.25) * atr
        if pullback_depth < min_depth or pullback_depth > max_depth:
            return None

        confirm_body = confirm.open - confirm.close
        if confirm_body <= self._float("confirmation_body_min_atr", 0.10) * atr:
            return None
        if confirm.close >= min(ema_pullback[-1], ema_trend[-1]):
            return None
        break_buffer = self._float("confirmation_break_buffer_atr", 0.05) * atr
        if confirm.close >= (pullback.low - break_buffer):
            return None

        entry = confirm.close
        stop_extreme = max(item.high for item in candles[-(stop_lookback + 1) : -1])
        atr_stop = entry + self._float("stop_atr_mult", 1.25) * atr
        stop = max(stop_extreme + self._float("stop_buffer_atr", 0.10) * atr, atr_stop)
        risk = stop - entry
        if risk <= 0:
            return None

        tp1_r = max(2.0, self._float("tp1_r", 2.0))
        tp2_r = max(self._float("tp2_r", 3.0), tp1_r)
        tp1 = entry - tp1_r * risk
        tp2 = entry - tp2_r * risk

        trend_strength = _clamp01((ema_anchor[-1] - ema_trend[-1]) / max(atr * self._float("trend_strength_atr", 1.5), 1e-9))
        zone_quality = _zone_quality(
            pullback_low=pullback.low,
            pullback_high=pullback.high,
            zone_low=zone_low,
            zone_high=zone_high,
            atr=atr,
        )
        confirm_quality = _clamp01(confirm_body / max(self._float("confirmation_body_min_atr", 0.10) * atr, 1e-9))
        setup_quality = _clamp01(0.40 * trend_strength + 0.30 * zone_quality + 0.30 * confirm_quality)

        return self.build_signal(
            context=context,
            direction=SignalDirection.SHORT,
            entry_mode=self._str("entry_timing_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "structure_valid": True,
                "setup_quality_score": setup_quality,
                "trend_strength_score": trend_strength,
                "pullback_quality_score": zone_quality,
                "confirm_quality_score": confirm_quality,
                "strategy_context_score": context_score,
                "pullback_depth_atr": pullback_depth / max(atr, 1e-9),
                "pullback_zone_low": zone_low,
                "pullback_zone_high": zone_high,
                "ema_pullback": ema_pullback[-1],
                "ema_trend": ema_trend[-1],
                "ema_anchor": ema_anchor[-1],
                "rr_primary": tp1_r,
                "rr_secondary": tp2_r,
                "reason_codes": ["trend_filter_pass", "pullback_zone_touch", "bearish_confirmation"],
                **mtf_meta,
            },
        )


def _trend_direction(
    *,
    close: float,
    ema_trend: list[float],
    ema_anchor: list[float],
    atr: float,
    slope_bars: int,
    min_trend_slope_atr: float,
) -> SignalDirection | None:
    if len(ema_trend) <= slope_bars or len(ema_anchor) <= slope_bars:
        return None

    trend_last = ema_trend[-1]
    anchor_last = ema_anchor[-1]
    slope = trend_last - ema_trend[-1 - slope_bars]
    slope_threshold = min_trend_slope_atr * atr

    if trend_last > anchor_last and close > anchor_last and slope > slope_threshold:
        return SignalDirection.LONG
    if trend_last < anchor_last and close < anchor_last and slope < -slope_threshold:
        return SignalDirection.SHORT
    return None


def _zone_quality(
    *,
    pullback_low: float,
    pullback_high: float,
    zone_low: float,
    zone_high: float,
    atr: float,
) -> float:
    if pullback_high < zone_low or pullback_low > zone_high:
        return 0.0
    mid_zone = (zone_low + zone_high) / 2.0
    candle_mid = (pullback_low + pullback_high) / 2.0
    distance = abs(candle_mid - mid_zone)
    return _clamp01(1.0 - distance / max(atr, 1e-9))


def _strategy_context_score(
    *,
    context: StrategyContext,
    strategy_name: str,
    fallback_regime: MarketRegime,
) -> float:
    state = context.regime_state
    if state is not None:
        return float(state.score_for_strategy(strategy_name))
    return 1.0 if context.regime == fallback_regime else 0.0


def _ema_series(values: list[float], *, period: int) -> list[float]:
    alpha = 2.0 / (max(period, 2) + 1.0)
    result: list[float] = []
    ema_value = values[0]
    result.append(ema_value)
    for value in values[1:]:
        ema_value = (alpha * value) + ((1.0 - alpha) * ema_value)
        result.append(ema_value)
    return result


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)
