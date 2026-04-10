"""Trend Pullback to VWAP/EMA strategy."""

from __future__ import annotations

from statistics import mean

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy


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

        if self._mtf_alignment_enabled():
            mtf_bias = _mtf_bias(
                candles=candles,
                factor=max(2, self._int("mtf_factor", 3)),
                fast=max(2, self._int("mtf_fast_ema", 8)),
                slow=max(3, self._int("mtf_slow_ema", 21)),
            )
            if mtf_bias is None or mtf_bias != direction:
                return None

        impulse = candles[-(impulse_bars + 2) : -2]
        pullback = candles[-2]
        confirm = candles[-1]

        if direction == SignalDirection.LONG:
            return self._evaluate_long(context, impulse, pullback, confirm)
        return self._evaluate_short(context, impulse, pullback, confirm)

    def _mtf_alignment_enabled(self) -> bool:
        value = str(self.params.get("mtf_alignment_enabled", "false")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _evaluate_long(
        self,
        context: StrategyContext,
        impulse: list,
        pullback,
        confirm,
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
        touched_vwap = pullback.low <= context.indicators.vwap <= pullback.high
        touched_ema = pullback.low <= context.indicators.ema_fast <= pullback.high
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
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
            },
        )

    def _evaluate_short(
        self,
        context: StrategyContext,
        impulse: list,
        pullback,
        confirm,
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
        touched_vwap = pullback.low <= context.indicators.vwap <= pullback.high
        touched_ema = pullback.low <= context.indicators.ema_fast <= pullback.high
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
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
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


def _mtf_bias(
    *,
    candles: list,
    factor: int,
    fast: int,
    slow: int,
) -> SignalDirection | None:
    if len(candles) < factor * (slow + 2):
        return None

    closes: list[float] = []
    for idx in range(0, len(candles), factor):
        chunk = candles[idx : idx + factor]
        if len(chunk) < factor:
            continue
        closes.append(chunk[-1].close)

    if len(closes) < slow + 2:
        return None

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    if fast_ema > slow_ema:
        return SignalDirection.LONG
    if fast_ema < slow_ema:
        return SignalDirection.SHORT
    return None


def _ema(values: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = (alpha * value) + ((1.0 - alpha) * result)
    return result
