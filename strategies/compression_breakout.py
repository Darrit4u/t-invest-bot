"""Compression breakout strategy."""

from __future__ import annotations

from statistics import mean

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy


class CompressionBreakoutStrategy(BaseStrategy):
    """Breakout entries from low-volatility compression ranges."""

    name = "compression_breakout"
    allowed_regime = MarketRegime.COMPRESSION

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        if context.regime != self.allowed_regime:
            return None

        bars = self._int("compression_window_bars", 12)
        if len(context.candles) < bars + 2:
            return None

        signal = self._evaluate_immediate(context, bars)
        if signal is not None:
            return signal

        return self._evaluate_retest(context, bars)

    def _evaluate_immediate(self, context: StrategyContext, bars: int) -> StrategySignal | None:
        atr = context.indicators.atr
        window = context.candles[-(bars + 1) : -1]
        breakout = context.candles[-1]

        range_high = max(item.high for item in window)
        range_low = min(item.low for item in window)
        range_width = range_high - range_low

        if not self._compression_is_valid(context, window, range_width):
            return None

        body = abs(breakout.close - breakout.open)
        if body < self._float("breakout_body_min_atr", 0.35) * atr:
            return None

        volume_ratio = breakout.volume / max(context.indicators.rolling_volume_avg, 1e-9)
        if volume_ratio < self._float("breakout_volume_mult", 1.25):
            return None

        max_extension = self._float("late_breakout_extension_atr", 0.30) * atr
        large_threshold = self._float("large_breakout_retest_threshold_atr", 0.90) * atr

        if breakout.close > range_high:
            extension = breakout.close - range_high
            if extension > max_extension:
                return None
            if body > large_threshold:
                return None
            return self._build_signal(
                context=context,
                direction=SignalDirection.LONG,
                entry=breakout.close,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                breakout_body_atr=body / atr,
                volume_ratio=volume_ratio,
                entry_mode="BREAKOUT_CLOSE",
                retest_required=False,
            )

        if breakout.close < range_low:
            extension = range_low - breakout.close
            if extension > max_extension:
                return None
            if body > large_threshold:
                return None
            return self._build_signal(
                context=context,
                direction=SignalDirection.SHORT,
                entry=breakout.close,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                breakout_body_atr=body / atr,
                volume_ratio=volume_ratio,
                entry_mode="BREAKOUT_CLOSE",
                retest_required=False,
            )

        return None

    def _evaluate_retest(self, context: StrategyContext, bars: int) -> StrategySignal | None:
        atr = context.indicators.atr
        if len(context.candles) < bars + 3:
            return None

        window = context.candles[-(bars + 2) : -2]
        breakout = context.candles[-2]
        confirm = context.candles[-1]

        range_high = max(item.high for item in window)
        range_low = min(item.low for item in window)
        range_width = range_high - range_low
        if not self._compression_is_valid(context, window, range_width):
            return None

        body = abs(breakout.close - breakout.open)
        large_threshold = self._float("large_breakout_retest_threshold_atr", 0.90) * atr
        if body <= large_threshold:
            return None

        tolerance = self._float("retest_tolerance_atr", 0.10) * atr
        volume_ratio = confirm.volume / max(context.indicators.rolling_volume_avg, 1e-9)

        if breakout.close > range_high:
            if confirm.low > range_high + tolerance:
                return None
            if confirm.close <= range_high:
                return None
            if confirm.close <= confirm.open:
                return None
            if volume_ratio < self._float("breakout_volume_mult", 1.25) * 0.8:
                return None
            return self._build_signal(
                context=context,
                direction=SignalDirection.LONG,
                entry=confirm.close,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                breakout_body_atr=body / atr,
                volume_ratio=volume_ratio,
                entry_mode="RETEST_CONFIRMATION_CLOSE",
                retest_required=True,
            )

        if breakout.close < range_low:
            if confirm.high < range_low - tolerance:
                return None
            if confirm.close >= range_low:
                return None
            if confirm.close >= confirm.open:
                return None
            if volume_ratio < self._float("breakout_volume_mult", 1.25) * 0.8:
                return None
            return self._build_signal(
                context=context,
                direction=SignalDirection.SHORT,
                entry=confirm.close,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                breakout_body_atr=body / atr,
                volume_ratio=volume_ratio,
                entry_mode="RETEST_CONFIRMATION_CLOSE",
                retest_required=True,
            )

        return None

    def _compression_is_valid(self, context: StrategyContext, window: list, range_width: float) -> bool:
        atr = context.indicators.atr
        if atr <= 0:
            return False

        if range_width < self._float("range_min_atr", 0.7) * atr:
            return False
        if range_width > self._float("range_max_atr", 2.0) * atr:
            return False

        if context.indicators.ema_distance > self._float("ema_distance_max_atr", 0.12) * atr:
            return False

        if abs(context.indicators.vwap_slope) > self._float("vwap_slope_abs_max_atr", 0.04) * atr:
            return False

        if context.indicators.overlap_ratio < self._float("overlap_ratio_min", 0.60):
            return False

        avg_vol = mean(item.volume for item in window)
        return avg_vol >= self._float("volume_floor_mult", 0.85) * context.indicators.rolling_volume_avg

    def _build_signal(
        self,
        *,
        context: StrategyContext,
        direction: SignalDirection,
        entry: float,
        range_high: float,
        range_low: float,
        range_width: float,
        breakout_body_atr: float,
        volume_ratio: float,
        entry_mode: str,
        retest_required: bool,
    ) -> StrategySignal:
        atr = context.indicators.atr
        stop_distance = max(
            self._float("stop_atr", 0.55) * atr,
            self._float("stop_range_factor", 0.50) * range_width,
        )

        if direction == SignalDirection.LONG:
            stop = entry - stop_distance
            risk = entry - stop
            tp1 = entry + self._float("tp1_r", 1.0) * risk
            tp2 = entry + self._float("tp2_r", 2.2) * risk
        else:
            stop = entry + stop_distance
            risk = stop - entry
            tp1 = entry - self._float("tp1_r", 1.0) * risk
            tp2 = entry - self._float("tp2_r", 2.2) * risk

        return self.build_signal(
            context=context,
            direction=direction,
            entry_mode=entry_mode,
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "compression_window_bars": self._int("compression_window_bars", 12),
                "overlap_ratio": context.indicators.overlap_ratio,
                "breakout_body_atr": breakout_body_atr,
                "volume_ratio": volume_ratio,
                "range_high": range_high,
                "range_low": range_low,
                "range_width": range_width,
                "retest_required": retest_required,
            },
        )
