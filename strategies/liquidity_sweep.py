"""Liquidity sweep reversal strategy."""

from __future__ import annotations

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy


class LiquiditySweepReversalStrategy(BaseStrategy):
    """Mean-reversion strategy after failed sweep beyond local levels."""

    name = "liquidity_sweep_reversal"
    allowed_regime = MarketRegime.BALANCE

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        if context.regime != self.allowed_regime:
            return None

        candles = context.candles
        lookback = self._int("reference_lookback_bars", 20)
        if len(candles) < lookback + 2:
            return None

        if not self._local_balance_valid(context):
            return None

        reference_window = candles[-(lookback + 2) : -2]
        sweep = candles[-2]
        confirm = candles[-1]

        short_signal = self._try_short(context, reference_window, sweep, confirm)
        if short_signal is not None:
            return short_signal

        return self._try_long(context, reference_window, sweep, confirm)

    def _local_balance_valid(self, context: StrategyContext) -> bool:
        atr = context.indicators.atr
        if atr <= 0:
            return False

        if context.indicators.crossing_count < self._int("balance_crosses_vwap_min", 4):
            return False

        if context.indicators.ema_distance > self._float("ema_distance_max_atr", 0.10) * atr:
            return False

        if abs(context.indicators.vwap_slope) > self._float("vwap_slope_abs_max_atr", 0.04) * atr:
            return False

        recent = context.candles[-20:]
        day_range = max(item.high for item in recent) - min(item.low for item in recent)
        if day_range > self._float("day_range_max_atr", 3.0) * atr:
            return False

        impulse_size = abs(recent[-1].close - recent[0].open)
        if impulse_size > self._float("impulse_block_atr", 1.6) * atr:
            return False

        return True

    def _try_short(
        self,
        context: StrategyContext,
        reference_window: list,
        sweep,
        confirm,
    ) -> StrategySignal | None:
        atr = context.indicators.atr
        level = max(item.high for item in reference_window)

        sweep_size = sweep.high - level
        if sweep_size < self._float("sweep_min_atr", 0.15) * atr:
            return None
        if sweep_size > self._float("sweep_max_atr", 0.75) * atr:
            return None

        candle_range = max(1e-9, sweep.high - sweep.low)
        wick_share = (sweep.high - max(sweep.open, sweep.close)) / candle_range
        if wick_share < self._float("wick_min_share", 0.35):
            return None

        volume_ratio = sweep.volume / max(context.indicators.rolling_volume_avg, 1e-9)
        if volume_ratio < self._float("sweep_volume_mult", 1.20):
            return None

        if confirm.close >= level:
            return None
        if (level - confirm.close) > self._float("return_close_distance_atr", 0.15) * atr:
            return None
        if confirm.close >= confirm.open:
            return None

        entry = confirm.close
        stop = sweep.high + self._float("stop_buffer_atr", 0.12) * atr
        risk = stop - entry
        if risk <= 0:
            return None

        tp1 = entry - self._float("tp1_r", 0.8) * risk
        vwap_target = context.indicators.vwap
        fallback_tp2 = entry - self._float("tp2_r", 1.5) * risk
        tp2 = min(vwap_target, fallback_tp2)

        return self.build_signal(
            context=context,
            direction=SignalDirection.SHORT,
            entry_mode=self._str("entry_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "level_type": "SESSION_RANGE_HIGH",
                "reference_level": level,
                "sweep_extreme": sweep.high,
                "sweep_size_atr": sweep_size / atr,
                "wick_share": wick_share,
                "volume_ratio": volume_ratio,
                "balance_valid": True,
            },
        )

    def _try_long(
        self,
        context: StrategyContext,
        reference_window: list,
        sweep,
        confirm,
    ) -> StrategySignal | None:
        atr = context.indicators.atr
        level = min(item.low for item in reference_window)

        sweep_size = level - sweep.low
        if sweep_size < self._float("sweep_min_atr", 0.15) * atr:
            return None
        if sweep_size > self._float("sweep_max_atr", 0.75) * atr:
            return None

        candle_range = max(1e-9, sweep.high - sweep.low)
        wick_share = (min(sweep.open, sweep.close) - sweep.low) / candle_range
        if wick_share < self._float("wick_min_share", 0.35):
            return None

        volume_ratio = sweep.volume / max(context.indicators.rolling_volume_avg, 1e-9)
        if volume_ratio < self._float("sweep_volume_mult", 1.20):
            return None

        if confirm.close <= level:
            return None
        if (confirm.close - level) > self._float("return_close_distance_atr", 0.15) * atr:
            return None
        if confirm.close <= confirm.open:
            return None

        entry = confirm.close
        stop = sweep.low - self._float("stop_buffer_atr", 0.12) * atr
        risk = entry - stop
        if risk <= 0:
            return None

        tp1 = entry + self._float("tp1_r", 0.8) * risk
        vwap_target = context.indicators.vwap
        fallback_tp2 = entry + self._float("tp2_r", 1.5) * risk
        tp2 = max(vwap_target, fallback_tp2)

        return self.build_signal(
            context=context,
            direction=SignalDirection.LONG,
            entry_mode=self._str("entry_mode", "NEXT_BAR_OPEN"),
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            metadata={
                "level_type": "SESSION_RANGE_LOW",
                "reference_level": level,
                "sweep_extreme": sweep.low,
                "sweep_size_atr": sweep_size / atr,
                "wick_share": wick_share,
                "volume_ratio": volume_ratio,
                "balance_valid": True,
            },
        )
