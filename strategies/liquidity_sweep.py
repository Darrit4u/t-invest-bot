"""Liquidity sweep reversal strategy."""

from __future__ import annotations

from dataclasses import replace

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy
from strategies.mtf import mtf_alignment


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

        if not self._local_balance_valid(context, lookback=lookback):
            return None

        reference_window = candles[-(lookback + 2) : -2]
        sweep = candles[-2]
        confirm = candles[-1]

        short_signal = self._try_short(context, reference_window, sweep, confirm)
        if short_signal is not None:
            return self._apply_mtf_alignment(context=context, signal=short_signal)

        long_signal = self._try_long(context, reference_window, sweep, confirm)
        if long_signal is None:
            return None
        return self._apply_mtf_alignment(context=context, signal=long_signal)

    def _apply_mtf_alignment(
        self,
        *,
        context: StrategyContext,
        signal: StrategySignal,
    ) -> StrategySignal | None:
        enabled = self._bool("use_mtf_filter", False)
        mtf_ok, mtf_meta = mtf_alignment(
            enabled=enabled,
            candles=context.candles,
            source_timeframe=context.timeframe,
            direction=signal.direction,
            trend_timeframe=self._str("trend_timeframe", "1hour"),
            setup_timeframe=self._str("setup_timeframe", "15min"),
            fast_ema=max(2, self._int("mtf_fast_ema", 8)),
            slow_ema=max(3, self._int("mtf_slow_ema", 21)),
            slope_bars=max(1, self._int("mtf_slope_bars", 2)),
        )
        if not enabled:
            return replace(signal, metadata=dict(signal.metadata) | mtf_meta)
        if mtf_ok:
            return replace(signal, metadata=dict(signal.metadata) | mtf_meta)

        # Reversal setups in BALANCE often fail trend-follow MTF checks by design.
        # Allow configurable override instead of hard rejecting all counter-trend sweeps.
        if self._bool("allow_balance_mtf_override", True):
            mode = self._str("balance_mtf_override_mode", "ALWAYS").strip().upper()
            desired = signal.direction.value
            trend_dir = str(mtf_meta.get("mtf_trend_direction", "NONE"))
            setup_dir = str(mtf_meta.get("mtf_setup_direction", "NONE"))

            override_ok = False
            if mode == "ALWAYS":
                override_ok = True
            elif mode == "COUNTER_TREND":
                override_ok = trend_dir != desired and setup_dir in {desired, "NONE"}
            else:
                # SETUP_ONLY (default): require setup TF to be aligned or neutral.
                override_ok = setup_dir in {desired, "NONE"}

            if override_ok:
                return replace(
                    signal,
                    metadata=dict(signal.metadata)
                    | mtf_meta
                    | {
                        "mtf_reversal_override": True,
                        "mtf_reversal_override_mode": mode,
                    },
                )
        return None

    def _local_balance_valid(self, context: StrategyContext, *, lookback: int) -> bool:
        atr = context.indicators.atr
        if atr <= 0:
            return False

        if context.indicators.crossing_count < self._int("balance_crosses_vwap_min", 4):
            return False

        if context.indicators.ema_distance > self._float("ema_distance_max_atr", 0.10) * atr:
            return False

        if abs(context.indicators.vwap_slope) > self._float("vwap_slope_abs_max_atr", 0.04) * atr:
            return False

        recent_window = max(lookback, 20)
        recent = context.candles[-recent_window:]
        day_range = max(item.high for item in recent) - min(item.low for item in recent)
        day_range_limit = self._float("day_range_max_atr", 3.0) * atr
        if day_range > day_range_limit:
            # Keep balance setups in slightly wider sessions if overlap remains high.
            if context.indicators.overlap_ratio < self._float("day_range_relax_overlap_min", 0.62):
                return False
            if day_range > self._float("day_range_hard_cap_atr", 5.5) * atr:
                return False

        impulse_size = abs(recent[-1].close - recent[0].open)
        if impulse_size > self._float("impulse_block_atr", 1.6) * atr:
            if context.indicators.crossing_count < self._int("impulse_block_crossing_override_min", 6):
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
        min_sweep, max_sweep, reference_range = self._sweep_bounds(
            context=context,
            atr=atr,
            reference_window=reference_window,
        )

        sweep_size = sweep.high - level
        if sweep_size < min_sweep:
            return None
        if sweep_size > max_sweep:
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
        return_tolerance = self._return_tolerance(
            atr=atr,
            sweep_size=sweep_size,
            reference_range=reference_range,
        )
        if (level - confirm.close) > return_tolerance:
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
                "sweep_min_effective": min_sweep,
                "sweep_max_effective": max_sweep,
                "wick_share": wick_share,
                "volume_ratio": volume_ratio,
                "return_tolerance": return_tolerance,
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
        min_sweep, max_sweep, reference_range = self._sweep_bounds(
            context=context,
            atr=atr,
            reference_window=reference_window,
        )

        sweep_size = level - sweep.low
        if sweep_size < min_sweep:
            return None
        if sweep_size > max_sweep:
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
        return_tolerance = self._return_tolerance(
            atr=atr,
            sweep_size=sweep_size,
            reference_range=reference_range,
        )
        if (confirm.close - level) > return_tolerance:
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
                "sweep_min_effective": min_sweep,
                "sweep_max_effective": max_sweep,
                "wick_share": wick_share,
                "volume_ratio": volume_ratio,
                "return_tolerance": return_tolerance,
                "balance_valid": True,
            },
        )

    def _sweep_bounds(
        self,
        *,
        context: StrategyContext,
        atr: float,
        reference_window: list,
    ) -> tuple[float, float, float]:
        reference_range = max(item.high for item in reference_window) - min(item.low for item in reference_window)
        tick_size = max(float(context.instrument.tick_size), 1e-9)

        min_atr = self._float("sweep_min_atr", 0.15) * atr
        min_range_share = max(0.0, self._float("sweep_min_range_share", 0.06))
        min_ticks = max(1, self._int("sweep_min_ticks", 1))
        min_from_range = reference_range * min_range_share
        min_from_ticks = min_ticks * tick_size

        # Adaptive floor: keep threshold meaningful in ATR terms but not too strict for 5m noise.
        min_sweep = max(min_from_ticks, min(min_atr, min_from_range))

        max_atr = self._float("sweep_max_atr", 0.75) * atr
        max_range_share = max(0.1, self._float("sweep_max_range_share", 0.9))
        max_from_range = reference_range * max_range_share
        max_sweep = max(min_sweep * 1.2, max(max_atr, max_from_range))
        return min_sweep, max_sweep, reference_range

    def _return_tolerance(self, *, atr: float, sweep_size: float, reference_range: float) -> float:
        base = self._float("return_close_distance_atr", 0.15) * atr
        sweep_component = max(0.0, self._float("return_close_sweep_mult", 1.0)) * sweep_size
        range_component = max(0.0, self._float("return_close_range_share", 0.08)) * reference_range
        return max(base, sweep_component, range_component)
