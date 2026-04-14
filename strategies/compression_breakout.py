"""Compression breakout strategy."""

from __future__ import annotations

from dataclasses import replace
from statistics import mean

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal
from strategies.base import BaseStrategy
from strategies.mtf import mtf_alignment


class CompressionBreakoutStrategy(BaseStrategy):
    """Breakout entries from low-volatility compression ranges."""

    name = "compression_breakout"
    allowed_regime = MarketRegime.COMPRESSION

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        context_score = _strategy_context_score(
            context=context,
            strategy_name=self.name,
            fallback_regime=self.allowed_regime,
        )
        if context_score < self._float("strategy_context_score_min", 0.0):
            return None

        bars = self._int("compression_window_bars", 12)
        if len(context.candles) < bars + 2:
            return None

        signal = self._evaluate_immediate(context, bars)
        if signal is not None:
            return self._apply_mtf_alignment(
                context=context,
                signal=self._with_context_score(signal=signal, context_score=context_score),
            )

        signal = self._evaluate_retest(context, bars)
        if signal is None:
            return None
        return self._apply_mtf_alignment(
            context=context,
            signal=self._with_context_score(signal=signal, context_score=context_score),
        )

    @staticmethod
    def _with_context_score(*, signal: StrategySignal, context_score: float) -> StrategySignal:
        return replace(
            signal,
            metadata=dict(signal.metadata) | {"strategy_context_score": float(context_score)},
        )

    def _apply_mtf_alignment(
        self,
        *,
        context: StrategyContext,
        signal: StrategySignal,
    ) -> StrategySignal | None:
        mtf_ok, mtf_meta = mtf_alignment(
            enabled=self._bool("use_mtf_filter", False),
            candles=context.candles,
            source_timeframe=context.timeframe,
            direction=signal.direction,
            trend_timeframe=self._str("trend_timeframe", "1hour"),
            setup_timeframe=self._str("setup_timeframe", "15min"),
            fast_ema=max(2, self._int("mtf_fast_ema", 8)),
            slow_ema=max(3, self._int("mtf_slow_ema", 21)),
            slope_bars=max(1, self._int("mtf_slope_bars", 2)),
        )
        if not mtf_ok:
            return None
        return replace(signal, metadata=dict(signal.metadata) | mtf_meta)

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

        volume_baseline = max(
            context.indicators.rolling_volume_avg,
            mean(item.volume for item in window),
            1e-9,
        )
        volume_ratio = breakout.volume / volume_baseline
        strong_body = body >= self._float("breakout_body_strong_atr", 0.55) * atr
        if volume_ratio < self._float("breakout_volume_mult", 1.25) and not strong_body:
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
        max_retest_bars = max(1, self._int("max_retest_bars", 2))
        if len(context.candles) < bars + max_retest_bars + 1:
            return None

        confirm = context.candles[-1]
        large_threshold = self._float("large_breakout_retest_threshold_atr", 0.90) * atr
        breakout_volume_min = self._float("breakout_volume_mult", 1.25)
        tolerance = self._float("retest_tolerance_atr", 0.10) * atr
        confirm_volume_ratio = confirm.volume / max(context.indicators.rolling_volume_avg, 1e-9)
        requires_large_breakout = self._bool("retest_requires_large_breakout", False)

        candles = context.candles
        for bars_since_breakout in range(1, max_retest_bars + 1):
            breakout_pos = len(candles) - 1 - bars_since_breakout
            window_start = breakout_pos - bars
            if window_start < 0:
                continue

            window = candles[window_start:breakout_pos]
            breakout = candles[breakout_pos]

            range_high = max(item.high for item in window)
            range_low = min(item.low for item in window)
            range_width = range_high - range_low
            if not self._compression_is_valid(context, window, range_width):
                continue

            body = abs(breakout.close - breakout.open)
            strong_breakout = body > large_threshold
            if requires_large_breakout and not strong_breakout:
                continue
            if not requires_large_breakout and body < self._float("retest_breakout_body_min_atr", 0.18) * atr:
                continue

            breakout_volume_ratio = breakout.volume / max(context.indicators.rolling_volume_avg, 1e-9)
            if breakout_volume_ratio < breakout_volume_min and not strong_breakout:
                continue

            if breakout.close > range_high:
                if confirm.low > range_high + tolerance:
                    continue
                if confirm.close <= range_high:
                    continue
                if confirm.close <= confirm.open:
                    continue
                if confirm_volume_ratio < breakout_volume_min * self._float(
                    "retest_confirm_volume_factor",
                    0.7,
                ) and not strong_breakout:
                    continue
                return self._build_signal(
                    context=context,
                    direction=SignalDirection.LONG,
                    entry=confirm.close,
                    range_high=range_high,
                    range_low=range_low,
                    range_width=range_width,
                    breakout_body_atr=body / atr,
                    volume_ratio=confirm_volume_ratio,
                    breakout_volume_ratio=breakout_volume_ratio,
                    entry_mode="RETEST_CONFIRMATION_CLOSE",
                    retest_required=True,
                    bars_since_breakout=bars_since_breakout,
                )

            if breakout.close < range_low:
                if confirm.high < range_low - tolerance:
                    continue
                if confirm.close >= range_low:
                    continue
                if confirm.close >= confirm.open:
                    continue
                if confirm_volume_ratio < breakout_volume_min * self._float(
                    "retest_confirm_volume_factor",
                    0.7,
                ) and not strong_breakout:
                    continue
                return self._build_signal(
                    context=context,
                    direction=SignalDirection.SHORT,
                    entry=confirm.close,
                    range_high=range_high,
                    range_low=range_low,
                    range_width=range_width,
                    breakout_body_atr=body / atr,
                    volume_ratio=confirm_volume_ratio,
                    breakout_volume_ratio=breakout_volume_ratio,
                    entry_mode="RETEST_CONFIRMATION_CLOSE",
                    retest_required=True,
                    bars_since_breakout=bars_since_breakout,
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

        ema_distance = context.indicators.ema_distance
        ema_limit = self._float("ema_distance_max_atr", 0.12) * atr
        if ema_distance > ema_limit:
            overlap_relax_min = self._float("ema_relax_overlap_min", 0.72)
            hard_cap = self._float("ema_distance_hard_cap_atr", 0.26) * atr
            if context.indicators.overlap_ratio < overlap_relax_min or ema_distance > hard_cap:
                return False

        vwap_slope_abs = abs(context.indicators.vwap_slope)
        vwap_limit = self._float("vwap_slope_abs_max_atr", 0.04) * atr
        if vwap_slope_abs > vwap_limit:
            if (
                context.indicators.overlap_ratio < self._float("vwap_relax_overlap_min", 0.75)
                or vwap_slope_abs > self._float("vwap_slope_hard_cap_atr", 0.12) * atr
            ):
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
        breakout_volume_ratio: float | None = None,
        entry_mode: str,
        retest_required: bool,
        bars_since_breakout: int = 0,
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
                "breakout_volume_ratio": breakout_volume_ratio,
                "range_high": range_high,
                "range_low": range_low,
                "range_width": range_width,
                "retest_required": retest_required,
                "bars_since_breakout": bars_since_breakout,
            },
        )


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
