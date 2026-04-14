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
        context_score = _strategy_context_score(
            context=context,
            strategy_name=self.name,
            fallback_regime=self.allowed_regime,
        )
        if context_score < self._float("strategy_context_score_min", 0.0):
            return None

        candles = context.candles
        impulse_bars = self._int("impulse_bars", 3)
        min_required = impulse_bars + 2
        if len(candles) < min_required:
            return None

        atr = context.indicators.atr
        if atr <= 0:
            return None

        if _entry_time_blocked(context=context, params=self.params):
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
                context_score=context_score,
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
            context_score=context_score,
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
        context_score: float,
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

        max_extension = max(
            item.high - impulse_vwap[idx]
            for idx, item in enumerate(impulse)
        )
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
        anchor_distance = min(
            _distance_to_range(level=pullback_vwap, low=pullback.low, high=pullback.high),
            _distance_to_range(level=pullback_ema_fast, low=pullback.low, high=pullback.high),
        )
        if zone_mode == "VWAP_ONLY" and not touched_vwap:
            return None
        if zone_mode == "EMA_FAST_ONLY" and not touched_ema:
            return None
        if zone_mode in {"VWAP_OR_EMA_FAST", "VWAP_EMA_ZONE"} and not (touched_vwap or touched_ema):
            return None
        if zone_mode == "ANY" and self._bool("enforce_anchor_proximity_any", False):
            max_anchor_distance = self._float("pullback_anchor_max_distance_atr", 0.35) * atr
            if anchor_distance > max_anchor_distance:
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
        if self._bool("confirmation_reclaim_ema_fast", False) and confirm.close <= pullback_ema_fast:
            return None
        if self._bool("confirmation_reclaim_vwap", False) and confirm.close <= pullback_vwap:
            return None

        stop = pullback.low - self._float("stop_buffer_atr", 0.15) * atr
        entry = confirm.close
        risk = entry - stop
        if risk <= 0:
            return None

        pullback_depth_atr = pullback_depth / atr
        anchor_distance_atr = anchor_distance / atr
        extension_atr = max_extension / atr
        confirm_body_atr = confirm_body / atr
        confirm_close_strength = _close_strength(
            close=confirm.close,
            low=confirm.low,
            high=confirm.high,
        )
        entry_location_in_day_range = _entry_location_in_day_range(
            candles=context.candles,
            entry=entry,
            timezone_name=_strategy_timezone(context),
        )
        timing_quality, timing_reason = _timing_quality(
            context=context,
            low_expectancy_hours_local=_parse_hours(
                self.params.get("low_expectancy_hours_local", [])
            ),
        )
        late_trend_flag = _is_late_trend(
            direction=SignalDirection.LONG,
            extension_atr=extension_atr,
            entry_location_in_day_range=entry_location_in_day_range,
            late_extension_atr=self._float("late_trend_extension_atr", 1.6),
            late_day_range_pos_long=self._float("late_trend_day_range_pos_long", 0.85),
            late_day_range_pos_short=self._float("late_trend_day_range_pos_short", 0.15),
        )
        if late_trend_flag and self._bool("block_late_trend_entries", False):
            return None
        if timing_reason is not None and self._bool("block_low_expectancy_hours", True):
            return None

        pullback_type = _pullback_type(
            depth_atr=pullback_depth_atr,
            anchor_distance_atr=anchor_distance_atr,
            pullback_max_atr=self._float("pullback_max_atr", 1.8),
            noise_anchor_distance_atr=self._float("noise_pullback_anchor_distance_atr", 0.45),
            exhaustion_share=self._float("exhaustion_pullback_share_of_max", 0.75),
        )
        impulse_quality = _clamp01(
            (
                0.45 * _safe_ratio(move / atr, self._float("impulse_atr_mult", 0.6))
                + 0.25 * _safe_ratio(float(bullish_count), float(len(impulse)))
                + 0.30 * _safe_ratio(volume_ratio, self._float("volume_impulse_mult", 0.9))
            )
        )
        pullback_depth_quality = _depth_quality(
            depth_atr=pullback_depth_atr,
            min_depth_atr=self._float("pullback_min_atr", 0.08),
            max_depth_atr=self._float("pullback_max_atr", 1.8),
        )
        pullback_anchor_quality = _clamp01(
            1.0 - _safe_ratio(anchor_distance_atr, self._float("pullback_anchor_max_distance_atr", 0.35))
        )
        pullback_quality = _clamp01(
            (0.55 * pullback_depth_quality + 0.45 * pullback_anchor_quality)
            * _pullback_type_multiplier(pullback_type)
        )
        confirm_quality = _clamp01(
            0.50 * _safe_ratio(confirm_body_atr, self._float("confirmation_body_min_atr", 0.05))
            + 0.50 * confirm_close_strength
        )
        extension_quality = _clamp01(
            1.0 - max(0.0, extension_atr - self._float("late_trend_extension_atr", 1.6)) / 2.0
        )
        setup_quality = _clamp01(
            0.30 * impulse_quality
            + 0.25 * pullback_quality
            + 0.25 * confirm_quality
            + 0.10 * extension_quality
            + 0.10 * timing_quality
        )
        if self._bool("enforce_setup_quality_threshold", False) and setup_quality < self._float(
            "min_setup_quality_score",
            0.55,
        ):
            return None

        reason_codes = _build_reason_codes(
            pullback_type=pullback_type,
            late_trend_flag=late_trend_flag,
            timing_reason=timing_reason,
        )

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
                "pullback_depth_atr": pullback_depth_atr,
                "volume_ratio": volume_ratio,
                "pullback_vwap": pullback_vwap,
                "pullback_ema_fast": pullback_ema_fast,
                "anchor_distance_atr": anchor_distance_atr,
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
                "setup_quality_score": setup_quality,
                "impulse_quality_score": impulse_quality,
                "pullback_quality_score": pullback_quality,
                "confirm_quality_score": confirm_quality,
                "late_trend_flag": late_trend_flag,
                "entry_location_in_day_range": entry_location_in_day_range,
                "pullback_type": pullback_type,
                "confirm_close_strength": confirm_close_strength,
                "strategy_context_score": context_score,
                "reason_codes": reason_codes,
                **mtf_meta,
            },
        )

    def _evaluate_short(
        self,
        context: StrategyContext,
        context_score: float,
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

        max_extension = max(
            impulse_vwap[idx] - item.low
            for idx, item in enumerate(impulse)
        )
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
        anchor_distance = min(
            _distance_to_range(level=pullback_vwap, low=pullback.low, high=pullback.high),
            _distance_to_range(level=pullback_ema_fast, low=pullback.low, high=pullback.high),
        )
        if zone_mode == "VWAP_ONLY" and not touched_vwap:
            return None
        if zone_mode == "EMA_FAST_ONLY" and not touched_ema:
            return None
        if zone_mode in {"VWAP_OR_EMA_FAST", "VWAP_EMA_ZONE"} and not (touched_vwap or touched_ema):
            return None
        if zone_mode == "ANY" and self._bool("enforce_anchor_proximity_any", False):
            max_anchor_distance = self._float("pullback_anchor_max_distance_atr", 0.35) * atr
            if anchor_distance > max_anchor_distance:
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
        if self._bool("confirmation_reclaim_ema_fast", False) and confirm.close >= pullback_ema_fast:
            return None
        if self._bool("confirmation_reclaim_vwap", False) and confirm.close >= pullback_vwap:
            return None

        stop = pullback.high + self._float("stop_buffer_atr", 0.15) * atr
        entry = confirm.close
        risk = stop - entry
        if risk <= 0:
            return None

        pullback_depth_atr = pullback_depth / atr
        anchor_distance_atr = anchor_distance / atr
        extension_atr = max_extension / atr
        confirm_body_atr = confirm_body / atr
        confirm_close_strength = _close_strength_short(
            close=confirm.close,
            low=confirm.low,
            high=confirm.high,
        )
        entry_location_in_day_range = _entry_location_in_day_range(
            candles=context.candles,
            entry=entry,
            timezone_name=_strategy_timezone(context),
        )
        timing_quality, timing_reason = _timing_quality(
            context=context,
            low_expectancy_hours_local=_parse_hours(
                self.params.get("low_expectancy_hours_local", [])
            ),
        )
        late_trend_flag = _is_late_trend(
            direction=SignalDirection.SHORT,
            extension_atr=extension_atr,
            entry_location_in_day_range=entry_location_in_day_range,
            late_extension_atr=self._float("late_trend_extension_atr", 1.6),
            late_day_range_pos_long=self._float("late_trend_day_range_pos_long", 0.85),
            late_day_range_pos_short=self._float("late_trend_day_range_pos_short", 0.15),
        )
        if late_trend_flag and self._bool("block_late_trend_entries", False):
            return None
        if timing_reason is not None and self._bool("block_low_expectancy_hours", True):
            return None

        pullback_type = _pullback_type(
            depth_atr=pullback_depth_atr,
            anchor_distance_atr=anchor_distance_atr,
            pullback_max_atr=self._float("pullback_max_atr", 1.8),
            noise_anchor_distance_atr=self._float("noise_pullback_anchor_distance_atr", 0.45),
            exhaustion_share=self._float("exhaustion_pullback_share_of_max", 0.75),
        )
        impulse_quality = _clamp01(
            (
                0.45 * _safe_ratio(move / atr, self._float("impulse_atr_mult", 0.6))
                + 0.25 * _safe_ratio(float(bearish_count), float(len(impulse)))
                + 0.30 * _safe_ratio(volume_ratio, self._float("volume_impulse_mult", 0.9))
            )
        )
        pullback_depth_quality = _depth_quality(
            depth_atr=pullback_depth_atr,
            min_depth_atr=self._float("pullback_min_atr", 0.08),
            max_depth_atr=self._float("pullback_max_atr", 1.8),
        )
        pullback_anchor_quality = _clamp01(
            1.0 - _safe_ratio(anchor_distance_atr, self._float("pullback_anchor_max_distance_atr", 0.35))
        )
        pullback_quality = _clamp01(
            (0.55 * pullback_depth_quality + 0.45 * pullback_anchor_quality)
            * _pullback_type_multiplier(pullback_type)
        )
        confirm_quality = _clamp01(
            0.50 * _safe_ratio(confirm_body_atr, self._float("confirmation_body_min_atr", 0.05))
            + 0.50 * confirm_close_strength
        )
        extension_quality = _clamp01(
            1.0 - max(0.0, extension_atr - self._float("late_trend_extension_atr", 1.6)) / 2.0
        )
        setup_quality = _clamp01(
            0.30 * impulse_quality
            + 0.25 * pullback_quality
            + 0.25 * confirm_quality
            + 0.10 * extension_quality
            + 0.10 * timing_quality
        )
        if self._bool("enforce_setup_quality_threshold", False) and setup_quality < self._float(
            "min_setup_quality_score",
            0.55,
        ):
            return None

        reason_codes = _build_reason_codes(
            pullback_type=pullback_type,
            late_trend_flag=late_trend_flag,
            timing_reason=timing_reason,
        )

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
                "pullback_depth_atr": pullback_depth_atr,
                "volume_ratio": volume_ratio,
                "pullback_vwap": pullback_vwap,
                "pullback_ema_fast": pullback_ema_fast,
                "anchor_distance_atr": anchor_distance_atr,
                "touched_zone": _touched_zone_label(touched_vwap=touched_vwap, touched_ema=touched_ema),
                "structure_valid": True,
                "setup_quality_score": setup_quality,
                "impulse_quality_score": impulse_quality,
                "pullback_quality_score": pullback_quality,
                "confirm_quality_score": confirm_quality,
                "late_trend_flag": late_trend_flag,
                "entry_location_in_day_range": entry_location_in_day_range,
                "pullback_type": pullback_type,
                "confirm_close_strength": confirm_close_strength,
                "strategy_context_score": context_score,
                "reason_codes": reason_codes,
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


def _distance_to_range(*, level: float, low: float, high: float) -> float:
    if low <= level <= high:
        return 0.0
    if level < low:
        return low - level
    return level - high


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return value / baseline


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _close_strength(*, close: float, low: float, high: float) -> float:
    candle_range = max(high - low, 1e-9)
    return _clamp01((close - low) / candle_range)


def _close_strength_short(*, close: float, low: float, high: float) -> float:
    candle_range = max(high - low, 1e-9)
    return _clamp01((high - close) / candle_range)


def _pullback_type(
    *,
    depth_atr: float,
    anchor_distance_atr: float,
    pullback_max_atr: float,
    noise_anchor_distance_atr: float,
    exhaustion_share: float,
) -> str:
    exhaustion_threshold = max(0.0, pullback_max_atr * exhaustion_share)
    if depth_atr >= exhaustion_threshold:
        return "exhaustion_pullback"
    if anchor_distance_atr > noise_anchor_distance_atr:
        return "random_noise_retracement"
    return "healthy_pullback"


def _pullback_type_multiplier(pullback_type: str) -> float:
    if pullback_type == "healthy_pullback":
        return 1.0
    if pullback_type == "exhaustion_pullback":
        return 0.6
    return 0.45


def _depth_quality(*, depth_atr: float, min_depth_atr: float, max_depth_atr: float) -> float:
    if max_depth_atr <= min_depth_atr:
        return 0.0
    if depth_atr < min_depth_atr or depth_atr > max_depth_atr:
        return 0.0
    midpoint = (min_depth_atr + max_depth_atr) / 2.0
    half = max((max_depth_atr - min_depth_atr) / 2.0, 1e-9)
    return _clamp01(1.0 - abs(depth_atr - midpoint) / half)


def _entry_location_in_day_range(*, candles: list, entry: float, timezone_name: str) -> float:
    if not candles:
        return 0.5
    zone = ZoneInfo(timezone_name)
    session_date = candles[-1].datetime.astimezone(zone).date()
    day_candles = [
        item
        for item in candles
        if item.datetime.astimezone(zone).date() == session_date
    ]
    if not day_candles:
        return 0.5
    day_high = max(item.high for item in day_candles)
    day_low = min(item.low for item in day_candles)
    day_range = max(day_high - day_low, 1e-9)
    return _clamp01((entry - day_low) / day_range)


def _is_late_trend(
    *,
    direction: SignalDirection,
    extension_atr: float,
    entry_location_in_day_range: float,
    late_extension_atr: float,
    late_day_range_pos_long: float,
    late_day_range_pos_short: float,
) -> bool:
    extension_late = extension_atr >= late_extension_atr
    if direction == SignalDirection.LONG:
        location_late = entry_location_in_day_range >= late_day_range_pos_long
    else:
        location_late = entry_location_in_day_range <= late_day_range_pos_short
    return extension_late or location_late


def _timing_quality(*, context: StrategyContext, low_expectancy_hours_local: set[int]) -> tuple[float, str | None]:
    if not low_expectancy_hours_local:
        return 1.0, None
    hour = _entry_hour_local(context=context)
    if hour in low_expectancy_hours_local:
        return 0.0, "bad_timing"
    return 1.0, None


def _build_reason_codes(
    *,
    pullback_type: str,
    late_trend_flag: bool,
    timing_reason: str | None,
) -> list[str]:
    reason_codes: list[str] = [pullback_type]
    if late_trend_flag:
        reason_codes.append("late_move")
    if timing_reason:
        reason_codes.append(timing_reason)
    return reason_codes


def _entry_time_blocked(*, context: StrategyContext, params: dict[str, Any]) -> bool:
    weekday = _entry_weekday_local(context=context)
    blocked_weekdays = _parse_weekdays(params.get("blocked_entry_weekdays_local", []))
    if blocked_weekdays and weekday in blocked_weekdays:
        return True

    allowed_weekdays = _parse_weekdays(params.get("allowed_entry_weekdays_local", []))
    if allowed_weekdays and weekday not in allowed_weekdays:
        return True

    raw = params.get("blocked_entry_hours_local", [])
    blocked = _parse_hours(raw)
    if not blocked:
        return False

    hour = _entry_hour_local(context=context)
    return hour in blocked


def _entry_hour_local(*, context: StrategyContext) -> int:
    zone_name = _strategy_timezone(context)
    zone = ZoneInfo(zone_name)
    return context.candles[-1].datetime.astimezone(zone).hour


def _entry_weekday_local(*, context: StrategyContext) -> int:
    zone_name = _strategy_timezone(context)
    zone = ZoneInfo(zone_name)
    return context.candles[-1].datetime.astimezone(zone).weekday()


def _parse_hours(raw: Any) -> set[int]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        return set()

    out: set[int] = set()
    for item in parts:
        try:
            hour = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23:
            out.add(hour)
    return out


def _parse_weekdays(raw: Any) -> set[int]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        return set()

    out: set[int] = set()
    for item in parts:
        try:
            weekday = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= weekday <= 6:
            out.add(weekday)
    return out
