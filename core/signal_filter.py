"""Centralized signal validation and decision layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.models import MarketRegime, MarketRegimeState, SignalDecision, StrategyContext, StrategySignal
from core.post_fill_validation import (
    PostFillValidationConfig,
    expected_fill_price,
    validate_post_fill,
)


class SignalFilterPipeline:
    """Applies global checks before accepting strategy signals."""

    def __init__(self, params: dict[str, Any]):
        self._params = params
        self._cfg = params.get("signal_filter", {}) if isinstance(params.get("signal_filter", {}), dict) else {}
        self._commission_roundtrip = float(self._cfg.get("commission_roundtrip", 0.0008))
        self._safety_multiplier = float(self._cfg.get("safety_multiplier", 1.5))
        self._min_rr_after_fill = float(self._cfg.get("min_rr_after_fill", 0.50))
        self._min_expected_edge_after_fees = float(self._cfg.get("min_expected_edge_after_fees", 0.0))
        self._min_signal_quality_score = float(self._cfg.get("min_signal_quality_score", 0.55))
        self._expected_open_slippage_atr = float(self._cfg.get("expected_open_slippage_atr", 0.03))
        self._low_expectancy_hours_local = _parse_hours(self._cfg.get("low_expectancy_hours_local", []))
        self._strategy_context_thresholds = {
            "trend_pullback_vwap_ema": float(self._cfg.get("trend_context_score_min", 0.52)),
            "compression_breakout": float(self._cfg.get("compression_context_score_min", 0.52)),
            "liquidity_sweep_reversal": float(self._cfg.get("balance_context_score_min", 0.52)),
        }
        self._post_fill_cfg = PostFillValidationConfig(
            commission_roundtrip=self._commission_roundtrip,
            safety_multiplier=self._safety_multiplier,
            min_rr_after_fill=self._min_rr_after_fill,
            min_expected_edge_after_fees=self._min_expected_edge_after_fees,
        )

    def evaluate(self, signal: StrategySignal, context: StrategyContext) -> SignalDecision:
        instrument = context.instrument

        if not instrument.enabled:
            return self._reject(reason="instrument_disabled", reason_codes=("weak_context",))

        if not context.session_active:
            return self._reject(reason="session_inactive", reason_codes=("bad_timing",))

        if context.blackout_active:
            return self._reject(reason="blackout_active", reason_codes=("bad_timing",))

        if signal.strategy not in instrument.allowed_strategies:
            return self._reject(
                reason="strategy_not_allowed_for_instrument",
                reason_codes=("weak_context",),
            )

        if not self._is_signal_shape_valid(signal):
            return self._reject(reason="invalid_signal_shape", reason_codes=("weak_structure",))

        regime_state = context.regime_state or _fallback_regime_state(context.regime)
        context_score = regime_state.score_for_strategy(signal.strategy)
        min_context_score = self._strategy_context_thresholds.get(signal.strategy, 0.5)
        if context_score < min_context_score:
            return self._reject(
                reason="weak_context",
                reason_codes=("weak_context",),
                signal_quality_score=_clamp01(context_score),
                enriched_metadata=self._enriched_common(
                    regime_state=regime_state,
                    context_score=context_score,
                    signal_quality_score=_clamp01(context_score),
                ),
            )

        last_candle = context.candles[-1]
        expected_fill = expected_fill_price(
            direction=signal.direction,
            entry_mode=signal.entry_mode,
            planned_entry=signal.entry,
            last_close=last_candle.close,
            atr=context.indicators.atr,
            expected_open_slippage_atr=self._expected_open_slippage_atr,
        )
        post_fill = validate_post_fill(
            direction=signal.direction,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            entry_price=expected_fill,
            config=self._post_fill_cfg,
        )
        if not post_fill.accepted:
            return self._reject(
                reason=post_fill.reason or "poor_rr_after_fill",
                reason_codes=((post_fill.reason or "poor_rr_after_fill"),),
                expected_fill_price=expected_fill,
                post_fill_rr=post_fill.metrics.post_fill_rr,
                expected_edge_after_fees=post_fill.metrics.expected_edge_after_fees,
                enriched_metadata=self._enriched_common(
                    regime_state=regime_state,
                    context_score=context_score,
                    signal_quality_score=0.0,
                )
                | {
                    "expected_fill_price": expected_fill,
                    "post_fill_rr": post_fill.metrics.post_fill_rr,
                    "post_fill_risk": post_fill.metrics.risk,
                    "post_fill_reward": post_fill.metrics.reward,
                    "expected_edge_after_fees": post_fill.metrics.expected_edge_after_fees,
                },
            )
        post_fill_rr = post_fill.metrics.post_fill_rr
        expected_edge_after_fees = post_fill.metrics.expected_edge_after_fees

        setup_quality = _clamp01(float(signal.metadata.get("setup_quality_score", 0.50)))
        timing_score, timing_reason = self._timing_score(context=context)
        fill_quality = _clamp01((post_fill_rr / max(self._min_rr_after_fill, 1e-9)) * 0.6 + 0.4)
        signal_quality_score = _clamp01(
            (0.45 * setup_quality)
            + (0.25 * _clamp01(context_score))
            + (0.20 * fill_quality)
            + (0.10 * timing_score)
        )

        reason_codes: list[str] = []
        if timing_reason is not None:
            reason_codes.append(timing_reason)
        if bool(signal.metadata.get("late_trend_flag", False)):
            reason_codes.append("late_move")

        if signal_quality_score < self._min_signal_quality_score:
            primary = "bad_timing" if timing_score < 0.5 else "weak_structure"
            if primary not in reason_codes:
                reason_codes.append(primary)
            return self._reject(
                reason=primary,
                reason_codes=tuple(reason_codes),
                signal_quality_score=signal_quality_score,
                expected_fill_price=expected_fill,
                post_fill_rr=post_fill_rr,
                expected_edge_after_fees=expected_edge_after_fees,
                enriched_metadata=self._enriched_common(
                    regime_state=regime_state,
                    context_score=context_score,
                    signal_quality_score=signal_quality_score,
                )
                | {
                    "expected_fill_price": expected_fill,
                    "post_fill_rr": post_fill_rr,
                    "expected_edge_after_fees": expected_edge_after_fees,
                },
            )

        enriched_metadata = self._enriched_common(
            regime_state=regime_state,
            context_score=context_score,
            signal_quality_score=signal_quality_score,
        ) | {
            "signal_regime": signal.regime.value,
            "context_regime": context.regime.value,
            "cross_regime_signal": signal.regime != context.regime,
            "expected_fill_price": expected_fill,
            "post_fill_rr": post_fill_rr,
            "post_fill_risk": post_fill.metrics.risk,
            "post_fill_reward": post_fill.metrics.reward,
            "expected_edge_after_fees": expected_edge_after_fees,
        }
        return SignalDecision(
            accepted=True,
            reason="accepted",
            reason_codes=tuple(reason_codes),
            signal_quality_score=signal_quality_score,
            expected_fill_price=expected_fill,
            post_fill_rr=post_fill_rr,
            expected_edge_after_fees=expected_edge_after_fees,
            enriched_metadata=enriched_metadata,
        )

    @staticmethod
    def _is_signal_shape_valid(signal: StrategySignal) -> bool:
        prices = [signal.entry, signal.stop_loss, signal.tp1, signal.tp2]
        if any(value <= 0 for value in prices):
            return False

        if signal.direction.value == "LONG":
            return signal.stop_loss < signal.entry < signal.tp1 <= signal.tp2
        return signal.tp2 <= signal.tp1 < signal.entry < signal.stop_loss

    def _timing_score(self, *, context: StrategyContext) -> tuple[float, str | None]:
        if not self._low_expectancy_hours_local:
            return 1.0, None
        entry_hour = _entry_hour_local(context=context)
        if entry_hour in self._low_expectancy_hours_local:
            return 0.0, "bad_timing"
        return 1.0, None

    def _enriched_common(
        self,
        *,
        regime_state: MarketRegimeState,
        context_score: float,
        signal_quality_score: float,
    ) -> dict[str, Any]:
        return {
            "regime_dominant": regime_state.dominant.value,
            "trend_score": regime_state.trend_score,
            "compression_score": regime_state.compression_score,
            "balance_score": regime_state.balance_score,
            "regime_reason_codes": list(regime_state.reason_codes),
            "context_score": context_score,
            "signal_quality_score": signal_quality_score,
        }

    @staticmethod
    def _reject(
        *,
        reason: str,
        reason_codes: tuple[str, ...],
        signal_quality_score: float = 0.0,
        expected_fill_price: float | None = None,
        post_fill_rr: float | None = None,
        expected_edge_after_fees: float | None = None,
        enriched_metadata: dict[str, Any] | None = None,
    ) -> SignalDecision:
        return SignalDecision(
            accepted=False,
            reason=reason,
            reason_codes=reason_codes,
            signal_quality_score=signal_quality_score,
            expected_fill_price=expected_fill_price,
            post_fill_rr=post_fill_rr,
            expected_edge_after_fees=expected_edge_after_fees,
            enriched_metadata=enriched_metadata or {},
        )


def _entry_hour_local(*, context: StrategyContext) -> int:
    if context.instrument.sessions:
        timezone_name = context.instrument.sessions[0].timezone
    else:
        timezone_name = str(context.params.get("timezone", "UTC"))
    zone = ZoneInfo(timezone_name)
    ts = _entry_reference_timestamp(context=context)
    return ts.astimezone(zone).hour


def _entry_reference_timestamp(*, context: StrategyContext) -> datetime:
    return context.candles[-1].datetime


def _fallback_regime_state(regime: MarketRegime) -> MarketRegimeState:
    trend = 1.0 if regime == MarketRegime.TREND else 0.0
    compression = 1.0 if regime == MarketRegime.COMPRESSION else 0.0
    balance = 1.0 if regime == MarketRegime.BALANCE else 0.0
    return MarketRegimeState(
        dominant=regime,
        trend_score=trend,
        compression_score=compression,
        balance_score=balance,
        reason_codes=tuple(),
        details={},
    )


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


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)
