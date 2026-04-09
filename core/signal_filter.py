"""Centralized signal validation pipeline."""

from __future__ import annotations

from typing import Any

from core.models import MarketRegime, SignalDecision, StrategyContext, StrategySignal


REGIME_ALLOWED_STRATEGY = {
    MarketRegime.TREND: {"trend_pullback_vwap_ema"},
    MarketRegime.COMPRESSION: {"compression_breakout"},
    MarketRegime.BALANCE: {"liquidity_sweep_reversal"},
}


class SignalFilterPipeline:
    """Applies global checks before accepting strategy signals."""

    def __init__(self, params: dict[str, Any]):
        self._params = params
        self._risk_cfg = params.get("signal_filter", {}) if isinstance(params.get("signal_filter", {}), dict) else {}
        self._commission_roundtrip = float(self._risk_cfg.get("commission_roundtrip", 0.0008))
        self._safety_multiplier = float(self._risk_cfg.get("safety_multiplier", 1.5))

    def evaluate(self, signal: StrategySignal, context: StrategyContext) -> SignalDecision:
        instrument = context.instrument

        if not instrument.enabled:
            return SignalDecision(False, "instrument_disabled")

        if not context.session_active:
            return SignalDecision(False, "session_inactive")

        if context.blackout_active:
            return SignalDecision(False, "blackout_active")

        if signal.regime != context.regime:
            return SignalDecision(False, "regime_mismatch")

        if signal.strategy not in instrument.allowed_strategies:
            return SignalDecision(False, "strategy_not_allowed_for_instrument")

        allowed_by_regime = REGIME_ALLOWED_STRATEGY.get(context.regime, set())
        if signal.strategy not in allowed_by_regime:
            return SignalDecision(False, "strategy_not_allowed_for_regime")

        if not self._is_signal_shape_valid(signal):
            return SignalDecision(False, "invalid_signal_shape")

        rr_ok = self._risk_reward_valid(signal)
        if not rr_ok:
            return SignalDecision(False, "invalid_risk_reward")

        if not self._expected_profit_after_fees(signal):
            return SignalDecision(False, "tp1_too_small_after_fees")

        return SignalDecision(True, "accepted")

    @staticmethod
    def _is_signal_shape_valid(signal: StrategySignal) -> bool:
        prices = [signal.entry, signal.stop_loss, signal.tp1, signal.tp2]
        if any(value <= 0 for value in prices):
            return False

        if signal.direction.value == "LONG":
            return signal.stop_loss < signal.entry < signal.tp1 <= signal.tp2
        return signal.tp2 <= signal.tp1 < signal.entry < signal.stop_loss

    @staticmethod
    def _risk_reward_valid(signal: StrategySignal) -> bool:
        risk = abs(signal.entry - signal.stop_loss)
        reward = abs(signal.tp1 - signal.entry)
        if risk <= 0:
            return False
        return reward / risk >= 0.6

    def _expected_profit_after_fees(self, signal: StrategySignal) -> bool:
        expected = abs(signal.tp1 - signal.entry)
        commission_cost = signal.entry * self._commission_roundtrip
        return expected >= commission_cost * self._safety_multiplier
