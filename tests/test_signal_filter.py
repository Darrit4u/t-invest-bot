from __future__ import annotations

import unittest
from dataclasses import replace

from core.models import MarketRegime, MarketRegimeState, SignalDirection
from core.signal_filter import SignalFilterPipeline
from tests.helpers import build_context, build_indicator, build_instrument_meta, build_signal, dt_at, make_candle


class SignalFilterPipelineTests(unittest.TestCase):
    def _base_context(self):
        candles = [make_candle(0, open_=100, close=100.2)]
        return build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(),
        )

    def test_accepts_valid_signal(self) -> None:
        pipeline = SignalFilterPipeline(params={"signal_filter": {"commission_roundtrip": 0.0008, "safety_multiplier": 1.5}})
        ctx = self._base_context()
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            regime=MarketRegime.TREND,
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )

        decision = pipeline.evaluate(signal, ctx)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")

    def test_rejects_session_inactive(self) -> None:
        pipeline = SignalFilterPipeline(params={"trading": {"mode": "intraday"}})
        candles = [make_candle(0, open_=100, close=100.2)]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            session_active=False,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(),
        )
        signal = build_signal(strategy="trend_pullback_vwap_ema", regime=MarketRegime.TREND)
        self.assertEqual(pipeline.evaluate(signal, ctx).reason, "session_inactive")

    def test_rejects_invalid_shape(self) -> None:
        pipeline = SignalFilterPipeline(params={})
        ctx = self._base_context()
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            regime=MarketRegime.TREND,
            direction=SignalDirection.LONG,
            entry=100,
            stop_loss=101,
            tp1=102,
            tp2=103,
        )
        self.assertEqual(pipeline.evaluate(signal, ctx).reason, "invalid_signal_shape")

    def test_rejects_too_small_tp_after_fees(self) -> None:
        pipeline = SignalFilterPipeline(params={"signal_filter": {"commission_roundtrip": 0.01, "safety_multiplier": 2.0}})
        ctx = self._base_context()
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            regime=MarketRegime.TREND,
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.9,
            tp1=100.1,
            tp2=100.2,
        )
        decision = pipeline.evaluate(signal, ctx)
        self.assertIn(decision.reason, {"low_expected_edge", "poor_rr_after_fill"})
        self.assertTrue(
            {"low_expected_edge", "poor_rr_after_fill"}.intersection(set(decision.reason_codes))
        )

    def test_rejects_weak_context_when_regime_score_is_low(self) -> None:
        pipeline = SignalFilterPipeline(params={})
        candles = [make_candle(0, open_=100, close=100.2)]
        regime_state = MarketRegimeState(
            dominant=MarketRegime.NEUTRAL,
            trend_score=0.2,
            compression_score=0.3,
            balance_score=0.1,
            reason_codes=("trend_alignment_missing",),
            details={},
        )
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.NEUTRAL,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(),
            regime_state=regime_state,
        )
        signal = build_signal(strategy="trend_pullback_vwap_ema", regime=MarketRegime.NEUTRAL)

        decision = pipeline.evaluate(signal, ctx)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "weak_context")
        self.assertIn("weak_context", decision.reason_codes)

    def test_accepts_cross_regime_label_when_strategy_score_is_high(self) -> None:
        pipeline = SignalFilterPipeline(params={})
        candles = [make_candle(0, open_=100, close=100.2)]
        regime_state = MarketRegimeState(
            dominant=MarketRegime.COMPRESSION,
            trend_score=0.72,
            compression_score=0.81,
            balance_score=0.20,
            reason_codes=("mixed_state",),
            details={},
        )
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.COMPRESSION,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(),
            regime_state=regime_state,
        )
        signal = build_signal(strategy="trend_pullback_vwap_ema", regime=MarketRegime.TREND)
        decision = pipeline.evaluate(signal, ctx)
        self.assertTrue(decision.accepted)
        self.assertTrue(bool(decision.enriched_metadata.get("cross_regime_signal", False)))

    def test_rejects_low_liquidity_when_ratio_threshold_enabled(self) -> None:
        pipeline = SignalFilterPipeline(params={"signal_filter": {"min_bar_volume_ratio": 0.8}})
        candles = [make_candle(0, open_=100, close=100.2, volume=200)]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(rolling_volume_avg=1000.0),
        )
        signal = build_signal(strategy="trend_pullback_vwap_ema", regime=MarketRegime.TREND)

        decision = pipeline.evaluate(signal, ctx)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "low_liquidity")

    def test_rejects_signal_near_expiry_when_enabled(self) -> None:
        pipeline = SignalFilterPipeline(
            params={
                "futures": {
                    "block_near_expiry": True,
                    "expiry_buffer_days": 3,
                    "expiries": {"ES": "2026-01-10"},
                }
            }
        )
        ts = dt_at(0).replace(year=2026, month=1, day=8)
        candles = [
            make_candle(
                0,
                base=ts,
                open_=100,
                close=100.2,
                instrument="ES",
            )
        ]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(symbol="ES", strategies=("trend_pullback_vwap_ema",)),
            indicators=build_indicator(timestamp=ts),
        )
        signal = replace(
            build_signal(strategy="trend_pullback_vwap_ema", regime=MarketRegime.TREND, timestamp=ts),
            instrument="ES",
        )

        decision = pipeline.evaluate(signal, ctx)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "near_expiry_block")

    def test_late_trend_flag_string_values_are_parsed_as_bool(self) -> None:
        pipeline = SignalFilterPipeline(params={})
        ctx = self._base_context()
        base_signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            regime=MarketRegime.TREND,
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )
        cases = (
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
        )
        for raw_value, expected_flag in cases:
            with self.subTest(raw_value=raw_value):
                signal = replace(
                    base_signal,
                    metadata=dict(base_signal.metadata) | {"late_trend_flag": raw_value},
                )
                decision = pipeline.evaluate(signal, ctx)
                self.assertTrue(decision.accepted)
                self.assertEqual("late_move" in decision.reason_codes, expected_flag)


if __name__ == "__main__":
    unittest.main()
