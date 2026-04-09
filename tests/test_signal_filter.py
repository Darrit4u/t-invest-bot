from __future__ import annotations

import unittest

from core.models import MarketRegime, SignalDirection
from core.signal_filter import SignalFilterPipeline
from tests.helpers import build_context, build_indicator, build_instrument_meta, build_signal, make_candle


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
        pipeline = SignalFilterPipeline(params={})
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
        self.assertEqual(pipeline.evaluate(signal, ctx).reason, "tp1_too_small_after_fees")


if __name__ == "__main__":
    unittest.main()
