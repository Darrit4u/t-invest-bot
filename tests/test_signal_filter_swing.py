from __future__ import annotations

import unittest

from core.models import MarketRegime, SignalDirection
from core.signal_filter import SignalFilterPipeline
from tests.helpers import build_context, build_instrument_meta, build_signal, make_candle


class SignalFilterSwingTests(unittest.TestCase):
    def test_swing_mode_can_accept_signal_outside_session(self) -> None:
        pipeline = SignalFilterPipeline(
            params={
                "trading": {"mode": "swing"},
                "signal_filter": {
                    "commission_roundtrip": 0.0008,
                    "safety_multiplier": 1.5,
                    "min_signal_quality_score": 0.0,
                    "trend_context_score_min": 0.0,
                },
            }
        )
        candles = [make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1, instrument="ES") for i in range(25)]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
            session_active=False,
        )
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry_mode="CONFIRMATION_CLOSE",
            entry=100.0,
            stop_loss=99.0,
            tp1=103.0,
            tp2=104.0,
        )

        decision = pipeline.evaluate(signal, ctx)
        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
