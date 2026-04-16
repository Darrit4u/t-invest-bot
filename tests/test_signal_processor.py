"""Unit tests for SignalProcessor pipeline stage."""

from __future__ import annotations

import unittest

from core.models import MarketRegime, SignalDirection
from core.signal_processor import SignalProcessor
from tests.helpers import build_context, build_instrument_meta, build_signal, make_candle


class SignalProcessorTests(unittest.TestCase):
    def test_accepts_valid_signal(self) -> None:
        processor = SignalProcessor(
            params={
                "signal_filter": {
                    "commission_roundtrip": 0.0008,
                    "safety_multiplier": 1.5,
                    "min_signal_quality_score": 0.0,
                    "strategy_context_thresholds": {"trend_pullback_vwap_ema": 0.0},
                }
            }
        )
        candles = [
            make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1, instrument="ES")
            for i in range(25)
        ]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
        )
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=103.0,
            tp2=104.0,
            entry_mode="CONFIRMATION_CLOSE",
        )

        result = processor.process_strategy_output(
            strategy_name="trend_pullback_vwap_ema",
            signals=[signal],
            context=ctx,
        )

        self.assertEqual(len(result.accepted_signals), 1)
        self.assertEqual(len(result.rejected_reasons), 0)

    def test_rejects_duplicate_by_instrument_strategy_timestamp(self) -> None:
        processor = SignalProcessor(
            params={
                "signal_filter": {
                    "commission_roundtrip": 0.0008,
                    "safety_multiplier": 1.5,
                    "min_signal_quality_score": 0.0,
                    "strategy_context_thresholds": {"trend_pullback_vwap_ema": 0.0},
                }
            }
        )
        candles = [
            make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1, instrument="ES")
            for i in range(25)
        ]
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.TREND,
            instrument=build_instrument_meta(strategies=("trend_pullback_vwap_ema",)),
        )
        signal = build_signal(
            strategy="trend_pullback_vwap_ema",
            timestamp=candles[-1].datetime,
            entry=100.0,
            stop_loss=99.0,
            tp1=103.0,
            tp2=104.0,
            entry_mode="CONFIRMATION_CLOSE",
        )

        first = processor.process_strategy_output(
            strategy_name="trend_pullback_vwap_ema",
            signals=[signal],
            context=ctx,
        )
        second = processor.process_strategy_output(
            strategy_name="trend_pullback_vwap_ema",
            signals=[signal],
            context=ctx,
        )

        self.assertEqual(len(first.accepted_signals), 1)
        self.assertEqual(len(second.accepted_signals), 0)
        self.assertIn("trend_pullback_vwap_ema:duplicate", second.rejected_reasons)


if __name__ == "__main__":
    unittest.main()
