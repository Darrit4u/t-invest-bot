from __future__ import annotations

import unittest

from core.models import MarketRegime
from strategies.compression_breakout import CompressionBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from strategies.pullback_vwap_ema import TrendPullbackVWAPEMAStrategy
from tests.helpers import build_context, build_indicator, build_trend_sequence, make_candle


class StrategyConditionTests(unittest.TestCase):
    def test_trend_pullback_generates_signal_in_trend(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(params={"impulse_atr_mult": 0.1, "volume_impulse_mult": 0.1, "pullback_location_mode": "ANY"})
        candles = build_trend_sequence()
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=candles[-1].close - 1,
            ema_fast=candles[-1].close + 0.2,
            ema_slow=candles[-1].close - 0.2,
            atr=1.0,
            rolling_volume_avg=1000,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.TREND, indicators=indicators)
        self.assertIsNotNone(strategy.evaluate(ctx))

    def test_compression_breakout_generates_signal(self) -> None:
        strategy = CompressionBreakoutStrategy(
            params={
                "compression_window_bars": 12,
                "range_min_atr": 0.2,
                "range_max_atr": 3.0,
                "breakout_body_min_atr": 0.2,
                "breakout_volume_mult": 1.0,
                "late_breakout_extension_atr": 0.8,
                "large_breakout_retest_threshold_atr": 2.0,
            }
        )
        candles = [make_candle(i, open_=100.0, close=100.05, high=100.25, low=99.9, volume=900) for i in range(13)]
        candles.append(make_candle(13, open_=100.25, close=100.75, high=100.85, low=100.2, volume=1400))
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=100.15,
            ema_fast=100.2,
            ema_slow=100.15,
            atr=1.0,
            rolling_volume_avg=1000,
            ema_distance=0.05,
            vwap_slope=0.01,
            overlap_ratio=0.8,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.COMPRESSION, indicators=indicators)
        self.assertIsNotNone(strategy.evaluate(ctx))

    def test_liquidity_sweep_reversal_generates_signal(self) -> None:
        strategy = LiquiditySweepReversalStrategy(params={})

        candles = [make_candle(i, open_=100.0, close=100.05, high=100.2, low=99.9, volume=900) for i in range(20)]
        sweep = make_candle(20, open_=100.05, close=100.0, high=100.45, low=99.9, volume=1300)
        confirm = make_candle(21, open_=100.18, close=100.1, high=100.2, low=99.95, volume=950)
        candles.extend([sweep, confirm])

        indicators = build_indicator(
            timestamp=confirm.datetime,
            close=confirm.close,
            vwap=100.0,
            ema_fast=100.03,
            ema_slow=99.99,
            atr=1.0,
            rolling_volume_avg=1000,
            crossing_count=6,
            ema_distance=0.04,
            vwap_slope=0.01,
            range_width=2.0,
            overlap_ratio=0.7,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.BALANCE, indicators=indicators)

        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.direction.value, "SHORT")

    def test_strategies_do_not_run_in_wrong_regime(self) -> None:
        candles = build_trend_sequence()
        indicators = build_indicator(timestamp=candles[-1].datetime)
        ctx = build_context(candles=candles, regime=MarketRegime.NEUTRAL, indicators=indicators)

        self.assertIsNone(TrendPullbackVWAPEMAStrategy(params={}).evaluate(ctx))
        self.assertIsNone(CompressionBreakoutStrategy(params={}).evaluate(ctx))
        self.assertIsNone(LiquiditySweepReversalStrategy(params={}).evaluate(ctx))


if __name__ == "__main__":
    unittest.main()
