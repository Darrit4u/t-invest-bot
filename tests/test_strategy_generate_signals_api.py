"""Contract tests for list-based strategy signal API."""

from __future__ import annotations

import unittest

from core.models import MarketRegime
from strategies.compression_breakout import CompressionBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from strategies.pullback_vwap_ema import TrendPullbackVWAPEMAStrategy
from tests.helpers import build_context, build_indicator, build_trend_sequence, make_candle


class StrategyGenerateSignalsApiTests(unittest.TestCase):
    def test_trend_pullback_returns_list(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(params={})
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
        signals = strategy.generate_signals(ctx)
        self.assertIsInstance(signals, list)

    def test_compression_breakout_returns_list(self) -> None:
        strategy = CompressionBreakoutStrategy(params={})
        candles = [make_candle(i, open_=100.0, close=100.05, high=100.25, low=99.9, volume=900) for i in range(16)]
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
        signals = strategy.generate_signals(ctx)
        self.assertIsInstance(signals, list)

    def test_liquidity_sweep_returns_list(self) -> None:
        strategy = LiquiditySweepReversalStrategy(params={})
        candles = [make_candle(i, open_=100.0, close=100.05, high=100.2, low=99.9, volume=900) for i in range(24)]
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=100.0,
            ema_fast=100.03,
            ema_slow=99.99,
            atr=1.0,
            rolling_volume_avg=1000,
            crossing_count=6,
            ema_distance=0.04,
            vwap_slope=0.01,
            overlap_ratio=0.7,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.BALANCE, indicators=indicators)
        signals = strategy.generate_signals(ctx)
        self.assertIsInstance(signals, list)


if __name__ == "__main__":
    unittest.main()
