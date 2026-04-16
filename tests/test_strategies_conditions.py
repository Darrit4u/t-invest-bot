from __future__ import annotations

import unittest

from core.models import MarketRegime, MarketRegimeState
from strategies.compression_breakout import CompressionBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from strategies.pullback_vwap_ema import TrendPullbackVWAPEMAStrategy
from tests.helpers import build_bear_trend_sequence, build_context, build_indicator, build_trend_sequence, make_candle


class StrategyConditionTests(unittest.TestCase):
    def test_trend_pullback_generates_signal_in_trend(self) -> None:
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
        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.direction.value, "LONG")
        self.assertGreaterEqual(signal.tp1, signal.entry + 2.0 * (signal.entry - signal.stop_loss))

    def test_trend_pullback_generates_short_signal_in_downtrend(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(params={})
        candles = build_bear_trend_sequence()
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=candles[-1].close + 1,
            ema_fast=candles[-1].close - 0.2,
            ema_slow=candles[-1].close + 0.2,
            atr=1.0,
            rolling_volume_avg=1000,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.TREND, indicators=indicators)
        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.direction.value, "SHORT")
        self.assertLessEqual(signal.tp1, signal.entry - 2.0 * (signal.stop_loss - signal.entry))

    def test_trend_pullback_rejects_when_pullback_never_reaches_ema_zone(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(params={"pullback_zone_tolerance_atr": 0.0})
        candles = []
        price = 100.0
        for i in range(240):
            if i < 236:
                open_, close = price, price + 0.14
            elif i < 239:
                open_, close = price, price + 0.05
            else:
                open_, close = price, price + 0.45
            candles.append(
                make_candle(
                    i,
                    open_=open_,
                    close=close,
                    high=max(open_, close) + 0.2,
                    low=min(open_, close) - 0.1,
                    volume=1200,
                )
            )
            price = close
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
        self.assertIsNone(strategy.evaluate(ctx))

    def test_trend_pullback_can_run_without_vwap_or_session_timing_bias(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(params={})
        candles = build_trend_sequence()
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=candles[-1].close + 5.0,
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

    def test_compression_breakout_retest_uses_max_retest_bars(self) -> None:
        strategy = CompressionBreakoutStrategy(
            params={
                "compression_window_bars": 12,
                "range_min_atr": 0.2,
                "range_max_atr": 3.0,
                "ema_distance_max_atr": 0.2,
                "vwap_slope_abs_max_atr": 0.2,
                "overlap_ratio_min": 0.5,
                "volume_floor_mult": 0.5,
                "breakout_body_min_atr": 0.2,
                "breakout_volume_mult": 1.0,
                "late_breakout_extension_atr": 0.8,
                "large_breakout_retest_threshold_atr": 0.4,
                "max_retest_bars": 2,
                "retest_tolerance_atr": 0.15,
            }
        )
        candles = [make_candle(i, open_=100.0, close=100.05, high=100.25, low=99.9, volume=900) for i in range(13)]
        breakout = make_candle(13, open_=100.1, close=100.65, high=100.75, low=100.0, volume=1400)
        mid = make_candle(14, open_=100.62, close=100.68, high=100.8, low=100.55, volume=950)
        confirm = make_candle(15, open_=100.62, close=100.72, high=100.78, low=100.22, volume=1200)
        candles.extend([breakout, mid, confirm])

        indicators = build_indicator(
            timestamp=confirm.datetime,
            close=confirm.close,
            vwap=100.2,
            ema_fast=100.25,
            ema_slow=100.21,
            atr=1.0,
            rolling_volume_avg=1000,
            ema_distance=0.04,
            vwap_slope=0.01,
            overlap_ratio=0.8,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.COMPRESSION, indicators=indicators)
        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.metadata.get("bars_since_breakout"), 2)

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

    def test_trend_pullback_uses_strategy_context_score_instead_of_exact_regime(self) -> None:
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
        regime_state = MarketRegimeState(
            dominant=MarketRegime.COMPRESSION,
            trend_score=0.72,
            compression_score=0.79,
            balance_score=0.31,
            reason_codes=("mixed_regime",),
            details={},
        )
        strategy = TrendPullbackVWAPEMAStrategy(
            params={
                "strategy_context_score_min": 0.60,
            }
        )
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.COMPRESSION,
            indicators=indicators,
            regime_state=regime_state,
        )
        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertGreaterEqual(float(signal.metadata.get("strategy_context_score", 0.0)), 0.60)

    def test_trend_pullback_blocks_when_strategy_context_score_below_threshold(self) -> None:
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
        regime_state = MarketRegimeState(
            dominant=MarketRegime.NEUTRAL,
            trend_score=0.35,
            compression_score=0.40,
            balance_score=0.25,
            reason_codes=("weak_trend",),
            details={},
        )
        strategy = TrendPullbackVWAPEMAStrategy(
            params={
                "strategy_context_score_min": 0.60,
            }
        )
        ctx = build_context(
            candles=candles,
            regime=MarketRegime.NEUTRAL,
            indicators=indicators,
            regime_state=regime_state,
        )
        self.assertIsNone(strategy.evaluate(ctx))


if __name__ == "__main__":
    unittest.main()
