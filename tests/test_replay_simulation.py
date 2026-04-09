from __future__ import annotations

import unittest

from core.models import MarketRegime
from core.trade_simulator import TradeSimulator, TradeStatus
from strategies.compression_breakout import CompressionBreakoutStrategy
from strategies.pullback_vwap_ema import TrendPullbackVWAPEMAStrategy
from tests.helpers import (
    build_context,
    build_indicator,
    build_signal,
    build_trend_sequence,
    make_candle,
)


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class ReplaySimulationTests(unittest.TestCase):
    def test_replay_pullback_setup_generates_signal(self) -> None:
        strategy = TrendPullbackVWAPEMAStrategy(
            params={
                "impulse_bars": 3,
                "impulse_atr_mult": 0.1,
                "min_bullish_bars_in_impulse": 1,
                "volume_impulse_mult": 0.1,
                "min_vwap_extension_atr": 0.0,
                "max_vwap_extension_atr": 10.0,
                "pullback_min_atr": 0.01,
                "pullback_max_atr": 10.0,
                "pullback_location_mode": "ANY",
            }
        )
        candles = build_trend_sequence()
        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=candles[-1].close - 1.0,
            ema_fast=candles[-1].close + 0.2,
            ema_slow=candles[-1].close - 0.2,
            atr=1.0,
            rolling_volume_avg=1000,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.TREND, indicators=indicators)

        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)

    def test_replay_compression_breakout_setup_generates_signal(self) -> None:
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
                "large_breakout_retest_threshold_atr": 2.0,
            }
        )

        candles = []
        for i in range(12):
            candles.append(make_candle(i, open_=100.0 + (i % 2) * 0.05, close=100.05 + (i % 2) * 0.05, high=100.3, low=99.9, volume=900))
        candles.append(make_candle(12, open_=100.1, close=100.15, high=100.25, low=99.95, volume=920))
        candles.append(make_candle(13, open_=100.3, close=100.75, high=100.85, low=100.25, volume=1400))

        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=100.2,
            ema_fast=100.25,
            ema_slow=100.22,
            atr=1.0,
            rolling_volume_avg=1000,
            vwap_slope=0.02,
            ema_distance=0.03,
            overlap_ratio=0.8,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.COMPRESSION, indicators=indicators)

        signal = strategy.evaluate(ctx)
        self.assertIsNotNone(signal)

    def test_replay_false_breakout_is_rejected(self) -> None:
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
                "late_breakout_extension_atr": 0.2,
                "large_breakout_retest_threshold_atr": 2.0,
            }
        )

        candles = []
        for i in range(12):
            candles.append(make_candle(i, open_=100.0, close=100.05, high=100.3, low=99.9, volume=900))
        candles.append(make_candle(12, open_=100.0, close=100.1, high=100.2, low=99.95, volume=900))
        # Oversized late extension beyond threshold -> rejected
        candles.append(make_candle(13, open_=100.4, close=101.0, high=101.2, low=100.35, volume=1500))

        indicators = build_indicator(
            timestamp=candles[-1].datetime,
            close=candles[-1].close,
            vwap=100.2,
            ema_fast=100.2,
            ema_slow=100.1,
            atr=1.0,
            rolling_volume_avg=1000,
            vwap_slope=0.01,
            ema_distance=0.1,
            overlap_ratio=0.8,
        )
        ctx = build_context(candles=candles, regime=MarketRegime.COMPRESSION, indicators=indicators)

        self.assertIsNone(strategy.evaluate(ctx))

    def test_replay_blackout_cancellation_before_activation(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        candle = make_candle(1, open_=98, close=98.2, high=98.4, low=97.8)
        out = sim.process_candle(candle=candle, session_active=True, blackout_active=True, blackout_reason="CPI")

        self.assertEqual([e.event_type for e in out], ["cancelled_by_news"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_NEWS)

    def test_replay_session_end_closure_during_active_trade(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100, close=100.1, high=100.2, low=99.8)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        session_end = make_candle(2, open_=100.0, close=100.0, high=100.1, low=99.9)
        out = sim.process_candle(candle=session_end, session_active=False, blackout_active=False, blackout_reason=None)

        self.assertEqual([e.event_type for e in out], ["cancelled_by_session_end"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_SESSION_END)


if __name__ == "__main__":
    unittest.main()
