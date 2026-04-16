from __future__ import annotations

import copy
import logging
import unittest

from core.config_loader import ConfigLoader
from core.instrument_registry import InstrumentRegistry
from core.news_filter import NewsBlackoutFilter
from core.signal_engine import SignalEngine
from storage.memory_store import MemoryCandleStore
from tests.helpers import build_trend_sequence, config_dir


class SignalEngineRegressionTests(unittest.TestCase):
    def _build_engine(self):
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)
        store = MemoryCandleStore(500)
        params = copy.deepcopy(cfg.params)
        params["regime_classifier"] = {
            "trend_ema_distance_atr": 0.01,
            "trend_vwap_slope_atr": 0.0,
            "trend_crossing_max": 20,
            "compression_range_min_atr": 999.0,
            "compression_range_max_atr": 1000.0,
            "compression_ema_distance_atr": 0.0,
            "compression_vwap_slope_abs_atr": 0.0,
            "compression_overlap_min": 2.0,
            "balance_crossing_min": 999,
            "balance_ema_distance_atr": 0.0,
            "balance_vwap_slope_abs_atr": 0.0,
        }
        params["strategy_params"] = {
            "trend_pullback_vwap_ema": {
                "impulse_bars": 3,
                "impulse_atr_mult": 0.1,
                "min_bullish_bars_in_impulse": 1,
                "min_bearish_bars_in_impulse": 1,
                "volume_impulse_mult": 0.1,
                "min_vwap_extension_atr": 0.0,
                "max_vwap_extension_atr": 10.0,
                "pullback_min_atr": 0.01,
                "pullback_max_atr": 10.0,
                "pullback_location_mode": "ANY",
            }
        }
        engine = SignalEngine(
            registry=registry,
            store=store,
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal_engine.reg"),
        )
        return engine, store

    def test_regime_allows_only_matching_strategy(self) -> None:
        engine, store = self._build_engine()
        for candle in build_trend_sequence():
            store.upsert(candle)

        result = engine.process_candle(instrument="ES", timeframe="1min")
        self.assertGreaterEqual(len(result.accepted_signals), 1)
        names = {signal.strategy for signal in result.accepted_signals}
        self.assertEqual(names, {"trend_pullback_vwap_ema"})

    def test_duplicate_evaluation_does_not_emit_duplicate_signal(self) -> None:
        engine, store = self._build_engine()
        for candle in build_trend_sequence():
            store.upsert(candle)

        first = engine.process_candle(instrument="ES", timeframe="1min")
        second = engine.process_candle(instrument="ES", timeframe="1min")

        self.assertGreaterEqual(len(first.accepted_signals), 1)
        self.assertEqual(len(second.accepted_signals), 0)
        self.assertTrue(any(reason.endswith(":duplicate") for reason in second.rejected_reasons))

    def test_builds_strategy_with_instrument_override_params(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)
        params = copy.deepcopy(cfg.params)
        params["strategy_params"] = {
            "defaults": {
                "trend_pullback_vwap_ema": {
                    "impulse_bars": 3,
                    "tp1_r": 1.0,
                }
            },
            "by_instrument": {
                "ES": {
                    "trend_pullback_vwap_ema": {
                        "tp1_r": 1.6,
                    }
                }
            },
        }

        engine = SignalEngine(
            registry=registry,
            store=MemoryCandleStore(100),
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal_engine.reg"),
        )

        strategy_es = engine._resolve_strategy(  # noqa: SLF001
            instrument="ES",
            strategy_name="trend_pullback_vwap_ema",
        )
        strategy_ng = engine._resolve_strategy(  # noqa: SLF001
            instrument="NG",
            strategy_name="trend_pullback_vwap_ema",
        )

        self.assertIsNotNone(strategy_es)
        self.assertIsNotNone(strategy_ng)
        self.assertEqual(strategy_es.params["tp1_r"], 1.6)
        self.assertEqual(strategy_ng.params["tp1_r"], 1.0)

    def test_strategy_can_be_disabled_per_instrument_pair(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)
        store = MemoryCandleStore(500)
        params = copy.deepcopy(cfg.params)
        params["regime_classifier"] = {
            "trend_ema_distance_atr": 0.01,
            "trend_vwap_slope_atr": 0.0,
            "trend_crossing_max": 20,
            "compression_range_min_atr": 999.0,
            "compression_range_max_atr": 1000.0,
            "compression_ema_distance_atr": 0.0,
            "compression_vwap_slope_abs_atr": 0.0,
            "compression_overlap_min": 2.0,
            "balance_crossing_min": 999,
            "balance_ema_distance_atr": 0.0,
            "balance_vwap_slope_abs_atr": 0.0,
        }
        params["strategy_params"] = {
            "defaults": {
                "trend_pullback_vwap_ema": {
                    "impulse_bars": 3,
                    "impulse_atr_mult": 0.1,
                    "min_bullish_bars_in_impulse": 1,
                    "min_bearish_bars_in_impulse": 1,
                    "volume_impulse_mult": 0.1,
                    "min_vwap_extension_atr": 0.0,
                    "max_vwap_extension_atr": 10.0,
                    "pullback_min_atr": 0.01,
                    "pullback_max_atr": 10.0,
                    "pullback_location_mode": "ANY",
                },
            },
            "by_instrument": {
                "ES": {
                    "trend_pullback_vwap_ema": {
                        "enabled": False,
                    }
                }
            },
        }

        engine = SignalEngine(
            registry=registry,
            store=store,
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal_engine.reg"),
        )
        for candle in build_trend_sequence():
            store.upsert(candle)

        result = engine.process_candle(instrument="ES", timeframe="1min")
        self.assertEqual(len(result.accepted_signals), 0)
        self.assertTrue(any(reason == "trend_pullback_vwap_ema:disabled" for reason in result.rejected_reasons))


if __name__ == "__main__":
    unittest.main()
