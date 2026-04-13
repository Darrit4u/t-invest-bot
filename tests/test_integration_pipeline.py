from __future__ import annotations

import copy
import logging
import tempfile
import unittest
from pathlib import Path

from core.config_loader import ConfigLoader
from core.instrument_registry import InstrumentRegistry
from core.news_filter import NewsBlackoutFilter
from core.signal_engine import SignalEngine
from core.trade_simulator import TradeSimulator
from storage.memory_store import MemoryCandleStore
from storage.sqlite_store import SQLiteStore
from tests.helpers import build_signal, build_trend_sequence, config_dir, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class IntegrationPipelineTests(unittest.TestCase):
    def test_candle_ingestion_to_signal_generation(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)
        store = MemoryCandleStore(history_depth=500)

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
                "stop_buffer_atr": 0.05,
                "tp1_r": 1.0,
                "tp2_r": 2.0,
            }
        }

        engine = SignalEngine(
            registry=registry,
            store=store,
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal_engine"),
        )

        for candle in build_trend_sequence():
            store.upsert(candle)

        result = engine.process_candle(instrument="ES", timeframe="1min")
        self.assertIsNotNone(result.regime)
        self.assertGreaterEqual(len(result.accepted_signals), 1)

        signal = result.accepted_signals[0]
        self.assertGreater(signal.entry, 0)
        self.assertGreater(signal.tp1, signal.entry)

    def test_signal_to_activation_and_stop_outcome(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activation = make_candle(1, open_=100.0, high=100.3, low=99.7, close=100.1)
        sim.process_candle(candle=activation, session_active=True, blackout_active=False, blackout_reason=None)

        stop = make_candle(2, open_=99.3, high=99.8, low=98.9, close=99.1)
        out = sim.process_candle(candle=stop, session_active=True, blackout_active=False, blackout_reason=None)

        self.assertEqual([e.event_type for e in out], ["sl_hit"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertIsNotNone(trade.closed_at)
        self.assertLess(trade.net_pnl, 0)

    def test_sqlite_writes_signal_trade_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            store = SQLiteStore(db_path)
            sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=store)

            signal = build_signal(entry=100, stop_loss=99, tp1=101, tp2=102)
            store.save_signal(signal)
            events = sim.register_signal(signal, timeframe="1min")

            activation = make_candle(1, open_=100.0, high=100.3, low=99.7, close=100.1)
            sim.process_candle(candle=activation, session_active=True, blackout_active=False, blackout_reason=None)

            stop = make_candle(2, open_=99.0, high=99.3, low=98.7, close=99.1)
            sim.process_candle(candle=stop, session_active=True, blackout_active=False, blackout_reason=None)

            conn = store._conn
            signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            events_count = conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()[0]

            self.assertEqual(signals, 1)
            self.assertEqual(trades, 1)
            self.assertGreaterEqual(events_count, 3)

            store.close()


if __name__ == "__main__":
    unittest.main()
