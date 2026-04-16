"""Regression: SignalEngine must consume generate_signals API."""

from __future__ import annotations

import logging
import unittest

from core.instrument_registry import InstrumentRegistry
from core.market_data import Candle
from core.models import MarketRegime, SignalDirection, StrategySignal
from core.news_filter import NewsBlackoutFilter
from core.signal_engine import SignalEngine
from storage.memory_store import MemoryCandleStore
from tests.helpers import build_instrument_meta, make_candle


class _GenerateOnlyStrategy:
    name = "trend_pullback_vwap_ema"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def evaluate(self, context):  # pragma: no cover - should never be called.
        raise AssertionError("SignalEngine must use generate_signals, not evaluate")

    def generate_signals(self, context):
        return [
            StrategySignal(
                signal_id="sig-1",
                instrument=context.instrument.symbol,
                strategy=self.name,
                regime=MarketRegime.TREND,
                direction=SignalDirection.LONG,
                timestamp=context.candles[-1].datetime,
                entry_mode="CONFIRMATION_CLOSE",
                entry=float(context.candles[-1].close),
                stop_loss=float(context.candles[-1].close - 1.0),
                tp1=float(context.candles[-1].close + 3.0),
                tp2=float(context.candles[-1].close + 4.0),
                metadata={"source": "dummy"},
            )
        ]


class SignalEngineGenerateSignalsTests(unittest.TestCase):
    def test_uses_generate_signals_contract(self) -> None:
        meta = build_instrument_meta(strategies=("trend_pullback_vwap_ema",))
        registry = InstrumentRegistry(items={meta.symbol: meta})
        store = MemoryCandleStore(history_depth=500)
        for i in range(40):
            candle: Candle = make_candle(
                i,
                open_=100 + i * 0.1,
                high=100.3 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.1 + i * 0.1,
                instrument=meta.symbol,
                timeframe="1min",
            )
            store.upsert(candle)

        params = {
            "signal_filter": {
                "min_signal_quality_score": 0.0,
                "trend_context_score_min": 0.0,
            }
        }
        engine = SignalEngine(
            registry=registry,
            store=store,
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal.engine.generate"),
        )
        engine._strategy_classes = {"trend_pullback_vwap_ema": _GenerateOnlyStrategy}

        result = engine.process_candle(instrument=meta.symbol, timeframe="1min")

        self.assertGreaterEqual(len(result.accepted_signals), 1)
        self.assertEqual(result.accepted_signals[0].strategy, "trend_pullback_vwap_ema")


if __name__ == "__main__":
    unittest.main()
