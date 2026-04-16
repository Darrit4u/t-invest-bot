from __future__ import annotations

import logging
import unittest

from core.instrument_registry import InstrumentRegistry
from core.models import MarketRegime, SignalDirection, StrategySignal
from core.news_filter import NewsBlackoutFilter
from core.signal_engine import SignalEngine
from storage.memory_store import MemoryCandleStore
from tests.helpers import build_instrument_meta, make_candle


class _ModeAwareDummyStrategy:
    name = "trend_pullback_vwap_ema"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def generate_signals(self, context):
        if not bool(self.params.get("enabled", True)):
            return []
        if len(context.candles) < 20:
            return []
        close = float(context.candles[-1].close)
        return [
            StrategySignal(
                signal_id="mode-aware-signal",
                instrument=context.instrument.symbol,
                strategy=self.name,
                regime=MarketRegime.TREND,
                direction=SignalDirection.LONG,
                timestamp=context.candles[-1].datetime,
                entry_mode="CONFIRMATION_CLOSE",
                entry=close,
                stop_loss=close - 0.8,
                tp1=close + 1.6,
                tp2=close + 2.4,
                metadata={"source": "mode-aware-dummy"},
            )
        ]


class StrategyModePresetsPhase5Tests(unittest.TestCase):
    def _build_engine(self, *, trading_mode: str) -> SignalEngine:
        meta = build_instrument_meta(strategies=("trend_pullback_vwap_ema",))
        registry = InstrumentRegistry(items={meta.symbol: meta})
        store = MemoryCandleStore(history_depth=500)
        for i in range(40):
            store.upsert(
                make_candle(
                    i,
                    open_=100 + i * 0.1,
                    high=100.2 + i * 0.1,
                    low=99.9 + i * 0.1,
                    close=100.1 + i * 0.1,
                    instrument=meta.symbol,
                    timeframe="1min",
                )
            )

        params = {
            "trading": {"mode": trading_mode},
            "signal_filter": {
                "min_signal_quality_score": 0.0,
                "trend_context_score_min": 0.0,
            },
            "strategy_params": {
                "by_mode": {
                    "intraday": {
                        "defaults": {
                            "trend_pullback_vwap_ema": {"enabled": False},
                        }
                    },
                    "swing": {
                        "defaults": {
                            "trend_pullback_vwap_ema": {"enabled": True},
                        }
                    },
                }
            },
        }
        engine = SignalEngine(
            registry=registry,
            store=store,
            params=params,
            blackout_filter=NewsBlackoutFilter(tuple()),
            logger=logging.getLogger("test.signal.mode.presets"),
        )
        engine._strategy_classes = {"trend_pullback_vwap_ema": _ModeAwareDummyStrategy}
        return engine

    def test_intraday_and_swing_presets_apply_differently(self) -> None:
        intraday_engine = self._build_engine(trading_mode="intraday")
        swing_engine = self._build_engine(trading_mode="swing")

        intraday_result = intraday_engine.process_candle(instrument="ES", timeframe="1min")
        swing_result = swing_engine.process_candle(instrument="ES", timeframe="1min")

        self.assertEqual(len(intraday_result.accepted_signals), 0)
        self.assertGreaterEqual(len(swing_result.accepted_signals), 1)


if __name__ == "__main__":
    unittest.main()
