from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from core.indicator_engine import IndicatorConfig, IndicatorEngine
from tests.helpers import make_candle


class IndicatorEngineTests(unittest.TestCase):
    def test_returns_none_when_history_is_insufficient(self) -> None:
        engine = IndicatorEngine(IndicatorConfig())
        candles = [make_candle(i, open_=100 + i, close=100 + i + 0.1) for i in range(8)]
        self.assertIsNone(engine.snapshot(candles))

    def test_snapshot_contains_expected_fields(self) -> None:
        engine = IndicatorEngine(IndicatorConfig())
        candles = [make_candle(i, open_=100 + i * 0.1, close=100 + i * 0.2, volume=900 + i * 5) for i in range(30)]
        snap = engine.snapshot(candles)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertGreater(snap.atr, 0)
        self.assertGreater(snap.rolling_volume_avg, 0)
        self.assertGreaterEqual(snap.overlap_ratio, 0)
        self.assertLessEqual(snap.overlap_ratio, 1)
        self.assertIsInstance(snap.crossing_count, int)

    def test_vwap_is_reset_per_session_day(self) -> None:
        engine = IndicatorEngine(IndicatorConfig())
        base = datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc)
        candles = []

        # Day 1 candles
        for i in range(22):
            candles.append(
                make_candle(
                    i,
                    open_=100 + i * 0.1,
                    close=100 + i * 0.1 + 0.05,
                    high=100 + i * 0.1 + 0.15,
                    low=100 + i * 0.1 - 0.05,
                    volume=1000,
                    base=base,
                )
            )

        # Day 2 candles (UTC date changed)
        day2_base = base + timedelta(days=1)
        day2 = [
            make_candle(0, open_=200.0, close=200.1, high=200.2, low=199.9, volume=500, base=day2_base),
            make_candle(1, open_=200.1, close=200.3, high=200.4, low=200.0, volume=500, base=day2_base),
            make_candle(2, open_=200.3, close=200.2, high=200.5, low=200.1, volume=500, base=day2_base),
        ]
        candles.extend(day2)

        snapshot = engine.snapshot(candles, session_timezone=timezone.utc)
        assert snapshot is not None

        tps = [((c.high + c.low + c.close) / 3.0) * c.volume for c in day2]
        vols = [c.volume for c in day2]
        expected_vwap_day2 = sum(tps) / sum(vols)

        self.assertAlmostEqual(snapshot.vwap, expected_vwap_day2, places=6)


if __name__ == "__main__":
    unittest.main()
