from __future__ import annotations

import unittest

from core.models import SignalDirection
from strategies.mtf import mtf_alignment
from tests.helpers import make_candle


class MtfRealismTests(unittest.TestCase):
    def test_incomplete_higher_timeframe_bucket_is_not_used(self) -> None:
        candles = [
            make_candle(
                i,
                open_=100.0 + i * 0.1,
                high=100.2 + i * 0.1,
                low=99.9 + i * 0.1,
                close=100.1 + i * 0.1,
                timeframe="5min",
            )
            for i in range(11)
        ]

        aligned, metadata = mtf_alignment(
            enabled=True,
            candles=candles,
            source_timeframe="5min",
            trend_timeframe="15min",
            setup_timeframe="15min",
            direction=SignalDirection.LONG,
            fast_ema=2,
            slow_ema=2,
            slope_bars=1,
        )

        self.assertFalse(aligned)
        self.assertEqual(metadata["mtf_trend_direction"], "NONE")
        self.assertEqual(metadata["mtf_setup_direction"], "NONE")


if __name__ == "__main__":
    unittest.main()
