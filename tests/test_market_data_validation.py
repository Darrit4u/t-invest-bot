from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.market_data import Candle, CandleValidationError


class CandleValidationTests(unittest.TestCase):
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "dt": datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
            "open_": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "instrument": "ES",
            "timeframe": "1min",
        }

    def test_rejects_non_numeric_ohlcv_values(self) -> None:
        field_map = {
            "open": "open_",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        for field_name, kw_name in field_map.items():
            with self.subTest(field_name=field_name):
                kwargs = self._base_kwargs()
                kwargs[kw_name] = "not-a-number"
                with self.assertRaises(CandleValidationError) as exc_ctx:
                    Candle.validated(**kwargs)
                self.assertIn(field_name, str(exc_ctx.exception))
                self.assertIn("finite number", str(exc_ctx.exception))

    def test_rejects_non_finite_ohlcv_values(self) -> None:
        field_map = {
            "open": "open_",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        invalid_values = (float("nan"), float("inf"), float("-inf"))
        for field_name, kw_name in field_map.items():
            for invalid in invalid_values:
                with self.subTest(field_name=field_name, invalid=repr(invalid)):
                    kwargs = self._base_kwargs()
                    kwargs[kw_name] = invalid
                    with self.assertRaises(CandleValidationError) as exc_ctx:
                        Candle.validated(**kwargs)
                    self.assertIn(field_name, str(exc_ctx.exception))
                    self.assertIn("finite number", str(exc_ctx.exception))


if __name__ == "__main__":
    unittest.main()
