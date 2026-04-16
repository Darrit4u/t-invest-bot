from __future__ import annotations

import unittest

from core.trading_mode import TradingMode, resolve_primary_timeframe, resolve_trading_mode, should_enforce_session_filter


class TradingModePhase3Tests(unittest.TestCase):
    def test_default_mode_is_swing(self) -> None:
        self.assertEqual(resolve_trading_mode({}), TradingMode.SWING)

    def test_swing_mode_is_resolved_from_config(self) -> None:
        params = {"trading": {"mode": "swing"}}
        self.assertEqual(resolve_trading_mode(params), TradingMode.SWING)

    def test_primary_timeframe_uses_swing_override(self) -> None:
        params = {"trading": {"mode": "swing"}, "timeframes": {"primary": "1hour"}}
        self.assertEqual(resolve_primary_timeframe(params=params, default_timeframe="1min"), "1hour")

    def test_session_filter_defaults_to_mode(self) -> None:
        self.assertTrue(should_enforce_session_filter({"trading": {"mode": "intraday"}}))
        self.assertFalse(should_enforce_session_filter({"trading": {"mode": "swing"}}))

    def test_session_filter_can_be_overridden_explicitly(self) -> None:
        params = {"trading": {"mode": "swing", "enforce_session_filter": True}}
        self.assertTrue(should_enforce_session_filter(params))


if __name__ == "__main__":
    unittest.main()
