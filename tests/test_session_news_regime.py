from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.config_loader import BlackoutWindow, SessionRule
from core.instrument_registry import InstrumentMeta
from core.models import MarketRegime
from core.news_filter import NewsBlackoutFilter
from core.regime_classifier import MarketRegimeClassifier
from core.session_manager import SessionManager
from tests.helpers import build_indicator


class SessionNewsRegimeTests(unittest.TestCase):
    def test_session_manager_handles_regular_and_overnight_windows(self) -> None:
        manager = SessionManager()
        instrument = InstrumentMeta(
            symbol="ES",
            enabled=True,
            uid=None,
            figi=None,
            ticker="ES",
            class_code="SPBFUT",
            tick_size=0.25,
            tick_value=12.5,
            lot=1,
            sessions=(
                SessionRule(name="US", start="09:30", end="16:00", timezone="America/New_York"),
                SessionRule(name="ASIA", start="23:00", end="02:00", timezone="UTC"),
            ),
            allowed_strategies=("trend_pullback_vwap_ema",),
        )

        active_us = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
        inactive_us = datetime(2026, 1, 5, 3, 0, tzinfo=timezone.utc)
        active_overnight = datetime(2026, 1, 5, 23, 30, tzinfo=timezone.utc)

        self.assertTrue(manager.get_state(instrument, active_us).is_active)
        self.assertFalse(manager.get_state(instrument, inactive_us).is_active)
        self.assertTrue(manager.get_state(instrument, active_overnight).is_active)

    def test_news_blackout_boundaries_are_inclusive(self) -> None:
        start = datetime(2026, 4, 10, 15, 25, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 15, 40, tzinfo=timezone.utc)
        flt = NewsBlackoutFilter((BlackoutWindow(start=start, end=end, description="CPI"),))

        self.assertEqual(flt.is_blocked(start), (True, "CPI"))
        self.assertEqual(flt.is_blocked(end), (True, "CPI"))
        self.assertEqual(flt.is_blocked(datetime(2026, 4, 10, 15, 41, tzinfo=timezone.utc)), (False, None))

    def test_regime_classifier_routes_expected_regimes(self) -> None:
        clf = MarketRegimeClassifier.from_params({})

        trend = build_indicator(
            close=101,
            vwap=100,
            ema_fast=101.2,
            ema_slow=100.8,
            atr=1.0,
            vwap_slope=0.05,
            ema_distance=0.4,
            crossing_count=1,
            range_width=2.5,
            overlap_ratio=0.3,
        )
        self.assertEqual(clf.classify(trend), MarketRegime.TREND)

        compression = build_indicator(
            close=100,
            vwap=100,
            ema_fast=100.01,
            ema_slow=100.0,
            atr=1.0,
            vwap_slope=0.01,
            ema_distance=0.01,
            crossing_count=2,
            range_width=1.0,
            overlap_ratio=0.8,
        )
        self.assertEqual(clf.classify(compression), MarketRegime.COMPRESSION)

        balance = build_indicator(
            close=100,
            vwap=100,
            ema_fast=100.03,
            ema_slow=100.02,
            atr=1.0,
            vwap_slope=0.01,
            ema_distance=0.01,
            crossing_count=6,
            range_width=2.6,
            overlap_ratio=0.5,
        )
        self.assertEqual(clf.classify(balance), MarketRegime.BALANCE)

        neutral = build_indicator(
            close=100,
            vwap=100,
            ema_fast=100.1,
            ema_slow=100.05,
            atr=1.0,
            vwap_slope=0.0,
            ema_distance=0.05,
            crossing_count=3,
            range_width=3.5,
            overlap_ratio=0.3,
        )
        self.assertEqual(clf.classify(neutral), MarketRegime.NEUTRAL)

    def test_regime_classifier_returns_score_state_with_reason_codes(self) -> None:
        clf = MarketRegimeClassifier.from_params({})
        snapshot = build_indicator(
            close=101,
            vwap=100,
            ema_fast=101.4,
            ema_slow=100.8,
            atr=1.0,
            vwap_slope=0.06,
            ema_distance=0.5,
            crossing_count=1,
            range_width=2.4,
            overlap_ratio=0.4,
        )
        state = clf.classify_state(snapshot)

        self.assertEqual(state.dominant, MarketRegime.TREND)
        self.assertGreater(state.trend_score, 0.7)
        self.assertIn("trend_alignment_ok", state.reason_codes)


if __name__ == "__main__":
    unittest.main()
