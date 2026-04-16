from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.portfolio_events import DomainEvent, DomainEventType
from core.stats_engine import StatsEngine


class StatsEngineRiskReportingTests(unittest.TestCase):
    def test_tracks_risk_reject_reasons_and_planned_risk_stats(self) -> None:
        engine = StatsEngine()
        accepted = DomainEvent(
            kind=DomainEventType.SIGNAL_ACCEPTED,
            event_time=datetime(2026, 4, 16, 9, 0, tzinfo=timezone.utc),
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            signal_id="sig-1",
            trade_id=None,
            payload={"position_qty": 2.0, "planned_risk_money": 900.0, "planned_risk_pct": 0.9},
        )
        rejected = DomainEvent(
            kind=DomainEventType.RISK_REJECTED,
            event_time=datetime(2026, 4, 16, 9, 5, tzinfo=timezone.utc),
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            signal_id="sig-2",
            trade_id=None,
            reason="sizing_reject",
            payload={},
        )
        engine.record_portfolio_event(accepted)
        engine.record_portfolio_event(rejected)
        summary = engine.summary()["portfolio"]
        self.assertEqual(summary["signal_accepted"], 1)
        self.assertEqual(summary["risk_rejected"], 1)
        self.assertAlmostEqual(float(summary["avg_qty"]), 2.0, places=8)
        self.assertAlmostEqual(float(summary["avg_planned_risk_money"]), 900.0, places=8)
        self.assertEqual(summary["risk_reject_reasons"].get("sizing_reject"), 1)


if __name__ == "__main__":
    unittest.main()
