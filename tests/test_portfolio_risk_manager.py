from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timezone

from core.models import Position, SignalDirection
from core.portfolio_risk_manager import PortfolioRiskManager
from tests.helpers import build_signal, dt_at


def _with_sizing_meta(signal):
    return replace(
        signal,
        metadata={
            "source": "test",
            "tick_size": 0.25,
            "tick_value": 12.5,
            "lot": 1,
            "lot_size": 1,
            "min_qty": 1.0,
            "qty_step": 1.0,
        },
    )


class PortfolioRiskManagerTests(unittest.TestCase):
    def _manager(self) -> PortfolioRiskManager:
        return PortfolioRiskManager.from_params(
            {
                "portfolio": {
                    "enabled": True,
                    "max_positions": 2,
                    "max_positions_per_instrument": 1,
                    "max_positions_per_strategy": 2,
                    "allow_multiple_positions_per_instrument": False,
                    "max_risk_per_trade_pct": 1.0,
                    "max_total_risk_pct": 2.0,
                    "max_instrument_risk_pct": 1.5,
                    "max_strategy_risk_pct": 2.0,
                    "max_positions_per_correlation_group": 1,
                    "correlation_groups": {
                        "energy": ["BRENT", "NG"],
                    },
                }
            }
        )

    def test_rejects_when_max_positions_reached(self) -> None:
        manager = self._manager()
        now = dt_at(0).astimezone(timezone.utc)
        open_positions = [
            Position(
                instrument="ES",
                side=SignalDirection.LONG,
                entry_price=100.0,
                size=1.0,
                opened_at=now,
                stop_loss=99.0,
                take_profit=102.0,
                strategy_id="trend_pullback_vwap_ema",
                status="activated",
            ),
            Position(
                instrument="SILVER",
                side=SignalDirection.LONG,
                entry_price=50.0,
                size=1.0,
                opened_at=now,
                stop_loss=49.5,
                take_profit=51.0,
                strategy_id="trend_pullback_vwap_ema",
                status="activated",
            ),
        ]
        signal = _with_sizing_meta(
            build_signal(instrument="BRENT", entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0)
        )

        result = manager.validate_signal(signal=signal, open_positions=open_positions)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "max_positions_limit")

    def test_rejects_by_correlation_group(self) -> None:
        manager = self._manager()
        now = dt_at(0).astimezone(timezone.utc)
        open_positions = [
            Position(
                instrument="BRENT",
                side=SignalDirection.LONG,
                entry_price=100.0,
                size=1.0,
                opened_at=now,
                stop_loss=99.0,
                take_profit=102.0,
                strategy_id="compression_breakout",
                status="activated",
            ),
        ]
        signal = _with_sizing_meta(
            build_signal(
                instrument="NG",
                strategy="trend_pullback_vwap_ema",
                entry=10.0,
                stop_loss=9.9,
                tp1=10.2,
                tp2=10.3,
            )
        )

        result = manager.validate_signal(signal=signal, open_positions=open_positions)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "correlation_group_limit")

    def test_accepts_signal_within_limits(self) -> None:
        manager = self._manager()
        signal = _with_sizing_meta(
            build_signal(instrument="ES", entry=100.0, stop_loss=99.2, tp1=101.5, tp2=102.0)
        )

        result = manager.validate_signal(signal=signal, open_positions=[])

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertGreater(result.signal_risk_pct, 0.0)

    def test_rejects_signal_when_sizing_metadata_missing(self) -> None:
        manager = self._manager()
        signal = _with_sizing_meta(
            build_signal(instrument="ES", entry=100.0, stop_loss=99.0, tp1=101.5, tp2=102.0)
        )
        signal = replace(signal, metadata={"source": "test"})

        result = manager.validate_signal(signal=signal, open_positions=[])

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "sizing_reject")
        self.assertEqual(result.details.get("sizing_reject_reason"), "missing_metadata")


if __name__ == "__main__":
    unittest.main()
