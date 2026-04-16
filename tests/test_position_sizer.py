from __future__ import annotations

import unittest

from core.position_sizer import PositionSizer


class PositionSizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._sizer = PositionSizer()
        self._meta = {
            "tick_size": 0.25,
            "tick_value": 12.5,
            "lot_size": 1,
            "min_qty": 1,
            "qty_step": 1,
        }

    def test_sizes_quantity_from_risk_model_formula(self) -> None:
        out = self._sizer.size(
            entry_price=100.0,
            stop_loss=99.0,
            instrument_metadata=self._meta,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )

        self.assertTrue(out.accepted)
        assert out.result is not None
        # distance=1.0 -> ticks=4; 4*12.5*1=50 per contract; risk=1000 => 20 contracts
        self.assertEqual(out.result.ticks, 4)
        self.assertAlmostEqual(out.result.money_per_contract, 50.0, places=6)
        self.assertAlmostEqual(out.result.risk_money, 1000.0, places=6)
        self.assertAlmostEqual(out.result.qty, 20.0, places=6)
        self.assertAlmostEqual(out.result.risk_pct, 0.01, places=6)

    def test_rounds_down_quantity_to_qty_step(self) -> None:
        meta = dict(self._meta) | {"qty_step": 0.5, "min_qty": 0.5}
        out = self._sizer.size(
            entry_price=100.0,
            stop_loss=99.2,
            instrument_metadata=meta,
            account_equity=10_000.0,
            risk_per_trade_pct=0.01,
        )

        self.assertTrue(out.accepted)
        assert out.result is not None
        # distance=0.8 -> ticks=4; 4*12.5=50; risk=100 => raw=2.0 -> qty=2.0
        self.assertAlmostEqual(out.result.qty, 2.0, places=6)

    def test_rejects_when_metadata_missing(self) -> None:
        out = self._sizer.size(
            entry_price=100.0,
            stop_loss=99.0,
            instrument_metadata=None,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )

        self.assertFalse(out.accepted)
        assert out.reject is not None
        self.assertEqual(out.reject.reason, "missing_metadata")

    def test_rejects_when_stop_invalid_or_zero_distance(self) -> None:
        out_invalid_stop = self._sizer.size(
            entry_price=100.0,
            stop_loss=0.0,
            instrument_metadata=self._meta,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )
        self.assertFalse(out_invalid_stop.accepted)
        assert out_invalid_stop.reject is not None
        self.assertEqual(out_invalid_stop.reject.reason, "invalid_stop_loss")

        out_zero_distance = self._sizer.size(
            entry_price=100.0,
            stop_loss=100.0,
            instrument_metadata=self._meta,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )
        self.assertFalse(out_zero_distance.accepted)
        assert out_zero_distance.reject is not None
        self.assertEqual(out_zero_distance.reject.reason, "zero_stop_distance")

    def test_rejects_when_qty_below_min_qty(self) -> None:
        meta = dict(self._meta) | {"min_qty": 5.0, "qty_step": 1.0}
        out = self._sizer.size(
            entry_price=100.0,
            stop_loss=99.0,
            instrument_metadata=meta,
            account_equity=10_000.0,
            risk_per_trade_pct=0.01,
        )

        self.assertFalse(out.accepted)
        assert out.reject is not None
        self.assertEqual(out.reject.reason, "qty_below_min_qty")

    def test_accepts_legacy_lot_key_in_metadata(self) -> None:
        meta = {
            "tick_size": 0.25,
            "tick_value": 12.5,
            "lot": 1,
            "min_qty": 1,
            "qty_step": 1,
        }
        out = self._sizer.size(
            entry_price=100.0,
            stop_loss=99.0,
            instrument_metadata=meta,
            account_equity=10_000.0,
            risk_per_trade_pct=0.01,
        )
        self.assertTrue(out.accepted)


if __name__ == "__main__":
    unittest.main()
