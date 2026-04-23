from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

from core.models import MarketRegime, SignalDirection
from core.trade_simulator import TradeSimulator, TradeStatus
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TradeSimulatorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)

    def test_trade_simulator_parses_string_bool_flags(self) -> None:
        cases = (
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
        )
        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                sim = TradeSimulator(
                    params={
                        "trade_simulator": {
                            "move_stop_to_breakeven": raw_value,
                            "close_active_on_blackout": raw_value,
                            "close_profitable_on_session_end": raw_value,
                            "revalidate_after_fill": raw_value,
                            "intrabar_stop_priority": raw_value,
                        }
                    },
                    logger=_DummyLogger(),
                    storage=None,
                )
                self.assertEqual(sim._move_stop_to_breakeven, expected)  # type: ignore[attr-defined]
                self.assertEqual(sim._close_active_on_blackout, expected)  # type: ignore[attr-defined]
                self.assertEqual(sim._close_profitable_on_session_end, expected)  # type: ignore[attr-defined]
                self.assertEqual(sim._revalidate_after_fill, expected)  # type: ignore[attr-defined]
                self.assertEqual(sim._intrabar_stop_priority, expected)  # type: ignore[attr-defined]

    def test_long_trade_hits_tp1_then_tp2_with_fees(self) -> None:
        signal = build_signal(
            regime=MarketRegime.TREND,
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )
        self.sim.register_signal(signal, timeframe="1min")

        c1 = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1, volume=1000, instrument="ES")
        e1 = self.sim.process_candle(candle=c1, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in e1], ["activated"])

        trade = self.sim.get_trade(e1[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.ACTIVATED)
        self.assertAlmostEqual(trade.fees_paid, 0.04, places=6)

        c2 = make_candle(2, open_=100.2, high=101.2, low=100.05, close=101.1, volume=1000, instrument="ES")
        e2 = self.sim.process_candle(candle=c2, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in e2], ["tp1_hit"])

        trade = self.sim.get_trade(e1[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.TP1_HIT)
        self.assertAlmostEqual(trade.remaining_qty, 0.5, places=6)
        self.assertAlmostEqual(trade.current_stop, 100.0, places=6)

        c3 = make_candle(3, open_=101.1, high=102.2, low=101.0, close=102.0, volume=1000, instrument="ES")
        e3 = self.sim.process_candle(candle=c3, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in e3], ["tp2_hit"])

        trade = self.sim.get_trade(e1[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.TP2_HIT)
        self.assertIsNotNone(trade.closed_at)
        self.assertAlmostEqual(trade.gross_pnl, 1.5, places=6)
        self.assertAlmostEqual(trade.fees_paid, 0.0806, places=6)
        self.assertAlmostEqual(trade.net_pnl, 1.4194, places=6)
        self.assertAlmostEqual(trade.r_multiple, 1.4194, places=6)

    def test_short_trade_hits_stop_loss_with_fee_accounting(self) -> None:
        signal = build_signal(
            regime=MarketRegime.TREND,
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.SHORT,
            entry=100.0,
            stop_loss=101.0,
            tp1=99.0,
            tp2=98.0,
        )
        self.sim.register_signal(signal, timeframe="1min")

        c1 = make_candle(1, open_=100.0, high=100.1, low=99.8, close=100.0, volume=1000, instrument="ES")
        e1 = self.sim.process_candle(candle=c1, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in e1], ["activated"])

        c2 = make_candle(2, open_=100.2, high=101.3, low=100.0, close=101.1, volume=1000, instrument="ES")
        e2 = self.sim.process_candle(candle=c2, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in e2], ["sl_hit"])

        trade = self.sim.get_trade(e1[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.SL_HIT)
        self.assertAlmostEqual(trade.gross_pnl, -1.0, places=6)
        self.assertAlmostEqual(trade.fees_paid, 0.0804, places=6)
        self.assertAlmostEqual(trade.net_pnl, -1.0804, places=6)
        self.assertLess(trade.r_multiple, -1.0)

    def test_post_fill_validation_can_expire_trade_when_rr_collapses(self) -> None:
        sim = TradeSimulator(
            params={"trade_simulator": {"min_rr_after_fill": 0.5, "revalidate_after_fill": True}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(
            regime=MarketRegime.TREND,
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )
        events = sim.register_signal(signal, timeframe="1min")

        # Gap-up fill worsens RR below threshold.
        c1 = make_candle(1, open_=100.6, high=100.8, low=100.5, close=100.7, volume=1000, instrument="ES")
        out = sim.process_candle(candle=c1, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in out], ["activated", "expired"])

        trade = sim.get_trade(events[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.EXPIRED)
        self.assertEqual(trade.exit_reason, "poor_rr_after_fill")

    def test_post_fill_validation_can_expire_trade_on_low_expected_edge(self) -> None:
        sim = TradeSimulator(
            params={
                "signal_filter": {"commission_roundtrip": 0.02, "safety_multiplier": 2.0},
                "trade_simulator": {
                    "min_rr_after_fill": 0.5,
                    "min_expected_edge_after_fees": 0.0,
                    "revalidate_after_fill": True,
                },
            },
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(
            regime=MarketRegime.TREND,
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=100.7,
            tp2=101.4,
        )
        events = sim.register_signal(signal, timeframe="1min")

        c1 = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1, volume=1000, instrument="ES")
        out = sim.process_candle(candle=c1, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([x.event_type for x in out], ["activated", "expired"])

        trade = sim.get_trade(events[0].trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.EXPIRED)
        self.assertEqual(trade.exit_reason, "low_expected_edge")

    def test_register_signal_uses_sized_quantity_from_metadata(self) -> None:
        signal = build_signal(
            regime=MarketRegime.TREND,
            strategy="trend_pullback_vwap_ema",
            direction=SignalDirection.LONG,
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )
        signal = replace(signal, metadata={"source": "test", "position_qty": 7.0, "tick_size": 0.01})
        events = self.sim.register_signal(signal, timeframe="1min")
        trade = self.sim.get_trade(events[0].trade_id)
        assert trade is not None
        self.assertAlmostEqual(trade.quantity, 7.0, places=6)
        self.assertAlmostEqual(trade.remaining_qty, 7.0, places=6)

    def test_r_multiple_is_normalized_by_full_position_risk_for_qty_gt_one(self) -> None:
        cases = (
            {
                "name": "long_qty2",
                "direction": SignalDirection.LONG,
                "entry": 100.0,
                "stop_loss": 99.0,
                "tp1": 101.0,
                "tp2": 102.0,
                "qty": 2.0,
                "first": {"open_": 100.0, "high": 100.3, "low": 99.8, "close": 100.1},
                "second": {"open_": 100.2, "high": 102.2, "low": 100.0, "close": 102.0},
                "expected_r": 2.0,
            },
            {
                "name": "short_qty3",
                "direction": SignalDirection.SHORT,
                "entry": 100.0,
                "stop_loss": 101.0,
                "tp1": 99.0,
                "tp2": 98.0,
                "qty": 3.0,
                "first": {"open_": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
                "second": {"open_": 99.8, "high": 100.4, "low": 97.7, "close": 98.1},
                "expected_r": 2.0,
            },
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                sim = TradeSimulator(
                    params={
                        "trade_simulator": {
                            "commission_per_side": 0.0,
                            "tp1_size": 0.0,
                        }
                    },
                    logger=_DummyLogger(),
                    storage=None,
                )
                signal = build_signal(
                    regime=MarketRegime.TREND,
                    strategy="trend_pullback_vwap_ema",
                    direction=case["direction"],
                    entry=case["entry"],
                    stop_loss=case["stop_loss"],
                    tp1=case["tp1"],
                    tp2=case["tp2"],
                )
                signal = replace(
                    signal,
                    metadata={"source": "test", "position_qty": case["qty"], "tick_size": 0.01},
                )
                events = sim.register_signal(signal, timeframe="1min")
                trade_id = events[0].trade_id

                c1 = make_candle(
                    1,
                    open_=case["first"]["open_"],
                    high=case["first"]["high"],
                    low=case["first"]["low"],
                    close=case["first"]["close"],
                    volume=1000,
                    instrument="ES",
                )
                sim.process_candle(
                    candle=c1,
                    session_active=True,
                    blackout_active=False,
                    blackout_reason=None,
                )

                c2 = make_candle(
                    2,
                    open_=case["second"]["open_"],
                    high=case["second"]["high"],
                    low=case["second"]["low"],
                    close=case["second"]["close"],
                    volume=1000,
                    instrument="ES",
                )
                sim.process_candle(
                    candle=c2,
                    session_active=True,
                    blackout_active=False,
                    blackout_reason=None,
                )

                trade = sim.get_trade(trade_id)
                assert trade is not None
                self.assertAlmostEqual(trade.r_multiple, case["expected_r"], places=6)


if __name__ == "__main__":
    unittest.main()
