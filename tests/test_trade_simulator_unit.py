from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
