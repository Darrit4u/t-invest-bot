from __future__ import annotations

import unittest

from core.models import MarketRegime, SignalDirection
from core.trade_simulator import TradeSimulator, TradeStatus
from storage.memory_store import MemoryCandleStore
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TradeEdgeCaseTests(unittest.TestCase):
    def test_next_bar_open_does_not_activate_on_signal_candle(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        same_candle = make_candle(0, open_=99.5, high=100.4, low=99.4, close=100.2)
        out_same = sim.process_candle(
            candle=same_candle,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual(out_same, tuple())

        next_candle = make_candle(1, open_=100.3, high=100.7, low=100.1, close=100.6)
        out_next = sim.process_candle(
            candle=next_candle,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([e.event_type for e in out_next], ["activated"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertAlmostEqual(trade.entry_fill_price, 100.3, places=6)

    def test_duplicate_and_partial_candles_update_store(self) -> None:
        store = MemoryCandleStore(history_depth=20)
        c1 = make_candle(0, open_=100, close=100.2, instrument="ES")
        c1_partial = make_candle(0, open_=100, close=100.4, high=100.6, low=99.9, instrument="ES")

        self.assertEqual(store.upsert(c1), "inserted")
        self.assertEqual(store.upsert(c1_partial), "updated")
        latest = store.latest("ES", "1min")
        assert latest is not None
        self.assertAlmostEqual(latest.close, 100.4)

    def test_gap_through_entry_activates_on_open_price(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        sim.register_signal(signal, timeframe="1min")

        gap_up = make_candle(1, open_=100.5, high=100.8, low=100.4, close=100.7)
        events = sim.process_candle(candle=gap_up, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([e.event_type for e in events], ["activated", "expired"])

        trade = sim.get_trade(events[0].trade_id)
        assert trade is not None
        self.assertAlmostEqual(trade.entry_fill_price, 100.5, places=6)
        self.assertEqual(trade.status, TradeStatus.EXPIRED)
        self.assertEqual(trade.exit_reason, "poor_rr_after_fill")

    def test_gap_through_stop_closes_trade(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        sim.register_signal(signal, timeframe="1min")

        activate = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)
        gap_down = make_candle(2, open_=98.2, high=98.6, low=97.9, close=98.3)
        events = sim.process_candle(candle=gap_down, session_active=True, blackout_active=False, blackout_reason=None)
        self.assertEqual([e.event_type for e in events], ["sl_hit"])

    def test_gap_through_tp_hits_tp2(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {"tp1_size": 0.5}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        sim.register_signal(signal, timeframe="1min")

        activate = make_candle(1, open_=100.0, high=100.1, low=99.9, close=100.0)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)
        gap_tp = make_candle(2, open_=103.0, high=103.5, low=102.7, close=103.2)
        events = sim.process_candle(candle=gap_tp, session_active=True, blackout_active=False, blackout_reason=None)
        # simulator emits tp1 then tp2 in same candle when both levels are passed
        self.assertEqual([e.event_type for e in events], ["tp1_hit", "tp2_hit"])

    def test_blackout_during_waiting_activation_cancels_trade(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        idle = make_candle(1, open_=98.0, high=98.2, low=97.9, close=98.1)
        out = sim.process_candle(candle=idle, session_active=True, blackout_active=True, blackout_reason="CPI")
        self.assertEqual([e.event_type for e in out], ["cancelled_by_news"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_NEWS)

    def test_blackout_during_active_trade_can_cancel_when_configured(self) -> None:
        sim = TradeSimulator(
            params={"trade_simulator": {"close_active_on_blackout": True}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        blocked = make_candle(2, open_=100.1, high=100.3, low=99.9, close=100.0)
        out = sim.process_candle(candle=blocked, session_active=True, blackout_active=True, blackout_reason="NFP")
        self.assertEqual([e.event_type for e in out], ["cancelled_by_news"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_NEWS)

    def test_session_end_during_active_trade_cancels(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.SHORT, entry=100, stop_loss=101, tp1=99, tp2=98)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100.0, high=100.1, low=99.7, close=99.9)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        closing = make_candle(2, open_=100.0, high=100.4, low=99.8, close=100.2)
        out = sim.process_candle(candle=closing, session_active=False, blackout_active=False, blackout_reason=None)
        self.assertEqual([e.event_type for e in out], ["cancelled_by_session_end"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_SESSION_END)

    def test_session_end_closes_only_profitable_when_configured(self) -> None:
        sim = TradeSimulator(
            params={"trade_simulator": {"close_profitable_on_session_end": True}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        session_end_profit = make_candle(2, open_=100.5, high=100.7, low=100.3, close=100.6)
        out = sim.process_candle(
            candle=session_end_profit,
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([e.event_type for e in out], ["cancelled_by_session_end"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.CANCELLED_BY_SESSION_END)

    def test_session_end_keeps_loser_open_when_configured(self) -> None:
        sim = TradeSimulator(
            params={"trade_simulator": {"close_profitable_on_session_end": True}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        session_end_loss = make_candle(2, open_=99.6, high=99.8, low=99.2, close=99.4)
        out = sim.process_candle(
            candle=session_end_loss,
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual(out, tuple())
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.ACTIVATED)

    def test_missing_candles_after_reconnect_keeps_lifecycle_valid(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activate = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1)
        sim.process_candle(candle=activate, session_active=True, blackout_active=False, blackout_reason=None)

        # Simulate feed reconnect gap: next candle appears much later.
        reconnect_gap_candle = make_candle(100, open_=101.5, high=102.5, low=101.4, close=102.2)
        out = sim.process_candle(
            candle=reconnect_gap_candle,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([e.event_type for e in out], ["tp1_hit", "tp2_hit"])
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.TP2_HIT)


if __name__ == "__main__":
    unittest.main()
