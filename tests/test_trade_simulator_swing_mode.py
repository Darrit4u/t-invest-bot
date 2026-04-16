from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.models import SignalDirection
from core.trade_simulator import TradeSimulator, TradeStatus
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TradeSimulatorSwingModeTests(unittest.TestCase):
    def test_swing_keeps_active_position_across_session_end(self) -> None:
        sim = TradeSimulator(
            params={"trading": {"mode": "swing"}, "swing": {"use_session_force_close": False}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=103, tp2=104)
        events = sim.register_signal(signal, timeframe="1hour")
        trade_id = events[0].trade_id

        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        out = sim.process_candle(
            candle=make_candle(2, open_=100.1, high=100.3, low=99.9, close=100.2, instrument="ES", timeframe="1hour"),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual(out, tuple())
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.ACTIVATED)

    def test_swing_expires_by_max_holding_bars(self) -> None:
        sim = TradeSimulator(
            params={
                "trading": {"mode": "swing"},
                "swing": {"max_holding_bars": 2, "max_holding_days": 0},
            },
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=90, tp1=110, tp2=120)
        events = sim.register_signal(signal, timeframe="1hour")
        trade_id = events[0].trade_id

        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        sim.process_candle(
            candle=make_candle(2, open_=100.0, high=100.2, low=99.8, close=100.1, instrument="ES", timeframe="1hour"),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        sim.process_candle(
            candle=make_candle(3, open_=100.1, high=100.2, low=99.9, close=100.1, instrument="ES", timeframe="1hour"),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        out = sim.process_candle(
            candle=make_candle(4, open_=100.1, high=100.2, low=99.9, close=100.1, instrument="ES", timeframe="1hour"),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out], ["expired"])
        self.assertEqual(out[0].payload.get("reason"), "max_holding_bars")
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.EXPIRED)

    def test_swing_expires_by_max_holding_days(self) -> None:
        sim = TradeSimulator(
            params={
                "trading": {"mode": "swing"},
                "swing": {"max_holding_bars": 0, "max_holding_days": 2},
            },
            logger=_DummyLogger(),
            storage=None,
        )
        base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        signal = build_signal(
            direction=SignalDirection.LONG,
            entry=100,
            stop_loss=90,
            tp1=110,
            tp2=120,
            timestamp=base,
        )
        events = sim.register_signal(signal, timeframe="1hour")
        trade_id = events[0].trade_id

        sim.process_candle(
            candle=make_candle(
                1,
                base=base,
                open_=100.0,
                high=100.2,
                low=99.8,
                close=100.0,
                instrument="ES",
                timeframe="1hour",
            ),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        sim.process_candle(
            candle=make_candle(
                24 * 60,
                base=base,
                open_=100.0,
                high=100.2,
                low=99.8,
                close=100.1,
                instrument="ES",
                timeframe="1hour",
            ),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        out = sim.process_candle(
            candle=make_candle(
                2 * 24 * 60 + 1,
                base=base,
                open_=100.1,
                high=100.2,
                low=99.9,
                close=100.1,
                instrument="ES",
                timeframe="1hour",
            ),
            session_active=False,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out], ["expired"])
        self.assertEqual(out[0].payload.get("reason"), "max_holding_days")
        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertEqual(trade.status, TradeStatus.EXPIRED)

    def test_swing_gap_through_stop_uses_open_price(self) -> None:
        sim = TradeSimulator(
            params={"trading": {"mode": "swing"}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=105, tp2=110)
        sim.register_signal(signal, timeframe="1hour")
        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )

        out = sim.process_candle(
            candle=make_candle(2, open_=97.0, high=98.0, low=96.5, close=97.2, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out], ["sl_hit"])
        self.assertAlmostEqual(float(out[0].price or 0.0), 97.0, places=6)

    def test_swing_gap_through_targets_uses_open_price(self) -> None:
        sim = TradeSimulator(
            params={"trading": {"mode": "swing"}, "trade_simulator": {"tp1_size": 0.5}},
            logger=_DummyLogger(),
            storage=None,
        )
        signal = build_signal(direction=SignalDirection.LONG, entry=100, stop_loss=99, tp1=101, tp2=102)
        sim.register_signal(signal, timeframe="1hour")
        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )

        out = sim.process_candle(
            candle=make_candle(2, open_=103.0, high=103.4, low=102.8, close=103.2, instrument="ES", timeframe="1hour"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out], ["tp1_hit", "tp2_hit"])
        self.assertAlmostEqual(float(out[0].price or 0.0), 103.0, places=6)
        self.assertAlmostEqual(float(out[1].price or 0.0), 103.0, places=6)


if __name__ == "__main__":
    unittest.main()
