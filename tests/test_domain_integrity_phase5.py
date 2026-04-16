from __future__ import annotations

import unittest

from core.trade_simulator import TradeSimulator
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass


class DomainIntegrityPhase5Tests(unittest.TestCase):
    def test_position_to_trade_lifecycle_integrity(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0, entry_mode="NEXT_BAR_OPEN")

        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        position = sim.get_position(trade_id)
        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.signal_id, signal.signal_id)
        self.assertEqual(position.status, "activated")

        sim.process_candle(
            candle=make_candle(2, open_=99.0, high=99.2, low=98.7, close=98.8),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        trade = sim.get_trade_record(trade_id)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertEqual(trade.signal_id, signal.signal_id)
        self.assertEqual(trade.status, "sl_hit")
        self.assertIsNotNone(trade.closed_at)


if __name__ == "__main__":
    unittest.main()
