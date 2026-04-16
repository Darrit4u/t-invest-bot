"""Regression tests for Phase-1 explicit domain entities."""

from __future__ import annotations

import unittest

from core.models import SignalDirection, StrategySignal
from core.trade_simulator import TradeSimulator
from domain.models import Signal
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class DomainModelsPhase1Tests(unittest.TestCase):
    def test_strategy_signal_alias_keeps_compat_fields_and_domain_props(self) -> None:
        signal = build_signal(direction=SignalDirection.LONG)
        self.assertIsInstance(signal, StrategySignal)
        self.assertIsInstance(signal, Signal)
        self.assertEqual(signal.strategy_id, signal.strategy)
        self.assertEqual(signal.side, signal.direction)
        self.assertEqual(signal.entry_price, signal.entry)
        self.assertEqual(signal.take_profit, signal.tp1)

    def test_trade_simulator_exports_position_and_trade_records(self) -> None:
        sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0, entry_mode="NEXT_BAR_OPEN")
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        open_position = sim.get_position(trade_id)
        assert open_position is not None
        self.assertEqual(open_position.instrument, signal.instrument)
        self.assertEqual(open_position.strategy_id, signal.strategy)
        self.assertEqual(open_position.status, "waiting_activation")

        sim.process_candle(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1, instrument="ES"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        sim.process_candle(
            candle=make_candle(2, open_=100.0, high=100.2, low=98.8, close=99.1, instrument="ES"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )

        closed_position = sim.get_position(trade_id)
        self.assertIsNone(closed_position)

        trade_record = sim.get_trade_record(trade_id)
        assert trade_record is not None
        self.assertEqual(trade_record.strategy_id, signal.strategy)
        self.assertEqual(trade_record.status, "sl_hit")
        self.assertEqual(trade_record.exit_price, 99.0)
        self.assertLess(trade_record.pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
