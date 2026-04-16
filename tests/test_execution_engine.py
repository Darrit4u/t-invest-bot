"""Execution engine tests for Phase 2 separation."""

from __future__ import annotations

import unittest

from core.execution_engine import ExecutionEngine
from core.trade_simulator import TradeSimulator
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class ExecutionEngineTests(unittest.TestCase):
    def test_open_from_signal_creates_position_snapshot(self) -> None:
        simulator = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        engine = ExecutionEngine(simulator=simulator)
        signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0)

        open_result = engine.open_from_signal(signal=signal, timeframe="1min")

        self.assertEqual(len(open_result.events), 1)
        self.assertIsNotNone(open_result.position)
        assert open_result.position is not None
        self.assertEqual(open_result.position.strategy_id, signal.strategy)
        self.assertEqual(open_result.position.status, "waiting_activation")

    def test_process_market_returns_closed_domain_trades(self) -> None:
        simulator = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        engine = ExecutionEngine(simulator=simulator)
        signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0, entry_mode="NEXT_BAR_OPEN")
        engine.open_from_signal(signal=signal, timeframe="1min")

        engine.process_market(
            candle=make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.1, instrument="ES"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        result = engine.process_market(
            candle=make_candle(2, open_=100.0, high=100.2, low=98.8, close=99.1, instrument="ES"),
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )

        self.assertGreaterEqual(len(result.events), 1)
        self.assertEqual(len(result.closed_trades), 1)
        trade = result.closed_trades[0]
        self.assertEqual(trade.status, "sl_hit")
        self.assertLess(trade.pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
