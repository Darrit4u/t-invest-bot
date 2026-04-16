from __future__ import annotations

import unittest
from dataclasses import replace

from core.trade_simulator import TradeSimulator
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TradeSimulatorRealismCostsTests(unittest.TestCase):
    def test_fixed_tick_slippage_is_applied_to_entry_and_stop_exit(self) -> None:
        sim = TradeSimulator(
            params={
                "trade_simulator": {
                    "revalidate_after_fill": False,
                    "slippage": {
                        "enabled": True,
                        "model": "fixed_ticks",
                        "default": {
                            "entry_ticks": 1.0,
                            "stop_exit_ticks": 1.0,
                        },
                    }
                }
            },
            logger=_DummyLogger(),
            storage=None,
        )
        signal = replace(
            build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0),
            metadata={"source": "test", "tick_size": 0.5},
        )
        events = sim.register_signal(signal, timeframe="1min")
        trade_id = events[0].trade_id

        activation = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0)
        out1 = sim.process_candle(
            candle=activation,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out1], ["activated"])
        self.assertAlmostEqual(float(out1[0].price or 0.0), 100.5, places=6)

        stop = make_candle(2, open_=99.0, high=99.2, low=98.6, close=98.9)
        out2 = sim.process_candle(
            candle=stop,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out2], ["sl_hit"])
        self.assertAlmostEqual(float(out2[0].price or 0.0), 98.5, places=6)

        trade = sim.get_trade(trade_id)
        assert trade is not None
        self.assertLess(float(trade.net_pnl), -2.0)

    def test_execution_section_slippage_is_applied_to_entry_and_target_exit(self) -> None:
        sim = TradeSimulator(
            params={
                "execution": {
                    "slippage": {
                        "enabled": True,
                        "model": "fixed_ticks",
                        "default": {
                            "entry_ticks": 1.0,
                            "target_exit_ticks": 1.0,
                        },
                    }
                },
                "trade_simulator": {
                    "revalidate_after_fill": False,
                    "tp1_size": 0.0,
                },
            },
            logger=_DummyLogger(),
            storage=None,
        )
        signal = replace(
            build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0),
            metadata={"source": "test", "tick_size": 0.5},
        )
        sim.register_signal(signal, timeframe="1hour")

        activation = make_candle(1, open_=100.0, high=100.2, low=99.8, close=100.0, timeframe="1hour")
        out1 = sim.process_candle(
            candle=activation,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out1], ["activated"])
        self.assertAlmostEqual(float(out1[0].price or 0.0), 100.5, places=6)

        target = make_candle(2, open_=102.0, high=102.4, low=101.9, close=102.2, timeframe="1hour")
        out2 = sim.process_candle(
            candle=target,
            session_active=True,
            blackout_active=False,
            blackout_reason=None,
        )
        self.assertEqual([item.event_type for item in out2], ["tp2_hit"])
        self.assertAlmostEqual(float(out2[0].price or 0.0), 101.5, places=6)


if __name__ == "__main__":
    unittest.main()
