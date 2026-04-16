from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.execution_engine import ExecutionEngine
from core.portfolio_engine import PortfolioEngine
from core.stats_engine import StatsEngine
from core.trade_simulator import TradeSimulator
from tests.helpers import build_signal, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


class SwingSimulationFlowTests(unittest.TestCase):
    def test_execution_and_stats_support_multi_day_swing_trade(self) -> None:
        simulator = TradeSimulator(
            params={
                "trading": {"mode": "swing"},
                "swing": {"max_holding_days": 1, "max_holding_bars": 0},
            },
            logger=_DummyLogger(),
            storage=None,
        )
        execution = ExecutionEngine(simulator=simulator)
        portfolio = PortfolioEngine(params={"portfolio": {"enabled": True}})
        stats = StatsEngine()

        signal = build_signal(entry=100, stop_loss=90, tp1=110, tp2=120)
        open_result = execution.open_from_signal(signal=signal, timeframe="1hour")
        for event in portfolio.normalize_execution_events(
            execution_events=open_result.events,
            execution_engine=execution,
        ):
            stats.record_portfolio_event(event)

        base = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)
        first = execution.process_market(
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
        for event in portfolio.normalize_execution_events(
            execution_events=first.events,
            execution_engine=execution,
        ):
            stats.record_portfolio_event(event)

        second = execution.process_market(
            candle=make_candle(
                2 * 24 * 60,
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
        for event in portfolio.normalize_execution_events(
            execution_events=second.events,
            execution_engine=execution,
        ):
            stats.record_portfolio_event(event)
        for trade in second.closed_trades:
            stats.record_trade_closed(trade)

        summary = stats.summary()["global"]
        self.assertGreaterEqual(int(summary["closed"]), 1)
        self.assertGreaterEqual(float(summary["expectancy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
