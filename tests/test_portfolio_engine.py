from __future__ import annotations

import unittest
from dataclasses import replace

from core.execution_engine import ExecutionEngine
from core.portfolio_engine import PortfolioEngine
from core.trade_simulator import TradeSimulator
from tests.helpers import build_signal


def _with_sizing_meta(signal, **extra):
    return replace(
        signal,
        metadata={
            "source": "test",
            "tick_size": 0.25,
            "tick_value": 12.5,
            "lot": 1,
            "lot_size": 1,
            "min_qty": 1.0,
            "qty_step": 1.0,
            **extra,
        },
    )


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class PortfolioEngineTests(unittest.TestCase):
    def _engine(self) -> PortfolioEngine:
        return PortfolioEngine(
            params={
                "portfolio": {
                    "enabled": True,
                    "max_positions": 1,
                    "max_positions_per_instrument": 1,
                    "max_positions_per_strategy": 1,
                    "max_risk_per_trade_pct": 1.0,
                    "max_total_risk_pct": 1.0,
                    "max_instrument_risk_pct": 1.0,
                    "max_strategy_risk_pct": 1.0,
                    "min_selection_rr": 1.5,
                }
            }
        )

    def test_select_signals_prefers_higher_priority_strategy_over_quality(self) -> None:
        engine = self._engine()
        signal_trend = build_signal(
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            entry=100.0,
            stop_loss=99.5,
            tp1=101.5,
            tp2=102.5,
        )
        signal_trend = _with_sizing_meta(signal_trend, signal_quality_score=0.2)
        signal_breakout = build_signal(
            instrument="SILVER",
            strategy="compression_breakout",
            entry=50.0,
            stop_loss=49.5,
            tp1=51.0,
            tp2=52.0,
        )
        signal_breakout = _with_sizing_meta(signal_breakout, signal_quality_score=0.9)

        result = engine.select_signals(signals=[signal_trend, signal_breakout], open_positions=[])

        self.assertEqual(len(result.accepted_signals), 1)
        self.assertEqual(result.accepted_signals[0].instrument, "ES")
        self.assertEqual(result.accepted_signals[0].strategy, "trend_pullback_vwap_ema")
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "lost_in_priority")
        self.assertEqual(result.rejected[0].details.get("rejection_category"), "priority")

    def test_select_signals_rejects_low_rr_before_allocation(self) -> None:
        engine = self._engine()
        low_rr = build_signal(
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            entry=100.0,
            stop_loss=99.0,
            tp1=101.0,
            tp2=102.0,
        )

        result = engine.select_signals(signals=[low_rr], open_positions=[])

        self.assertEqual(len(result.accepted_signals), 0)
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "low_rr")
        self.assertEqual(result.rejected[0].details.get("rejection_category"), "low_rr")

    def test_select_signals_prefers_diversified_instrument_group(self) -> None:
        engine = PortfolioEngine(
            params={
                "portfolio": {
                    "enabled": True,
                    "max_positions": 2,
                    "max_positions_per_instrument": 1,
                    "max_positions_per_strategy": 2,
                    "max_risk_per_trade_pct": 1.0,
                    "max_total_risk_pct": 2.0,
                    "max_instrument_risk_pct": 1.0,
                    "max_strategy_risk_pct": 2.0,
                    "max_positions_per_correlation_group": 2,
                    "min_selection_rr": 1.5,
                    "correlation_groups": {
                        "energy": ["BRENT", "NG"],
                        "index": ["ES"],
                    },
                }
            }
        )
        brent = _with_sizing_meta(
            build_signal(
                instrument="BRENT",
                strategy="trend_pullback_vwap_ema",
                entry=100.0,
                stop_loss=99.0,
                tp1=102.0,
                tp2=103.0,
            ),
            signal_quality_score=0.8,
        )
        ng = _with_sizing_meta(
            build_signal(
                instrument="NG",
                strategy="trend_pullback_vwap_ema",
                entry=10.0,
                stop_loss=9.9,
                tp1=10.2,
                tp2=10.3,
            ),
            signal_quality_score=0.82,
        )
        es = _with_sizing_meta(
            build_signal(
                instrument="ES",
                strategy="trend_pullback_vwap_ema",
                entry=200.0,
                stop_loss=198.5,
                tp1=203.5,
                tp2=205.0,
            ),
            signal_quality_score=0.75,
        )

        result = engine.select_signals(signals=[brent, ng, es], open_positions=[])

        self.assertEqual(len(result.accepted_signals), 2)
        accepted_instruments = {item.instrument for item in result.accepted_signals}
        self.assertIn("ES", accepted_instruments)
        self.assertEqual(len([item for item in accepted_instruments if item in {"BRENT", "NG"}]), 1)

    def test_submit_for_execution_returns_normalized_portfolio_events(self) -> None:
        portfolio_engine = self._engine()
        simulator = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)
        execution_engine = ExecutionEngine(simulator=simulator)
        signal = build_signal(
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            entry=100.0,
            stop_loss=99.0,
            tp1=102.0,
            tp2=103.0,
            entry_mode="NEXT_BAR_OPEN",
        )
        signal = _with_sizing_meta(signal)

        selection = portfolio_engine.select_signals(signals=[signal], open_positions=[])
        submitted = portfolio_engine.submit_for_execution(
            signals=selection.accepted_signals,
            execution_engine=execution_engine,
            timeframe="1min",
        )

        self.assertEqual(len(submitted.execution_events), 1)
        self.assertGreaterEqual(len(submitted.portfolio_events), 1)
        trade = execution_engine.get_trade_record(submitted.execution_events[0].trade_id)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertGreater(trade.size, 1.0)
        event_types = {item.event_type for item in submitted.portfolio_events}
        self.assertIn("position_updated", event_types)


if __name__ == "__main__":
    unittest.main()
