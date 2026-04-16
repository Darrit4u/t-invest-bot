from __future__ import annotations

import copy
import unittest
from datetime import timezone
from unittest.mock import patch

from core.backtest_matrix import run_portfolio_backtest
from core.config_loader import ConfigLoader
from core.models import MarketRegime, SignalDirection, StrategySignal
from tests.helpers import config_dir, make_candle


class _DummyAlwaysSignalStrategy:
    name = "trend_pullback_vwap_ema"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def generate_signals(self, context):
        if len(context.candles) < 20:
            return []
        close = float(context.candles[-1].close)
        return [
            StrategySignal(
                signal_id=f"{context.instrument.symbol}-{int(context.candles[-1].datetime.timestamp())}",
                instrument=context.instrument.symbol,
                strategy=self.name,
                regime=MarketRegime.TREND,
                direction=SignalDirection.LONG,
                timestamp=context.candles[-1].datetime,
                entry_mode="NEXT_BAR_OPEN",
                entry=close,
                stop_loss=close - 0.6,
                tp1=close + 1.0,
                tp2=close + 1.8,
                metadata={"signal_quality_score": 0.8},
            )
        ]


class PortfolioBacktestFlowTests(unittest.TestCase):
    def test_run_portfolio_backtest_multi_instrument(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        params = copy.deepcopy(cfg.params)
        params["portfolio"] = {
            "enabled": True,
            "max_positions": 2,
            "max_positions_per_instrument": 1,
            "max_positions_per_strategy": 2,
            "max_risk_per_trade_pct": 1.0,
            "max_total_risk_pct": 2.0,
            "max_instrument_risk_pct": 1.0,
            "max_strategy_risk_pct": 2.0,
            "max_positions_per_correlation_group": 1,
            "correlation_groups": {
                "energy": ["BRENT", "NG"],
                "index": ["ES"],
            },
        }
        params["signal_filter"] = {
            "commission_roundtrip": 0.0008,
            "safety_multiplier": 1.2,
            "min_rr_after_fill": 0.1,
            "min_signal_quality_score": 0.0,
            "trend_context_score_min": 0.0,
        }
        params["trade_simulator"] = {
            "commission_per_side": 0.0001,
            "max_wait_bars": 4,
            "max_trade_bars": 50,
            "move_stop_to_breakeven": True,
            "intrabar_stop_priority": True,
        }
        params["strategy_params"] = {
            "by_instrument": {
                "ES": {"trend_pullback_vwap_ema": {"enabled": True}},
                "BRENT": {"trend_pullback_vwap_ema": {"enabled": True}},
            }
        }

        candles_by_instrument = {
            "ES": [
                make_candle(
                    i,
                    open_=100.0 + (i * 0.2),
                    high=100.3 + (i * 0.2),
                    low=99.9 + (i * 0.2),
                    close=100.1 + (i * 0.2),
                    instrument="ES",
                    timeframe="1min",
                )
                for i in range(60)
            ],
            "BRENT": [
                make_candle(
                    i,
                    open_=80.0 + (i * 0.15),
                    high=80.25 + (i * 0.15),
                    low=79.85 + (i * 0.15),
                    close=80.1 + (i * 0.15),
                    instrument="BRENT",
                    timeframe="1min",
                )
                for i in range(60)
            ],
        }

        report_start = candles_by_instrument["ES"][10].datetime.astimezone(timezone.utc)
        report_end = candles_by_instrument["ES"][-1].datetime.astimezone(timezone.utc)

        with patch(
            "core.signal_engine.SignalEngine._build_strategy_classes",
            return_value={"trend_pullback_vwap_ema": _DummyAlwaysSignalStrategy},
        ):
            result = run_portfolio_backtest(
                profile="test_profile",
                candles_by_instrument=candles_by_instrument,
                app_config=cfg,
                params=params,
                timeframe="1min",
                report_start_utc=report_start,
                report_end_utc=report_end,
                selected_instruments=["ES", "BRENT"],
                selected_strategies=["trend_pullback_vwap_ema"],
            )

        self.assertEqual(result.status, "ok")
        self.assertGreater(result.metrics.get("signals", 0), 0)
        self.assertIn("exposure_by_instrument", result.metrics)
        self.assertIn("exposure_by_strategy", result.metrics)
        self.assertGreaterEqual(len(result.portfolio_events), 1)


if __name__ == "__main__":
    unittest.main()
