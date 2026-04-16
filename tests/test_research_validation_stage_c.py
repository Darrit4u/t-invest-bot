from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.backtest_matrix import PortfolioRunResult
from core.research_validation import (
    ResearchConfig,
    SensitivityConfig,
    SplitConfig,
    WalkForwardConfig,
    analyze_portfolio,
    run_backtest,
    run_sensitivity,
    run_walk_forward,
)
from tests.helpers import make_candle


def _fake_runner(**kwargs) -> PortfolioRunResult:
    start = kwargs["report_start_utc"]
    end = kwargs["report_end_utc"]
    params = kwargs.get("params", {})
    alpha = float(params.get("alpha", 1.0))
    bars = max(1, int((end - start).total_seconds() // 60) + 1)
    net_pnl = alpha * bars
    trade = {
        "trade_id": f"{int(start.timestamp())}",
        "instrument": "ES",
        "strategy": "trend_pullback_vwap_ema",
        "opened_at": start.isoformat(),
        "closed_at": end.isoformat(),
        "net_pnl": net_pnl,
        "r_multiple": net_pnl / 2.0,
    }
    return PortfolioRunResult(
        profile=str(kwargs.get("profile", "test")),
        status="ok",
        error=None,
        metrics={"net_pnl": net_pnl},
        signals=[],
        trades=[trade],
        events=[],
        portfolio_events=[],
    )


def _candles(count: int) -> dict[str, list]:
    return {
        "ES": [
            make_candle(
                i,
                open_=100.0 + i * 0.1,
                high=100.2 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.1 + i * 0.1,
                instrument="ES",
                timeframe="1min",
            )
            for i in range(count)
        ]
    }


class StageCResearchTests(unittest.TestCase):
    def test_run_backtest_respects_chronological_split(self) -> None:
        candles = _candles(40)
        config = ResearchConfig(
            profile="test",
            candles_by_instrument=candles,
            params={"alpha": 1.0},
            timeframe="1min",
            split=SplitConfig(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25),
            runner=_fake_runner,
        )
        result = run_backtest(config)

        train = result["split_windows"]["train"]
        validation = result["split_windows"]["validation"]
        test = result["split_windows"]["test"]
        train_end = datetime.fromisoformat(train["end_utc"])
        validation_start = datetime.fromisoformat(validation["start_utc"])
        validation_end = datetime.fromisoformat(validation["end_utc"])
        test_start = datetime.fromisoformat(test["start_utc"])

        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        self.assertEqual(result["strategy_bucket"], "working")
        self.assertIn("train_test_ratio", result["overfitting"])

    def test_run_walk_forward_builds_expected_folds(self) -> None:
        candles = _candles(30)
        config = ResearchConfig(
            profile="test",
            candles_by_instrument=candles,
            params={"alpha": 1.0},
            timeframe="1min",
            walk_forward=WalkForwardConfig(train_bars=10, test_bars=5, step_bars=5, min_folds=4),
            runner=_fake_runner,
        )
        result = run_walk_forward(config)
        folds = result["folds"]
        self.assertEqual(len(folds), 4)
        for fold in folds:
            train_end = datetime.fromisoformat(fold["train_window"]["end_utc"])
            test_start = datetime.fromisoformat(fold["test_window"]["start_utc"])
            self.assertLess(train_end, test_start)
        self.assertGreater(result["summary"]["positive_fold_ratio"], 0.0)

    def test_run_sensitivity_covers_grid_and_ranks(self) -> None:
        candles = _candles(24)
        config = ResearchConfig(
            profile="test",
            candles_by_instrument=candles,
            params={"alpha": 1.0},
            timeframe="1min",
            split=SplitConfig(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25),
            sensitivity=SensitivityConfig(
                param_grid={"alpha": (-1.0, 0.5, 2.0)},
                target_metric="net_pnl",
                use_test_window=False,
            ),
            runner=_fake_runner,
        )
        result = run_sensitivity(config)
        self.assertEqual(result["summary"]["runs"], 3)
        self.assertEqual(result["summary"]["best_params"]["alpha"], 2.0)
        score = float(result["summary"]["parameter_robustness_score"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_portfolio_reports_distribution_and_drawdown(self) -> None:
        base = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        trades = [
            {
                "trade_id": "1",
                "instrument": "ES",
                "strategy": "s1",
                "opened_at": base.isoformat(),
                "closed_at": base.replace(minute=5).isoformat(),
                "net_pnl": -1.0,
            },
            {
                "trade_id": "2",
                "instrument": "ES",
                "strategy": "s1",
                "opened_at": base.replace(minute=10).isoformat(),
                "closed_at": base.replace(minute=15).isoformat(),
                "net_pnl": -2.0,
            },
            {
                "trade_id": "3",
                "instrument": "NG",
                "strategy": "s2",
                "opened_at": base.replace(minute=20).isoformat(),
                "closed_at": base.replace(minute=25).isoformat(),
                "net_pnl": 3.0,
            },
            {
                "trade_id": "4",
                "instrument": "NG",
                "strategy": "s2",
                "opened_at": base.replace(minute=30).isoformat(),
                "closed_at": base.replace(minute=35).isoformat(),
                "net_pnl": -1.0,
            },
        ]
        analysis = analyze_portfolio(trades, initial_capital=10_000.0)
        metrics = analysis["portfolio_metrics"]
        self.assertEqual(metrics["trades"], 4)
        self.assertEqual(metrics["max_losing_streak"], 2)
        self.assertAlmostEqual(metrics["net_pnl"], -1.0, places=6)
        self.assertIn("by_strategy", analysis["contribution"])
        self.assertIn("drawdown_curve", analysis["equity"])


if __name__ == "__main__":
    unittest.main()
