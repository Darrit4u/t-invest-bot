from __future__ import annotations

import unittest
from datetime import datetime

from core.backtest_matrix import (
    ComboRunResult,
    aggregate_profile_metrics,
    build_combo_tasks,
    build_russian_report,
    parse_local_datetime,
)


class BacktestMatrixUtilsTests(unittest.TestCase):
    def test_parse_local_datetime_supports_date_and_time(self) -> None:
        start = parse_local_datetime("2026-01-05", timezone_name="Europe/Moscow", is_end=False)
        end = parse_local_datetime("2026-01-05", timezone_name="Europe/Moscow", is_end=True)
        self.assertEqual(start.hour, 0)
        self.assertEqual(end.hour, 23)
        self.assertEqual(end.minute, 59)
        self.assertEqual(end.second, 59)

        point = parse_local_datetime("2026-01-05 12:34:56", timezone_name="Europe/Moscow")
        self.assertEqual(point.hour, 12)
        self.assertEqual(point.minute, 34)
        self.assertEqual(point.second, 56)

    def test_build_combo_tasks_builds_full_matrix(self) -> None:
        tasks = build_combo_tasks(
            profiles=["a", "b"],
            strategies=["s1", "s2"],
            instruments=["i1", "i2"],
        )
        self.assertEqual(len(tasks), 8)
        self.assertEqual(tasks[0].profile, "a")
        self.assertEqual(tasks[0].strategy, "s1")
        self.assertEqual(tasks[0].instrument, "i1")

    def test_aggregate_profile_metrics(self) -> None:
        rows = [
            ComboRunResult(
                profile="balanced",
                strategy="x",
                instrument="ES",
                status="ok",
                error=None,
                metrics={
                    "signals": 10,
                    "activated": 8,
                    "closed": 6,
                    "wins": 4,
                    "losses": 2,
                    "net_pnl": 3.2,
                    "gross_pnl": 3.8,
                    "fees": 0.6,
                    "max_drawdown": 1.1,
                    "gross_wins_points": 5.0,
                    "gross_losses_points_abs": 1.8,
                    "open_trades": 1,
                },
                signals=[],
                trades=[],
                events=[],
            ),
            ComboRunResult(
                profile="balanced",
                strategy="y",
                instrument="NG",
                status="ok",
                error=None,
                metrics={
                    "signals": 5,
                    "activated": 5,
                    "closed": 4,
                    "wins": 2,
                    "losses": 2,
                    "net_pnl": -0.5,
                    "gross_pnl": 0.0,
                    "fees": 0.5,
                    "max_drawdown": 0.9,
                    "gross_wins_points": 1.2,
                    "gross_losses_points_abs": 1.7,
                    "open_trades": 0,
                },
                signals=[],
                trades=[],
                events=[],
            ),
        ]
        out = aggregate_profile_metrics(rows)
        self.assertIn("balanced", out)
        bucket = out["balanced"]
        self.assertEqual(bucket["closed"], 10)
        self.assertEqual(bucket["wins"], 6)
        self.assertAlmostEqual(bucket["net_pnl"], 2.7)

    def test_report_contains_profile_block(self) -> None:
        result = ComboRunResult(
            profile="active",
            strategy="compression_breakout",
            instrument="BRENT",
            status="ok",
            error=None,
            metrics={
                "signals": 3,
                "closed": 2,
                "wins": 1,
                "losses": 1,
                "win_rate_pct": 50.0,
                "net_pnl": 1.2,
                "fees": 0.2,
                "profit_factor": 1.3,
                "max_drawdown": 0.4,
                "avg_win_points": 1.1,
                "avg_loss_points_abs": 0.8,
            },
            signals=[],
            trades=[],
            events=[],
        )
        text = build_russian_report(
            period_start_local=datetime(2026, 1, 5, 0, 0, 0),
            period_end_local=datetime(2026, 4, 10, 23, 59, 59),
            timeframe="5min",
            results=[result],
            profile_metrics={
                "active": {
                    "profile": "active",
                    "combos": 1,
                    "signals": 3,
                    "closed": 2,
                    "wins": 1,
                    "losses": 1,
                    "net_pnl": 1.2,
                    "profit_factor": 1.3,
                    "win_rate_pct": 50.0,
                    "avg_drawdown_per_combo": 0.4,
                }
            },
        )
        self.assertIn("СРАВНЕНИЕ ПРОФИЛЕЙ", text)
        self.assertIn("active", text)


if __name__ == "__main__":
    unittest.main()

