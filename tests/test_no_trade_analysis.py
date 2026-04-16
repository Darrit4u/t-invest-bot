from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import TestCase

from core.no_trade_analysis import (
    AnalysisConfig,
    TradeSample,
    analyze_no_trade_pairs,
    build_pair_recommendation,
    compute_trade_metrics,
)


def _make_trade(*, day: int, hour: int, net: float, r: float) -> TradeSample:
    ts = datetime(2026, 3, 1, hour, 0, tzinfo=timezone.utc) + timedelta(days=day)
    return TradeSample(
        profile="active",
        strategy="trend_pullback_vwap_ema",
        instrument="SILVER",
        trade_id=f"{day}-{hour}-{net}",
        timestamp_utc=ts,
        timestamp_local=ts,
        local_hour=hour,
        local_weekday=ts.weekday(),
        local_date=ts.strftime("%Y-%m-%d"),
        net_pnl=net,
        r_multiple=r,
    )


class NoTradeAnalysisTests(TestCase):
    def test_compute_trade_metrics_max_loss_streak(self) -> None:
        trades = [
            _make_trade(day=0, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=0, hour=11, net=-0.5, r=-0.5),
            _make_trade(day=0, hour=12, net=0.7, r=0.7),
            _make_trade(day=0, hour=13, net=-0.4, r=-0.4),
        ]
        metrics = compute_trade_metrics(trades)
        self.assertEqual(metrics.trades_count, 4)
        self.assertEqual(metrics.wins, 1)
        self.assertEqual(metrics.losses, 3)
        self.assertEqual(metrics.max_loss_streak, 2)

    def test_analyze_no_trade_pairs_finds_confirmed_hour(self) -> None:
        trades = [
            _make_trade(day=0, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=0, hour=12, net=1.0, r=1.0),
            _make_trade(day=1, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=1, hour=12, net=1.0, r=1.0),
            _make_trade(day=2, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=2, hour=12, net=1.0, r=1.0),
            _make_trade(day=3, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=3, hour=12, net=1.0, r=1.0),
            _make_trade(day=4, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=4, hour=12, net=1.0, r=1.0),
            _make_trade(day=5, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=5, hour=12, net=1.0, r=1.0),
        ]
        config = AnalysisConfig(
            min_trades_per_bucket=2,
            min_winrate_gap_pp=10.0,
            min_negative_avg_r=0.0,
            validation_ratio=0.33,
            min_expected_trades_per_day=1.0,
            include_combined_buckets=False,
        )
        results = analyze_no_trade_pairs(trades=tuple(trades), config=config)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.candidates_hour_local)
        self.assertEqual(int(result.candidates_hour_local[0].bucket_value), 10)
        self.assertEqual(list(result.recommendation.blocked_entry_hours_local), [10])

    def test_recommendation_marks_high_risk_when_frequency_drops(self) -> None:
        trades = [
            _make_trade(day=0, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=0, hour=12, net=1.0, r=1.0),
            _make_trade(day=1, hour=10, net=-1.0, r=-1.0),
            _make_trade(day=1, hour=12, net=1.0, r=1.0),
        ]
        config = AnalysisConfig(min_trades_per_bucket=1, validation_ratio=0.5)
        results = analyze_no_trade_pairs(trades=tuple(trades), config=config)
        self.assertEqual(len(results), 1)
        candidate = results[0].candidates_hour_local[0]
        recommendation = build_pair_recommendation(
            pair_trades=trades,
            hour_candidates=(candidate,),
            weekday_candidates=tuple(),
            min_expected_trades_per_day=1.5,
        )
        self.assertEqual(list(recommendation.blocked_entry_hours_local), [])
        self.assertTrue(recommendation.high_risk_filter)
