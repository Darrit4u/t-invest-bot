"""CLI tool to find no-trade windows from matrix backtest artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import ConfigLoader
from core.instrument_registry import InstrumentRegistry
from core.no_trade_analysis import (
    AnalysisConfig,
    analyze_no_trade_pairs,
    build_no_trade_patch_payload,
    load_closed_trades,
    render_bucket_label,
    resolve_report_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find no-trade windows from backtest trades.csv with train/validation confirmation."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "backtest_reports",
        help="Root folder with matrix_backtest_* reports.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="latest",
        help="Report directory name/path or 'latest'.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "config",
        help="Path to config directory (used for instrument session timezone mapping).",
    )
    parser.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=10,
        help="Minimum closed trades in bucket to evaluate it as candidate.",
    )
    parser.add_argument(
        "--min-winrate-gap-pp",
        type=float,
        default=12.0,
        help="Minimum winrate degradation (percentage points) vs baseline.",
    )
    parser.add_argument(
        "--min-negative-avg-r",
        type=float,
        default=0.0,
        help="Threshold for negative expectancy in R units (bucket avg_r <= value).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.3,
        help="Validation split ratio by time (tail part).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "backtest_reports",
        help="Base output directory for no_trade_<timestamp> artifacts.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="",
        help="Optional instrument filter (e.g. SILVER).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        help="Optional strategy filter.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="Optional profile filter.",
    )
    parser.add_argument(
        "--time-source",
        type=str,
        default="created_at",
        choices=("created_at", "activated_at"),
        help="Trade timestamp source for bucketing.",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also analyze hour+weekday combined buckets.",
    )
    parser.add_argument(
        "--min-trades-per-day",
        type=float,
        default=1.0,
        help="Minimum expected trades/day after applying recommended filters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report_dir = resolve_report_dir(reports_dir=args.reports_dir, report=args.report)
    except Exception as exc:
        print(f"Failed to resolve report: {exc}", file=sys.stderr)
        return 2

    try:
        app_config = ConfigLoader(args.config_dir).load()
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 2

    registry = InstrumentRegistry.from_config(app_config)

    trades_csv = report_dir / "trades.csv"
    signals_csv = report_dir / "signals.csv"
    loaded = load_closed_trades(
        trades_csv_path=trades_csv,
        app_config=app_config,
        registry=registry,
        time_source=args.time_source,
        instrument_filter=args.instrument,
        strategy_filter=args.strategy,
        profile_filter=args.profile,
    )
    if not loaded.trades:
        print("No closed trades found after filtering.", file=sys.stderr)
        return 2

    config = AnalysisConfig(
        min_trades_per_bucket=max(1, int(args.min_trades_per_bucket)),
        min_winrate_gap_pp=max(0.0, float(args.min_winrate_gap_pp)),
        min_negative_avg_r=float(args.min_negative_avg_r),
        validation_ratio=float(args.validation_ratio),
        min_expected_trades_per_day=max(0.1, float(args.min_trades_per_day)),
        include_combined_buckets=bool(args.include_combined),
    )
    results = analyze_no_trade_pairs(trades=loaded.trades, config=config)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = args.output_dir / f"no_trade_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_report_dir": str(report_dir),
        "time_source": args.time_source,
        "filters": {
            "instrument": args.instrument or None,
            "strategy": args.strategy or None,
            "profile": args.profile or None,
        },
        "config": {
            "min_trades_per_bucket": config.min_trades_per_bucket,
            "min_winrate_gap_pp": config.min_winrate_gap_pp,
            "min_negative_avg_r": config.min_negative_avg_r,
            "validation_ratio": config.validation_ratio,
            "min_expected_trades_per_day": config.min_expected_trades_per_day,
            "include_combined_buckets": config.include_combined_buckets,
        },
        "input_counters": {
            "trades_rows_total": _count_csv_rows(trades_csv),
            "signals_rows_total": _count_csv_rows(signals_csv) if signals_csv.exists() else 0,
            "trades_used_closed": len(loaded.trades),
            "skipped_not_closed": loaded.skipped_not_closed,
            "skipped_missing_time_source": loaded.skipped_missing_time_source,
            "skipped_parse_errors": loaded.skipped_parse_errors,
        },
        "results": [item.to_dict() for item in results],
        "summary": {
            "pairs_total": len(results),
            "pairs_with_recommendations": sum(
                1
                for item in results
                if item.recommendation.blocked_entry_hours_local
                or item.recommendation.blocked_entry_weekdays_local
                or item.recommendation.blocked_time_windows_local
            ),
        },
    }
    (out_dir / "no_trade_windows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    patch_payload = build_no_trade_patch_payload(results)
    patch_yaml = yaml.safe_dump(
        patch_payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    (out_dir / "no_trade_patch.yaml").write_text(patch_yaml, encoding="utf-8")

    summary_md = build_summary_markdown(
        report_dir=report_dir,
        results=results,
        payload=payload,
        patch_yaml=patch_yaml,
    )
    (out_dir / "no_trade_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Analysis complete. Artifacts: {out_dir}")
    print("- no_trade_summary.md")
    print("- no_trade_windows.json")
    print("- no_trade_patch.yaml")
    print("")
    _print_short_recommendations(results)
    return 0


def build_summary_markdown(
    *,
    report_dir: Path,
    results: tuple[Any, ...],
    payload: dict[str, Any],
    patch_yaml: str,
) -> str:
    lines: list[str] = []
    lines.append("# No-Trade Window Analysis")
    lines.append("")
    lines.append(f"- Source report: `{report_dir}`")
    lines.append(f"- Generated at (UTC): `{payload['generated_at_utc']}`")
    lines.append(f"- Time source: `{payload['time_source']}`")
    lines.append(f"- Pairs analyzed: `{payload['summary']['pairs_total']}`")
    lines.append(
        f"- Pairs with recommendations: `{payload['summary']['pairs_with_recommendations']}`"
    )
    lines.append("")
    lines.append("## Input Counters")
    lines.append("")
    for key, value in payload["input_counters"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## Pair Results")
    lines.append("")
    if not results:
        lines.append("No pairs were eligible for analysis.")
    for item in results:
        lines.append(f"### {item.instrument} + {item.strategy}")
        lines.append("")
        lines.append(f"- Timezone: `{item.timezone}`")
        lines.append(f"- Total trades: `{item.trades_total}`")
        lines.append(
            "- Baseline train: "
            f"`n={item.train_metrics.trades_count}` | "
            f"`winrate={item.train_metrics.winrate * 100.0:.2f}%` | "
            f"`avg_r={item.train_metrics.avg_r:.4f}` | "
            f"`net_pnl={item.train_metrics.net_pnl:.4f}` | "
            f"`max_loss_streak={item.train_metrics.max_loss_streak}`"
        )
        lines.append(
            "- Baseline validation: "
            f"`n={item.validation_metrics.trades_count}` | "
            f"`winrate={item.validation_metrics.winrate * 100.0:.2f}%` | "
            f"`avg_r={item.validation_metrics.avg_r:.4f}` | "
            f"`net_pnl={item.validation_metrics.net_pnl:.4f}` | "
            f"`max_loss_streak={item.validation_metrics.max_loss_streak}`"
        )
        lines.append(
            "- Recommendation: "
            f"`hours={list(item.recommendation.blocked_entry_hours_local)}` | "
            f"`weekdays={list(item.recommendation.blocked_entry_weekdays_local)}` | "
            f"`high_risk_filter={item.recommendation.high_risk_filter}` | "
            f"`trades/day {item.recommendation.expected_trades_per_day_before:.2f} -> "
            f"{item.recommendation.expected_trades_per_day_after:.2f}`"
        )
        lines.append("")
        lines.append("Confirmed hour candidates:")
        if item.candidates_hour_local:
            for candidate in item.candidates_hour_local:
                lines.append(
                    "- "
                    f"`{render_bucket_label(candidate.bucket_type, candidate.bucket_value)}` | "
                    f"severity=`{candidate.severity_score:.2f}` | "
                    f"train winrate=`{candidate.train_metrics.winrate * 100.0:.1f}%` | "
                    f"val winrate=`{candidate.validation_metrics.winrate * 100.0:.1f}%` | "
                    f"reasons=`{','.join(candidate.reasons)}`"
                )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Confirmed weekday candidates:")
        if item.candidates_weekday_local:
            for candidate in item.candidates_weekday_local:
                lines.append(
                    "- "
                    f"`{render_bucket_label(candidate.bucket_type, candidate.bucket_value)}` | "
                    f"severity=`{candidate.severity_score:.2f}` | "
                    f"train winrate=`{candidate.train_metrics.winrate * 100.0:.1f}%` | "
                    f"val winrate=`{candidate.validation_metrics.winrate * 100.0:.1f}%` | "
                    f"reasons=`{','.join(candidate.reasons)}`"
                )
        else:
            lines.append("- none")
        lines.append("")
        if item.candidates_hour_weekday_local:
            lines.append("Confirmed combined hour+weekday candidates:")
            for candidate in item.candidates_hour_weekday_local:
                lines.append(
                    "- "
                    f"`{render_bucket_label(candidate.bucket_type, candidate.bucket_value)}` | "
                    f"severity=`{candidate.severity_score:.2f}` | "
                    f"reasons=`{','.join(candidate.reasons)}`"
                )
            lines.append("")

    lines.append("## Config Patch")
    lines.append("")
    lines.append("```yaml")
    lines.append(patch_yaml.rstrip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _print_short_recommendations(results: tuple[Any, ...]) -> None:
    if not results:
        print("No eligible instrument+strategy pairs.")
        return

    with_recommendations = [
        item
        for item in results
        if item.recommendation.blocked_entry_hours_local
        or item.recommendation.blocked_entry_weekdays_local
        or item.recommendation.high_risk_filter
    ]
    if not with_recommendations:
        print("No confirmed no-trade windows were found.")
        return

    print("Recommendations:")
    for item in with_recommendations:
        print(
            f"- {item.instrument}/{item.strategy}: "
            f"hours={list(item.recommendation.blocked_entry_hours_local)}, "
            f"weekdays={list(item.recommendation.blocked_entry_weekdays_local)}, "
            f"high_risk_filter={item.recommendation.high_risk_filter}, "
            f"trades/day={item.recommendation.expected_trades_per_day_before:.2f}"
            f"->{item.recommendation.expected_trades_per_day_after:.2f}"
        )


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        count = -1
        for count, _ in enumerate(reader):
            pass
    return max(0, count)


if __name__ == "__main__":
    raise SystemExit(main())
