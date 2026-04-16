"""Run matrix backtest profile x strategy x instrument on T-Invest historical candles."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_matrix import (
    DEFAULT_STRATEGIES,
    ComboRunResult,
    aggregate_profile_metrics,
    build_combo_tasks,
    build_russian_report,
    compute_warmup_start,
    default_json_payload,
    fetch_historical_candles_tinvest,
    load_profile_params,
    parse_local_datetime,
    run_combo_backtest,
    split_csv_items,
)
from core.config_loader import ConfigLoader
from core.instrument_registry import InstrumentRegistry


DEFAULT_PROFILES = ("conservative", "balanced", "active")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matrix backtest for MOEX futures: profile x strategy x instrument"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config",
        help="Path to config directory with instruments.yaml/strategies.yaml/profiles/",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
        help="Path to .env with INVEST_TOKEN",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Explicit INVEST_TOKEN value (has priority over .env)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-01-05",
        help="Start of test period (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]) in local timezone",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-04-10",
        help="End of test period (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]) in local timezone",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5min",
        help="Historical timeframe (default: 5min)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated profile names from config/profiles, e.g. conservative,balanced,active",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategy names",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default="",
        help="Comma-separated instrument symbols. Empty means all enabled instruments from config",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel worker threads for combo simulation",
    )
    parser.add_argument(
        "--api-limit",
        type=int,
        default=1000,
        help="Max candles per one API call",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Chunk size in days for historical API loading",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "backtest_reports",
        help="Directory where report artifacts will be stored",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and not value.startswith(("'", '"')) and "#" in value:
            value = value.split("#", 1)[0].strip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_token(*, args: argparse.Namespace) -> str:
    load_env_file(args.env_file)
    if args.token.strip():
        return args.token.strip()
    return os.getenv("INVEST_TOKEN", "").strip()


def ensure_selected_instruments(
    *,
    available_enabled: list[str],
    requested: list[str],
) -> list[str]:
    if not requested:
        return sorted(available_enabled)

    unknown = [name for name in requested if name not in available_enabled]
    if unknown:
        raise ValueError(f"Unknown or disabled instruments: {', '.join(unknown)}")
    return sorted(requested)


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _eta_seconds(*, done: int, total: int, elapsed_seconds: float) -> float:
    if done <= 0 or total <= done:
        return 0.0
    rate = done / max(elapsed_seconds, 1e-9)
    return (total - done) / max(rate, 1e-9)


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (done / total) * 100.0


def _combo_label(*, profile: str, strategy: str, instrument: str) -> str:
    return f"{profile}/{strategy}/{instrument}"


async def load_all_history(
    *,
    token: str,
    registry: InstrumentRegistry,
    instruments: list[str],
    timeframe: str,
    warmup_start_utc: datetime,
    report_end_utc: datetime,
    api_limit: int,
    chunk_days: int,
) -> dict[str, list[Any]]:
    async def _fetch_for_symbol(symbol: str) -> tuple[str, list[Any], Exception | None]:
        meta = registry.get(symbol)
        try:
            rows = await fetch_historical_candles_tinvest(
                token=token,
                instrument_uid=meta.uid,
                instrument_figi=meta.figi,
                instrument_symbol=symbol,
                timeframe=timeframe,
                start_utc=warmup_start_utc,
                end_utc=report_end_utc,
                api_limit=api_limit,
                chunk_days=chunk_days,
            )
            return symbol, rows, None
        except Exception as exc:
            return symbol, [], exc

    started = time.monotonic()
    tasks = [asyncio.create_task(_fetch_for_symbol(symbol), name=f"history:{symbol}") for symbol in instruments]
    total = len(tasks)
    done = 0
    out: dict[str, list[Any]] = {}

    print(f"[history] started: {total} instrument(s)")
    for task in asyncio.as_completed(tasks):
        symbol, rows, error = await task
        done += 1
        elapsed = time.monotonic() - started
        eta = _eta_seconds(done=done, total=total, elapsed_seconds=elapsed)
        if error is not None:
            print(
                f"[history] {done}/{total} ({_pct(done, total):.1f}%) "
                f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} | {symbol} -> error: {error}"
            )
            raise error
        out[symbol] = rows
        print(
            f"[history] {done}/{total} ({_pct(done, total):.1f}%) "
            f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} | {symbol} -> candles={len(rows)}"
        )
    return out


def write_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def combo_summary_row(result: ComboRunResult) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "profile": result.profile,
        "strategy": result.strategy,
        "instrument": result.instrument,
        "status": result.status,
        "error": result.error or "",
        "signals": metrics.get("signals", 0),
        "closed": metrics.get("closed", 0),
        "wins": metrics.get("wins", 0),
        "losses": metrics.get("losses", 0),
        "win_rate_pct": metrics.get("win_rate_pct", 0.0),
        "loss_rate_pct": metrics.get("loss_rate_pct", 0.0),
        "trading_days": metrics.get("trading_days", 0),
        "trades_per_day": metrics.get("trades_per_day", 0.0),
        "net_pnl": metrics.get("net_pnl", 0.0),
        "gross_pnl": metrics.get("gross_pnl", 0.0),
        "fees": metrics.get("fees", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "expectancy": metrics.get("expectancy", 0.0),
        "avg_r": metrics.get("avg_r", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "avg_win_points": metrics.get("avg_win_points", 0.0),
        "avg_loss_points_abs": metrics.get("avg_loss_points_abs", 0.0),
        "long_closed": metrics.get("long_closed", 0),
        "short_closed": metrics.get("short_closed", 0),
        "hour_distribution_json": json.dumps(metrics.get("hour_distribution", {}), ensure_ascii=False),
        "weekday_distribution_json": json.dumps(metrics.get("weekday_distribution", {}), ensure_ascii=False),
        "exit_reason_breakdown_json": json.dumps(
            metrics.get("exit_reason_breakdown", {}),
            ensure_ascii=False,
        ),
        "entry_mode_breakdown_json": json.dumps(
            metrics.get("entry_mode_breakdown", {}),
            ensure_ascii=False,
        ),
        "reason_code_expectancy_json": json.dumps(
            metrics.get("reason_code_expectancy", {}),
            ensure_ascii=False,
        ),
        "setup_quality_expectancy_json": json.dumps(
            metrics.get("setup_quality_expectancy", {}),
            ensure_ascii=False,
        ),
        "signal_quality_expectancy_json": json.dumps(
            metrics.get("signal_quality_expectancy", {}),
            ensure_ascii=False,
        ),
        "open_trades": metrics.get("open_trades", 0),
    }


def _expectancy_table(
    *,
    rows: list[dict[str, Any]],
    bucket_field: str,
    bucket_name: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        if not str(row.get("closed_at", "")).strip():
            continue
        raw_bucket = row.get(bucket_field, "")
        bucket_value = str(raw_bucket)
        if bucket_value == "":
            continue
        net = float(row.get("net_pnl", 0.0) or 0.0)
        bucket = buckets.setdefault(
            bucket_value,
            {
                "closed": 0.0,
                "wins": 0.0,
                "losses": 0.0,
                "net_pnl": 0.0,
                "gross_wins": 0.0,
                "gross_losses_abs": 0.0,
            },
        )
        bucket["closed"] += 1.0
        bucket["net_pnl"] += net
        if net >= 0:
            bucket["wins"] += 1.0
            bucket["gross_wins"] += net
        else:
            bucket["losses"] += 1.0
            bucket["gross_losses_abs"] += abs(net)

    def _sort_key(item: tuple[str, dict[str, float]]) -> tuple[int, str]:
        key, _ = item
        try:
            return int(key), key
        except ValueError:
            return 10_000, key

    output: list[dict[str, Any]] = []
    for bucket_value, data in sorted(buckets.items(), key=_sort_key):
        closed = int(data["closed"])
        wins = int(data["wins"])
        losses = int(data["losses"])
        gross_losses_abs = float(data["gross_losses_abs"])
        profit_factor = (float(data["gross_wins"]) / gross_losses_abs) if gross_losses_abs > 0 else 0.0
        output.append(
            {
                bucket_name: bucket_value,
                "closed": closed,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / closed * 100.0) if closed else 0.0,
                "net_pnl": float(data["net_pnl"]),
                "expectancy": (float(data["net_pnl"]) / closed) if closed else 0.0,
                "profit_factor": profit_factor,
            }
        )
    return output


def _metadata_expectancy_table(
    *,
    rows: list[dict[str, Any]],
    bucket_name: str,
    resolve_buckets: Any,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        if not str(row.get("closed_at", "")).strip():
            continue
        metadata = _safe_trade_metadata(row)
        bucket_values = resolve_buckets(metadata)
        if not bucket_values:
            continue

        net = float(row.get("net_pnl", 0.0) or 0.0)
        for bucket_value in set(bucket_values):
            bucket = buckets.setdefault(
                str(bucket_value),
                {
                    "closed": 0.0,
                    "wins": 0.0,
                    "losses": 0.0,
                    "net_pnl": 0.0,
                    "gross_wins": 0.0,
                    "gross_losses_abs": 0.0,
                },
            )
            bucket["closed"] += 1.0
            bucket["net_pnl"] += net
            if net >= 0:
                bucket["wins"] += 1.0
                bucket["gross_wins"] += net
            else:
                bucket["losses"] += 1.0
                bucket["gross_losses_abs"] += abs(net)

    output: list[dict[str, Any]] = []
    for bucket_value, data in sorted(buckets.items(), key=lambda item: item[0]):
        closed = int(data["closed"])
        wins = int(data["wins"])
        losses = int(data["losses"])
        gross_losses_abs = float(data["gross_losses_abs"])
        profit_factor = (float(data["gross_wins"]) / gross_losses_abs) if gross_losses_abs > 0 else 0.0
        output.append(
            {
                bucket_name: bucket_value,
                "closed": closed,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / closed * 100.0) if closed else 0.0,
                "net_pnl": float(data["net_pnl"]),
                "expectancy": (float(data["net_pnl"]) / closed) if closed else 0.0,
                "profit_factor": profit_factor,
            }
        )
    return output


def _safe_trade_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json", "{}")
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _reason_code_buckets(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("reason_codes")
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ("none",)
    if isinstance(raw, (list, tuple, set)):
        values = tuple(str(item).strip() for item in raw if str(item).strip())
        return values or ("none",)
    return ("none",)


def _quality_bucket(value: float, *, thresholds: tuple[float, ...]) -> str:
    if value < thresholds[0]:
        return f"<{thresholds[0]:.2f}"
    for idx in range(1, len(thresholds)):
        if value < thresholds[idx]:
            return f"{thresholds[idx - 1]:.2f}-{thresholds[idx]:.2f}"
    return f">={thresholds[-1]:.2f}"


def _setup_quality_bucket(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("setup_quality_score")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return tuple()
    value = max(0.0, min(1.0, value))
    return (_quality_bucket(value, thresholds=(0.45, 0.55, 0.65, 0.75)),)


def _signal_quality_bucket(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("signal_quality_score")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return tuple()
    value = max(0.0, min(1.0, value))
    return (_quality_bucket(value, thresholds=(0.55, 0.65, 0.75, 0.85)),)


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    token = resolve_token(args=args)
    if not token:
        print("INVEST_TOKEN is empty. Set token in .env or pass --token.", file=sys.stderr)
        return 2

    try:
        app_config = ConfigLoader(args.config_dir).load()
    except Exception as exc:
        print(f"Config loading failed: {exc}", file=sys.stderr)
        return 2

    timezone_name = str(app_config.params.get("timezone", "Europe/Moscow"))
    try:
        period_start_local = parse_local_datetime(args.start, timezone_name=timezone_name, is_end=False)
        period_end_local = parse_local_datetime(args.end, timezone_name=timezone_name, is_end=True)
    except ValueError as exc:
        print(f"Period parsing failed: {exc}", file=sys.stderr)
        return 2

    if period_end_local <= period_start_local:
        print("Invalid period: --end must be greater than --start", file=sys.stderr)
        return 2

    profiles = split_csv_items(args.profiles)
    if not profiles:
        print("No profiles selected", file=sys.stderr)
        return 2
    strategies = split_csv_items(args.strategies)
    if not strategies:
        print("No strategies selected", file=sys.stderr)
        return 2

    registry = InstrumentRegistry.from_config(app_config)
    enabled_instruments = [item.symbol for item in registry.enabled()]
    try:
        instruments = ensure_selected_instruments(
            available_enabled=enabled_instruments,
            requested=split_csv_items(args.instruments),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        params_by_profile = load_profile_params(args.config_dir, profiles)
    except Exception as exc:
        print(f"Failed to load profile params: {exc}", file=sys.stderr)
        return 2

    try:
        warmup_start_local = compute_warmup_start(
            report_start_local=period_start_local,
            timeframe=args.timeframe,
            params_by_profile=params_by_profile,
        )
    except Exception as exc:
        print(f"Warmup period calculation failed: {exc}", file=sys.stderr)
        return 2

    warmup_start_utc = warmup_start_local.astimezone(timezone.utc)
    report_start_utc = period_start_local.astimezone(timezone.utc)
    report_end_utc = period_end_local.astimezone(timezone.utc)

    print(
        "Загрузка истории из T-Invest: "
        f"{len(instruments)} инструмент(ов), период {warmup_start_local.isoformat()} -> {period_end_local.isoformat()} ..."
    )
    try:
        candles_by_instrument = asyncio.run(
            load_all_history(
                token=token,
                registry=registry,
                instruments=instruments,
                timeframe=args.timeframe,
                warmup_start_utc=warmup_start_utc,
                report_end_utc=report_end_utc,
                api_limit=args.api_limit,
                chunk_days=args.chunk_days,
            )
        )
    except Exception as exc:
        print(f"Failed to load historical candles from T-Invest: {exc}", file=sys.stderr)
        return 1

    for symbol, rows in candles_by_instrument.items():
        if not rows:
            print(f"Нет свечей для {symbol}. Проверьте UID/FIGI и период.", file=sys.stderr)
            return 1
        print(f"{symbol}: загружено свечей {len(rows)}")

    combos = build_combo_tasks(
        profiles=profiles,
        strategies=strategies,
        instruments=instruments,
    )
    print(f"Всего комбинаций для теста: {len(combos)}")

    results: list[ComboRunResult] = []
    workers = max(1, int(args.workers))
    total_combos = len(combos)
    done_combos = 0
    combo_started = time.monotonic()
    print(f"[combos] started: {total_combos} combo(s), workers={workers}")

    if workers == 1:
        for combo in combos:
            params = params_by_profile[combo.profile]
            candles = candles_by_instrument.get(combo.instrument, [])
            result = run_combo_backtest(
                task=combo,
                candles=candles,
                app_config=app_config,
                params=params,
                timeframe=args.timeframe,
                report_start_utc=report_start_utc,
                report_end_utc=report_end_utc,
            )
            results.append(result)
            done_combos += 1
            elapsed = time.monotonic() - combo_started
            eta = _eta_seconds(done=done_combos, total=total_combos, elapsed_seconds=elapsed)
            print(
                f"[combos] {done_combos}/{total_combos} ({_pct(done_combos, total_combos):.1f}%) "
                f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} | "
                f"{_combo_label(profile=combo.profile, strategy=combo.strategy, instrument=combo.instrument)} "
                f"-> {result.status}"
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_combo = {}
            for combo in combos:
                params = params_by_profile[combo.profile]
                candles = candles_by_instrument.get(combo.instrument, [])
                future = executor.submit(
                    run_combo_backtest,
                    task=combo,
                    candles=candles,
                    app_config=app_config,
                    params=params,
                    timeframe=args.timeframe,
                    report_start_utc=report_start_utc,
                    report_end_utc=report_end_utc,
                )
                future_to_combo[future] = combo

            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ComboRunResult(
                        profile=combo.profile,
                        strategy=combo.strategy,
                        instrument=combo.instrument,
                        status="error",
                        error=str(exc),
                        metrics={},
                        signals=[],
                        trades=[],
                        events=[],
                    )
                results.append(result)
                done_combos += 1
                elapsed = time.monotonic() - combo_started
                eta = _eta_seconds(done=done_combos, total=total_combos, elapsed_seconds=elapsed)
                print(
                    f"[combos] {done_combos}/{total_combos} ({_pct(done_combos, total_combos):.1f}%) "
                    f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} | "
                    f"{_combo_label(profile=combo.profile, strategy=combo.strategy, instrument=combo.instrument)} "
                    f"-> {result.status}"
                )

    results = sorted(results, key=lambda row: (row.profile, row.strategy, row.instrument))
    profile_metrics = aggregate_profile_metrics(results)

    all_signals: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []

    for item in results:
        combo_rows.append(combo_summary_row(item))
        all_signals.extend(item.signals)
        all_trades.extend(item.trades)
        all_events.extend(item.events)

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"matrix_backtest_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_fields = [
        "profile",
        "strategy",
        "instrument",
        "status",
        "error",
        "signals",
        "closed",
        "wins",
        "losses",
        "win_rate_pct",
        "loss_rate_pct",
        "trading_days",
        "trades_per_day",
        "net_pnl",
        "gross_pnl",
        "fees",
        "profit_factor",
        "expectancy",
        "avg_r",
        "max_drawdown",
        "avg_win_points",
        "avg_loss_points_abs",
        "long_closed",
        "short_closed",
        "hour_distribution_json",
        "weekday_distribution_json",
        "exit_reason_breakdown_json",
        "entry_mode_breakdown_json",
        "reason_code_expectancy_json",
        "setup_quality_expectancy_json",
        "signal_quality_expectancy_json",
        "open_trades",
    ]
    write_csv(out_dir / "combo_summary.csv", rows=combo_rows, fieldnames=combo_fields)

    signal_fields = [
        "profile",
        "strategy",
        "instrument",
        "signal_id",
        "timestamp",
        "regime",
        "direction",
        "entry_mode",
        "entry",
        "stop_loss",
        "tp1",
        "tp2",
        "metadata_json",
    ]
    write_csv(out_dir / "signals.csv", rows=all_signals, fieldnames=signal_fields)

    trade_fields = [
        "profile",
        "strategy",
        "instrument",
        "trade_id",
        "signal_id",
        "status",
        "direction",
        "created_at",
        "activated_at",
        "closed_at",
        "entry_hour_local",
        "entry_weekday_local",
        "entry_mode",
        "entry",
        "entry_fill_price",
        "stop_loss",
        "tp1",
        "tp2",
        "bars_waiting",
        "bars_in_trade",
        "gross_pnl",
        "net_pnl",
        "fees_paid",
        "r_multiple",
        "exit_reason",
        "remaining_qty",
        "metadata_json",
    ]
    write_csv(out_dir / "trades.csv", rows=all_trades, fieldnames=trade_fields)

    hour_expectancy = _expectancy_table(rows=all_trades, bucket_field="entry_hour_local", bucket_name="hour")
    weekday_expectancy = _expectancy_table(
        rows=all_trades,
        bucket_field="entry_weekday_local",
        bucket_name="weekday",
    )
    write_csv(
        out_dir / "expectancy_by_hour.csv",
        rows=hour_expectancy,
        fieldnames=["hour", "closed", "wins", "losses", "win_rate_pct", "net_pnl", "expectancy", "profit_factor"],
    )
    write_csv(
        out_dir / "expectancy_by_weekday.csv",
        rows=weekday_expectancy,
        fieldnames=[
            "weekday",
            "closed",
            "wins",
            "losses",
            "win_rate_pct",
            "net_pnl",
            "expectancy",
            "profit_factor",
        ],
    )
    reason_code_expectancy = _metadata_expectancy_table(
        rows=all_trades,
        bucket_name="reason_code",
        resolve_buckets=_reason_code_buckets,
    )
    setup_quality_expectancy = _metadata_expectancy_table(
        rows=all_trades,
        bucket_name="setup_quality_bucket",
        resolve_buckets=_setup_quality_bucket,
    )
    signal_quality_expectancy = _metadata_expectancy_table(
        rows=all_trades,
        bucket_name="signal_quality_bucket",
        resolve_buckets=_signal_quality_bucket,
    )
    write_csv(
        out_dir / "expectancy_by_reason_code.csv",
        rows=reason_code_expectancy,
        fieldnames=[
            "reason_code",
            "closed",
            "wins",
            "losses",
            "win_rate_pct",
            "net_pnl",
            "expectancy",
            "profit_factor",
        ],
    )
    write_csv(
        out_dir / "expectancy_by_setup_quality.csv",
        rows=setup_quality_expectancy,
        fieldnames=[
            "setup_quality_bucket",
            "closed",
            "wins",
            "losses",
            "win_rate_pct",
            "net_pnl",
            "expectancy",
            "profit_factor",
        ],
    )
    write_csv(
        out_dir / "expectancy_by_signal_quality.csv",
        rows=signal_quality_expectancy,
        fieldnames=[
            "signal_quality_bucket",
            "closed",
            "wins",
            "losses",
            "win_rate_pct",
            "net_pnl",
            "expectancy",
            "profit_factor",
        ],
    )

    event_fields = [
        "profile",
        "strategy",
        "instrument",
        "trade_id",
        "signal_id",
        "event_type",
        "status",
        "event_time",
        "price",
        "size",
        "payload_json",
    ]
    write_csv(out_dir / "events.csv", rows=all_events, fieldnames=event_fields)

    report_ru = build_russian_report(
        period_start_local=period_start_local,
        period_end_local=period_end_local,
        timeframe=args.timeframe,
        results=results,
        profile_metrics=profile_metrics,
    )
    (out_dir / "summary_ru.txt").write_text(report_ru, encoding="utf-8")

    json_payload = default_json_payload(
        period_start_local=period_start_local,
        period_end_local=period_end_local,
        timeframe=args.timeframe,
        results=results,
        profile_metrics=profile_metrics,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combo_elapsed = time.monotonic() - combo_started
    full_elapsed = time.monotonic() - started_at
    print(f"[combos] finished in {_fmt_duration(combo_elapsed)}")
    print(f"[total] finished in {_fmt_duration(full_elapsed)}")
    print(f"Готово. Отчеты сохранены: {out_dir}")
    print("- summary_ru.txt")
    print("- combo_summary.csv")
    print("- signals.csv")
    print("- trades.csv")
    print("- events.csv")
    print("- expectancy_by_hour.csv")
    print("- expectancy_by_weekday.csv")
    print("- expectancy_by_reason_code.csv")
    print("- expectancy_by_setup_quality.csv")
    print("- expectancy_by_signal_quality.csv")
    print("- summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
