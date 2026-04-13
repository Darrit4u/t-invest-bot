"""Matrix backtesting utilities for profile x strategy x instrument runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from core.config_loader import AppConfig
from core.history_preloader import estimate_required_bars
from core.instrument_registry import InstrumentRegistry
from core.market_data import Candle, CandleValidationError, _load_tinvest_sdk, _quotation_to_float
from core.news_filter import NewsBlackoutFilter
from core.session_manager import SessionManager
from core.signal_engine import SignalEngine
from core.stats_engine import StatsEngine
from core.trade_simulator import SimulatedTrade, TradeEvent, TradeSimulator
from storage.memory_store import MemoryCandleStore


DEFAULT_STRATEGIES = (
    "trend_pullback_vwap_ema",
    "compression_breakout",
    "liquidity_sweep_reversal",
)

_TIMEFRAME_TO_MINUTES = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
}

_TIMEFRAME_TO_CANDLE_INTERVAL_ATTR = {
    "1min": "CANDLE_INTERVAL_1_MIN",
    "2min": "CANDLE_INTERVAL_2_MIN",
    "3min": "CANDLE_INTERVAL_3_MIN",
    "5min": "CANDLE_INTERVAL_5_MIN",
    "10min": "CANDLE_INTERVAL_10_MIN",
    "15min": "CANDLE_INTERVAL_15_MIN",
    "30min": "CANDLE_INTERVAL_30_MIN",
    "1hour": "CANDLE_INTERVAL_HOUR",
}

_CLOSE_EVENTS = {"tp2_hit", "sl_hit", "expired", "cancelled_by_news", "cancelled_by_session_end"}


@dataclass(frozen=True, slots=True)
class ComboTask:
    profile: str
    strategy: str
    instrument: str


@dataclass(slots=True)
class ComboRunResult:
    profile: str
    strategy: str
    instrument: str
    status: str
    error: str | None
    metrics: dict[str, Any]
    signals: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    events: list[dict[str, Any]]


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def debug(self, *args: Any, **kwargs: Any) -> None:
        return None


def parse_local_datetime(value: str, *, timezone_name: str, is_end: bool = False) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("datetime value is empty")

    zone = ZoneInfo(timezone_name)
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.replace(tzinfo=zone)
        except ValueError:
            continue

    try:
        parsed_date = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "Unsupported datetime format. Use YYYY-MM-DD, YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS"
        ) from exc

    if is_end:
        return datetime.combine(parsed_date, time(23, 59, 59), tzinfo=zone)
    return datetime.combine(parsed_date, time(0, 0, 0), tzinfo=zone)


def split_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def timeframe_delta(timeframe: str) -> timedelta:
    minutes = _TIMEFRAME_TO_MINUTES.get(timeframe.lower().strip())
    if minutes is None:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Supported: {', '.join(sorted(_TIMEFRAME_TO_MINUTES))}"
        )
    return timedelta(minutes=minutes)


def build_combo_tasks(
    *,
    profiles: list[str],
    strategies: list[str],
    instruments: list[str],
) -> list[ComboTask]:
    tasks = [
        ComboTask(profile=profile, strategy=strategy, instrument=instrument)
        for profile in profiles
        for strategy in strategies
        for instrument in instruments
    ]
    return sorted(tasks, key=lambda row: (row.profile, row.strategy, row.instrument))


def load_profile_params(config_dir: str | Any, profile_names: list[str]) -> dict[str, dict[str, Any]]:
    from pathlib import Path

    root = Path(config_dir)
    profile_dir = root / "profiles"
    out: dict[str, dict[str, Any]] = {}
    for name in profile_names:
        path = profile_dir / f"params.{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Profile params file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid profile file (expected mapping): {path}")
        out[name] = payload
    return out


def compute_warmup_start(
    *,
    report_start_local: datetime,
    timeframe: str,
    params_by_profile: dict[str, dict[str, Any]],
) -> datetime:
    max_bars = 0
    for params in params_by_profile.values():
        max_bars = max(max_bars, estimate_required_bars(params=params, timeframe=timeframe))
    max_bars = max(max_bars, 120)
    # Additional safety multiplier due non-trading periods and weekend gaps.
    return report_start_local - (timeframe_delta(timeframe) * max_bars * 3)


async def fetch_historical_candles_tinvest(
    *,
    token: str,
    instrument_uid: str | None,
    instrument_figi: str | None,
    instrument_symbol: str,
    timeframe: str,
    start_utc: datetime,
    end_utc: datetime,
    api_limit: int,
    chunk_days: int,
) -> list[Candle]:
    sdk = _load_tinvest_sdk()
    if sdk is None:
        raise RuntimeError(
            "T-Invest SDK is not available. Install package 't-tech-investments' and retry."
        )

    interval = _map_history_interval(timeframe=timeframe, candle_interval_cls=sdk["CandleInterval"])
    instrument_id = (instrument_uid or "").strip() or (instrument_figi or "").strip()
    if not instrument_id:
        raise RuntimeError(f"Instrument {instrument_symbol} has neither UID nor FIGI")

    async_client = sdk["AsyncClient"]
    step = timeframe_delta(timeframe)
    candles_by_time: dict[datetime, Candle] = {}
    safe_limit = max(50, int(api_limit))
    safe_chunk_days = max(1, int(chunk_days))

    async with async_client(token) as client:
        chunk_start = start_utc
        while chunk_start <= end_utc:
            chunk_end = min(chunk_start + timedelta(days=safe_chunk_days), end_utc)
            cursor = chunk_start

            while cursor <= chunk_end:
                response = await client.market_data.get_candles(
                    instrument_id=instrument_id,
                    interval=interval,
                    from_=cursor,
                    to=chunk_end,
                    limit=safe_limit,
                )
                rows = sorted(
                    list(getattr(response, "candles", []) or []),
                    key=lambda item: getattr(item, "time", datetime.min.replace(tzinfo=timezone.utc)),
                )
                if not rows:
                    break

                max_dt: datetime | None = None
                for row in rows:
                    try:
                        candle = Candle.validated(
                            dt=getattr(row, "time"),
                            open_=_quotation_to_float(getattr(row, "open")),
                            high=_quotation_to_float(getattr(row, "high")),
                            low=_quotation_to_float(getattr(row, "low")),
                            close=_quotation_to_float(getattr(row, "close")),
                            volume=float(getattr(row, "volume", 0.0)),
                            instrument=instrument_symbol,
                            timeframe=timeframe,
                        )
                    except (CandleValidationError, TypeError, ValueError):
                        continue
                    candles_by_time[candle.datetime] = candle
                    if max_dt is None or candle.datetime > max_dt:
                        max_dt = candle.datetime

                if max_dt is None:
                    break

                next_cursor = max_dt + step
                if next_cursor <= cursor:
                    next_cursor = cursor + step
                if next_cursor > chunk_end:
                    break
                cursor = next_cursor

            chunk_start = chunk_end + step

    return sorted(candles_by_time.values(), key=lambda item: item.datetime)


def run_combo_backtest(
    *,
    task: ComboTask,
    candles: list[Candle],
    app_config: AppConfig,
    params: dict[str, Any],
    timeframe: str,
    report_start_utc: datetime,
    report_end_utc: datetime,
) -> ComboRunResult:
    if not candles:
        return ComboRunResult(
            profile=task.profile,
            strategy=task.strategy,
            instrument=task.instrument,
            status="error",
            error="no_candles",
            metrics={},
            signals=[],
            trades=[],
            events=[],
        )

    full_registry = InstrumentRegistry.from_config(app_config)
    base_meta = full_registry.get(task.instrument)
    selected_meta = replace(base_meta, allowed_strategies=(task.strategy,))
    registry = InstrumentRegistry(items={task.instrument: selected_meta})

    store = MemoryCandleStore(history_depth=max(app_config.history_depth, len(candles) + 10))
    blackout_filter = NewsBlackoutFilter(app_config.blackout_windows)
    signal_engine = SignalEngine(
        registry=registry,
        store=store,
        params=params,
        blackout_filter=blackout_filter,
        logger=_NullLogger(),
    )
    trade_simulator = TradeSimulator(params=params, logger=_NullLogger(), storage=None)
    stats_engine = StatsEngine()
    session_manager = SessionManager()

    signal_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    ordered = sorted(candles, key=lambda item: item.datetime)
    for candle in ordered:
        store.upsert(candle)
        if candle.datetime < report_start_utc:
            continue
        if candle.datetime > report_end_utc:
            break

        if trade_simulator.open_trades_count() == 0:
            result = signal_engine.process_candle(instrument=task.instrument, timeframe=timeframe)
            for signal in result.accepted_signals:
                if signal.timestamp < report_start_utc or signal.timestamp > report_end_utc:
                    continue

                stats_engine.record_signal(instrument=signal.instrument, strategy=signal.strategy)
                registration_events = trade_simulator.register_signal(signal, timeframe=timeframe)
                signal_rows.append(
                    {
                        "profile": task.profile,
                        "strategy": task.strategy,
                        "instrument": task.instrument,
                        "signal_id": signal.signal_id,
                        "timestamp": signal.timestamp.isoformat(),
                        "regime": signal.regime.value,
                        "direction": signal.direction.value,
                        "entry_mode": signal.entry_mode,
                        "entry": signal.entry,
                        "stop_loss": signal.stop_loss,
                        "tp1": signal.tp1,
                        "tp2": signal.tp2,
                        "metadata_json": json.dumps(signal.metadata, ensure_ascii=False),
                    }
                )
                for event in registration_events:
                    stats_engine.record_event(event)
                    event_rows.append(_event_row(task=task, event=event))
                break

        session_state = session_manager.get_state(selected_meta, candle.datetime)
        blackout_active, blackout_reason = blackout_filter.is_blocked(candle.datetime)
        events = trade_simulator.process_candle(
            candle=candle,
            session_active=session_state.is_active,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
        )
        for event in events:
            stats_engine.record_event(event)
            event_rows.append(_event_row(task=task, event=event))
            trade = trade_simulator.get_trade(event.trade_id)
            if event.event_type in _CLOSE_EVENTS and trade is not None:
                stats_engine.record_trade_closed(trade)

    all_trades = list(trade_simulator.trades())
    trade_rows = [_trade_row(task=task, trade=trade) for trade in all_trades]
    closed_trades = [trade for trade in all_trades if trade.closed_at is not None]
    point_stats = _point_stats(closed_trades)

    summary_global = stats_engine.summary().get("global", {})
    wins = int(summary_global.get("wins", 0))
    losses = int(summary_global.get("losses", 0))
    closed = int(summary_global.get("closed", 0))
    win_rate = float(summary_global.get("win_rate", 0.0))
    loss_rate = 1.0 - win_rate if closed > 0 else 0.0

    metrics = {
        "profile": task.profile,
        "strategy": task.strategy,
        "instrument": task.instrument,
        "signals": int(summary_global.get("signals", 0)),
        "activated": int(summary_global.get("activated", 0)),
        "closed": closed,
        "open_trades": trade_simulator.open_trades_count(),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100.0,
        "loss_rate_pct": loss_rate * 100.0,
        "gross_pnl": float(summary_global.get("gross_pnl", 0.0)),
        "net_pnl": float(summary_global.get("net_pnl", 0.0)),
        "fees": float(summary_global.get("fees", 0.0)),
        "expectancy": float(summary_global.get("expectancy", 0.0)),
        "avg_r": float(summary_global.get("avg_r", 0.0)),
        "profit_factor": float(summary_global.get("profit_factor", 0.0)),
        "max_drawdown": float(summary_global.get("max_drawdown", 0.0)),
        "tp1_hits": int(summary_global.get("tp1_hits", 0)),
        "tp2_hits": int(summary_global.get("tp2_hits", 0)),
        "sl_hits": int(summary_global.get("sl_hits", 0)),
        "expired": int(summary_global.get("expired", 0)),
        "cancelled_news": int(summary_global.get("cancelled_news", 0)),
        "cancelled_session": int(summary_global.get("cancelled_session", 0)),
        "avg_win_points": point_stats["avg_win_points"],
        "avg_loss_points_abs": point_stats["avg_loss_points_abs"],
        "gross_wins_points": point_stats["gross_wins_points"],
        "gross_losses_points_abs": point_stats["gross_losses_points_abs"],
        "signals_rows": len(signal_rows),
        "trades_rows": len(trade_rows),
        "events_rows": len(event_rows),
    }

    return ComboRunResult(
        profile=task.profile,
        strategy=task.strategy,
        instrument=task.instrument,
        status="ok",
        error=None,
        metrics=metrics,
        signals=signal_rows,
        trades=trade_rows,
        events=event_rows,
    )


def aggregate_profile_metrics(results: list[ComboRunResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in results:
        if item.status != "ok":
            continue
        bucket = grouped.setdefault(
            item.profile,
            {
                "profile": item.profile,
                "combos": 0,
                "signals": 0,
                "activated": 0,
                "closed": 0,
                "wins": 0,
                "losses": 0,
                "net_pnl": 0.0,
                "gross_pnl": 0.0,
                "fees": 0.0,
                "max_drawdown_sum": 0.0,
                "gross_wins_points": 0.0,
                "gross_losses_points_abs": 0.0,
                "open_trades": 0,
            },
        )
        metrics = item.metrics
        bucket["combos"] += 1
        bucket["signals"] += int(metrics.get("signals", 0))
        bucket["activated"] += int(metrics.get("activated", 0))
        bucket["closed"] += int(metrics.get("closed", 0))
        bucket["wins"] += int(metrics.get("wins", 0))
        bucket["losses"] += int(metrics.get("losses", 0))
        bucket["net_pnl"] += float(metrics.get("net_pnl", 0.0))
        bucket["gross_pnl"] += float(metrics.get("gross_pnl", 0.0))
        bucket["fees"] += float(metrics.get("fees", 0.0))
        bucket["max_drawdown_sum"] += float(metrics.get("max_drawdown", 0.0))
        bucket["gross_wins_points"] += float(metrics.get("gross_wins_points", 0.0))
        bucket["gross_losses_points_abs"] += float(metrics.get("gross_losses_points_abs", 0.0))
        bucket["open_trades"] += int(metrics.get("open_trades", 0))

    for bucket in grouped.values():
        closed = max(0, int(bucket["closed"]))
        wins = int(bucket["wins"])
        losses = int(bucket["losses"])
        bucket["win_rate"] = (wins / closed) if closed else 0.0
        bucket["win_rate_pct"] = bucket["win_rate"] * 100.0
        bucket["loss_rate_pct"] = ((losses / closed) * 100.0) if closed else 0.0
        bucket["profit_factor"] = (
            float(bucket["gross_wins_points"]) / float(bucket["gross_losses_points_abs"])
            if float(bucket["gross_losses_points_abs"]) > 0
            else 0.0
        )
        bucket["avg_drawdown_per_combo"] = (
            float(bucket["max_drawdown_sum"]) / float(bucket["combos"])
            if int(bucket["combos"]) > 0
            else 0.0
        )

    return dict(sorted(grouped.items(), key=lambda row: row[0]))


def build_russian_report(
    *,
    period_start_local: datetime,
    period_end_local: datetime,
    timeframe: str,
    results: list[ComboRunResult],
    profile_metrics: dict[str, dict[str, Any]],
) -> str:
    ok_results = [item for item in results if item.status == "ok"]
    bad_results = [item for item in results if item.status != "ok"]
    sorted_profiles = sorted(profile_metrics.values(), key=lambda row: float(row["net_pnl"]), reverse=True)
    sorted_combos = sorted(ok_results, key=lambda row: float(row.metrics.get("net_pnl", 0.0)), reverse=True)

    lines: list[str] = []
    lines.append("МАТРИЧНЫЙ БЭКТЕСТ СТРАТЕГИЙ (T-Invest, MOEX futures)")
    lines.append(
        "Период теста (локальное время): "
        f"{period_start_local.strftime('%Y-%m-%d %H:%M:%S')} - {period_end_local.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"Таймфрейм: {timeframe}")
    lines.append(f"Комбинаций рассчитано: {len(ok_results)}")
    if bad_results:
        lines.append(f"Комбинаций с ошибкой: {len(bad_results)}")
    lines.append("")

    lines.append("СРАВНЕНИЕ ПРОФИЛЕЙ")
    for bucket in sorted_profiles:
        lines.append(
            f"{bucket['profile']}: combos={bucket['combos']}, signals={bucket['signals']}, "
            f"closed={bucket['closed']}, winrate={bucket['win_rate_pct']:.2f}%, "
            f"net={bucket['net_pnl']:.4f} п., PF={bucket['profit_factor']:.3f}, "
            f"avg_dd={bucket['avg_drawdown_per_combo']:.4f} п."
        )
    lines.append("")

    lines.append("ТОП-10 КОМБИНАЦИЙ ПО NET PNL")
    for item in sorted_combos[:10]:
        m = item.metrics
        lines.append(
            f"{item.profile} | {item.strategy} | {item.instrument}: "
            f"net={m.get('net_pnl', 0.0):.4f} п., closed={m.get('closed', 0)}, "
            f"winrate={m.get('win_rate_pct', 0.0):.2f}%, PF={m.get('profit_factor', 0.0):.3f}, "
            f"max_dd={m.get('max_drawdown', 0.0):.4f} п."
        )
    lines.append("")

    lines.append("ПОЛНАЯ МАТРИЦА")
    for item in sorted(
        ok_results,
        key=lambda row: (row.profile, row.strategy, row.instrument),
    ):
        m = item.metrics
        lines.append(
            f"{item.profile} | {item.strategy} | {item.instrument}: "
            f"signals={m.get('signals', 0)}, closed={m.get('closed', 0)}, "
            f"wins={m.get('wins', 0)}, losses={m.get('losses', 0)}, "
            f"winrate={m.get('win_rate_pct', 0.0):.2f}%, net={m.get('net_pnl', 0.0):.4f} п., "
            f"fees={m.get('fees', 0.0):.4f}, PF={m.get('profit_factor', 0.0):.3f}, "
            f"avg_win={m.get('avg_win_points', 0.0):.4f} п., "
            f"avg_loss={m.get('avg_loss_points_abs', 0.0):.4f} п."
        )

    if bad_results:
        lines.append("")
        lines.append("КОМБИНАЦИИ С ОШИБКАМИ")
        for item in bad_results:
            lines.append(f"{item.profile} | {item.strategy} | {item.instrument}: {item.error}")

    return "\n".join(lines) + "\n"


def default_json_payload(
    *,
    period_start_local: datetime,
    period_end_local: datetime,
    timeframe: str,
    results: list[ComboRunResult],
    profile_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "period_local": {
            "start": period_start_local.isoformat(),
            "end": period_end_local.isoformat(),
        },
        "timeframe": timeframe,
        "profiles": profile_metrics,
        "combos": [
            {
                "profile": item.profile,
                "strategy": item.strategy,
                "instrument": item.instrument,
                "status": item.status,
                "error": item.error,
                "metrics": item.metrics,
            }
            for item in results
        ],
    }


def _map_history_interval(*, timeframe: str, candle_interval_cls: Any) -> Any:
    attr = _TIMEFRAME_TO_CANDLE_INTERVAL_ATTR.get(timeframe.lower().strip())
    if attr is None or not hasattr(candle_interval_cls, attr):
        raise ValueError(f"Unsupported historical timeframe: {timeframe}")
    return getattr(candle_interval_cls, attr)


def _trade_row(*, task: ComboTask, trade: SimulatedTrade) -> dict[str, Any]:
    return {
        "profile": task.profile,
        "strategy": task.strategy,
        "instrument": task.instrument,
        "trade_id": trade.trade_id,
        "signal_id": trade.signal_id,
        "status": trade.status.value,
        "direction": trade.direction.value,
        "created_at": trade.created_at.isoformat(),
        "activated_at": trade.activated_at.isoformat() if trade.activated_at else "",
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
        "entry": trade.entry,
        "entry_fill_price": trade.entry_fill_price if trade.entry_fill_price is not None else "",
        "stop_loss": trade.stop_loss,
        "tp1": trade.tp1,
        "tp2": trade.tp2,
        "bars_waiting": trade.bars_waiting,
        "bars_in_trade": trade.bars_in_trade,
        "gross_pnl": trade.gross_pnl,
        "net_pnl": trade.net_pnl,
        "fees_paid": trade.fees_paid,
        "r_multiple": trade.r_multiple,
        "exit_reason": trade.exit_reason or "",
        "remaining_qty": trade.remaining_qty,
        "metadata_json": json.dumps(trade.metadata, ensure_ascii=False),
    }


def _event_row(*, task: ComboTask, event: TradeEvent) -> dict[str, Any]:
    return {
        "profile": task.profile,
        "strategy": task.strategy,
        "instrument": task.instrument,
        "trade_id": event.trade_id,
        "signal_id": event.signal_id,
        "event_type": event.event_type,
        "status": event.status,
        "event_time": event.event_time.isoformat(),
        "price": event.price if event.price is not None else "",
        "size": event.size if event.size is not None else "",
        "payload_json": json.dumps(event.payload, ensure_ascii=False),
    }


def _point_stats(trades: list[SimulatedTrade]) -> dict[str, float]:
    wins = [item.net_pnl for item in trades if item.net_pnl >= 0]
    losses = [item.net_pnl for item in trades if item.net_pnl < 0]
    gross_wins = float(sum(wins))
    gross_losses_abs = float(sum(abs(item) for item in losses))
    return {
        "avg_win_points": (gross_wins / len(wins)) if wins else 0.0,
        "avg_loss_points_abs": (gross_losses_abs / len(losses)) if losses else 0.0,
        "gross_wins_points": gross_wins,
        "gross_losses_points_abs": gross_losses_abs,
    }

