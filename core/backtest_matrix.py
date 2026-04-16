"""Matrix backtesting utilities for profile x strategy x instrument runs."""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from core.config_loader import AppConfig
from core.execution_engine import ExecutionEngine
from core.history_preloader import estimate_required_bars
from core.instrument_registry import InstrumentRegistry
from core.market_data import Candle, CandleValidationError, _load_tinvest_sdk, _quotation_to_float
from core.models import Trade
from core.news_filter import NewsBlackoutFilter
from core.portfolio_engine import PortfolioEngine
from core.portfolio_events import PortfolioEvent
from core.session_manager import SessionManager
from core.signal_engine import SignalEngine
from core.stats_engine import StatsEngine
from core.trade_simulator import TradeEvent, TradeSimulator
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


@dataclass(slots=True)
class PortfolioRunResult:
    profile: str
    status: str
    error: str | None
    metrics: dict[str, Any]
    signals: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    events: list[dict[str, Any]]
    portfolio_events: list[dict[str, Any]]


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
    execution_engine = ExecutionEngine(simulator=trade_simulator)
    portfolio_engine = PortfolioEngine(params=params)
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

        if execution_engine.open_positions_count() == 0:
            result = signal_engine.process_candle(instrument=task.instrument, timeframe=timeframe)
            for signal in result.accepted_signals:
                if signal.timestamp < report_start_utc or signal.timestamp > report_end_utc:
                    continue

                stats_engine.record_signal(instrument=signal.instrument, strategy=signal.strategy)
                selection = portfolio_engine.select_signals(
                    signals=[signal],
                    open_positions=execution_engine.positions(),
                )
                for item in selection.events:
                    stats_engine.record_portfolio_event(item)
                if not selection.accepted_signals:
                    continue
                sized_signal = selection.accepted_signals[0]
                open_result = execution_engine.open_from_signal(signal=sized_signal, timeframe=timeframe)
                signal_rows.append(
                    {
                        "profile": task.profile,
                        "strategy": task.strategy,
                        "instrument": task.instrument,
                        "signal_id": sized_signal.signal_id,
                        "timestamp": sized_signal.timestamp.isoformat(),
                        "regime": sized_signal.regime.value,
                        "direction": sized_signal.direction.value,
                        "entry_mode": sized_signal.entry_mode,
                        "entry": sized_signal.entry,
                        "stop_loss": sized_signal.stop_loss,
                        "tp1": sized_signal.tp1,
                        "tp2": sized_signal.tp2,
                        "metadata_json": json.dumps(sized_signal.metadata, ensure_ascii=False),
                    }
                )
                for event in open_result.events:
                    event_rows.append(_event_row(task=task, event=event))
                for normalized in portfolio_engine.normalize_execution_events(
                    execution_events=open_result.events,
                    execution_engine=execution_engine,
                ):
                    stats_engine.record_portfolio_event(normalized)
                break

        session_state = session_manager.get_state(selected_meta, candle.datetime)
        blackout_active, blackout_reason = blackout_filter.is_blocked(candle.datetime)
        process_result = execution_engine.process_market(
            candle=candle,
            session_active=session_state.is_active,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
        )
        for event in process_result.events:
            event_rows.append(_event_row(task=task, event=event))
        for normalized in portfolio_engine.normalize_execution_events(
            execution_events=process_result.events,
            execution_engine=execution_engine,
        ):
            stats_engine.record_portfolio_event(normalized)
        for trade in process_result.closed_trades:
            stats_engine.record_trade_closed(trade)

    timezone_name = str(app_config.params.get("timezone", "Europe/Moscow"))
    all_trades = list(execution_engine.trade_records())
    trade_rows = [
        _trade_row(
            task=task,
            trade=trade,
            timezone_name=timezone_name,
        )
        for trade in all_trades
    ]
    closed_trades = [trade for trade in all_trades if trade.closed_at is not None]
    point_stats = _point_stats(closed_trades)
    trading_days = _trading_days_count(
        candles=ordered,
        report_start_utc=report_start_utc,
        report_end_utc=report_end_utc,
        timezone_name=timezone_name,
    )
    trades_per_day = _trades_per_day(closed=len(closed_trades), trading_days=trading_days)
    trade_breakdowns = _trade_breakdowns(
        closed_trades=closed_trades,
        timezone_name=timezone_name,
    )
    quality_analytics = _trade_quality_analytics(closed_trades=closed_trades)

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
        "open_trades": execution_engine.open_positions_count(),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100.0,
        "loss_rate_pct": loss_rate * 100.0,
        "trading_days": trading_days,
        "trades_per_day": trades_per_day,
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
        "long_closed": trade_breakdowns["long_closed"],
        "short_closed": trade_breakdowns["short_closed"],
        "hour_distribution": trade_breakdowns["hour_distribution"],
        "weekday_distribution": trade_breakdowns["weekday_distribution"],
        "exit_reason_breakdown": trade_breakdowns["exit_reason_breakdown"],
        "entry_mode_breakdown": trade_breakdowns["entry_mode_breakdown"],
        "reason_code_expectancy": quality_analytics["reason_code_expectancy"],
        "setup_quality_expectancy": quality_analytics["setup_quality_expectancy"],
        "signal_quality_expectancy": quality_analytics["signal_quality_expectancy"],
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


def run_portfolio_backtest(
    *,
    profile: str,
    candles_by_instrument: dict[str, list[Candle]],
    app_config: AppConfig,
    params: dict[str, Any],
    timeframe: str,
    report_start_utc: datetime,
    report_end_utc: datetime,
    selected_instruments: list[str] | None = None,
    selected_strategies: list[str] | None = None,
) -> PortfolioRunResult:
    full_registry = InstrumentRegistry.from_config(app_config)
    instrument_set = set(selected_instruments or [item.symbol for item in full_registry.enabled()])
    strategy_set = set(selected_strategies or [])

    if not instrument_set:
        return PortfolioRunResult(
            profile=profile,
            status="error",
            error="no_instruments_selected",
            metrics={},
            signals=[],
            trades=[],
            events=[],
            portfolio_events=[],
        )

    registry_items = {}
    for symbol in sorted(instrument_set):
        if symbol not in full_registry:
            continue
        if symbol not in candles_by_instrument:
            continue

        meta = full_registry.get(symbol)
        if strategy_set:
            filtered = tuple(item for item in meta.allowed_strategies if item in strategy_set)
        else:
            filtered = meta.allowed_strategies
        if not filtered:
            continue
        registry_items[symbol] = replace(meta, allowed_strategies=filtered)

    if not registry_items:
        return PortfolioRunResult(
            profile=profile,
            status="error",
            error="no_registry_items",
            metrics={},
            signals=[],
            trades=[],
            events=[],
            portfolio_events=[],
        )

    registry = InstrumentRegistry(items=registry_items)
    max_rows = max((len(candles_by_instrument.get(symbol, [])) for symbol in registry_items), default=0)
    store = MemoryCandleStore(history_depth=max(app_config.history_depth, max_rows + 10))
    blackout_filter = NewsBlackoutFilter(app_config.blackout_windows)
    signal_engine = SignalEngine(
        registry=registry,
        store=store,
        params=params,
        blackout_filter=blackout_filter,
        logger=_NullLogger(),
    )
    trade_simulator = TradeSimulator(params=params, logger=_NullLogger(), storage=None)
    execution_engine = ExecutionEngine(simulator=trade_simulator)
    portfolio_engine = PortfolioEngine(params=params)
    stats_engine = StatsEngine()
    session_manager = SessionManager()

    signal_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    portfolio_event_rows: list[dict[str, Any]] = []

    ordered_candles: list[Candle] = []
    for symbol in sorted(registry_items):
        ordered_candles.extend(candles_by_instrument.get(symbol, []))
    ordered_candles.sort(key=lambda item: (item.datetime, item.instrument))

    for candle in ordered_candles:
        store.upsert(candle)
        if candle.datetime < report_start_utc:
            continue
        if candle.datetime > report_end_utc:
            break

        signal_result = signal_engine.process_candle(instrument=candle.instrument, timeframe=timeframe)
        for signal in signal_result.accepted_signals:
            stats_engine.record_signal(instrument=signal.instrument, strategy=signal.strategy)
            signal_rows.append(
                {
                    "profile": profile,
                    "strategy": signal.strategy,
                    "instrument": signal.instrument,
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

        selection = portfolio_engine.select_signals(
            signals=signal_result.accepted_signals,
            open_positions=execution_engine.positions(),
        )
        for item in selection.events:
            stats_engine.record_portfolio_event(item)
            portfolio_event_rows.append(_portfolio_event_row(profile=profile, event=item))

        execution_open = portfolio_engine.submit_for_execution(
            signals=selection.accepted_signals,
            execution_engine=execution_engine,
            timeframe=timeframe,
        )
        for item in execution_open.portfolio_events:
            stats_engine.record_portfolio_event(item)
            portfolio_event_rows.append(_portfolio_event_row(profile=profile, event=item))
        for event in execution_open.execution_events:
            event_rows.append(_event_row_portfolio(profile=profile, event=event))

        instrument_meta = registry.get(candle.instrument)
        session_state = session_manager.get_state(instrument_meta, candle.datetime)
        blackout_active, blackout_reason = blackout_filter.is_blocked(candle.datetime)
        process_result = execution_engine.process_market(
            candle=candle,
            session_active=session_state.is_active,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
        )
        for event in process_result.events:
            event_rows.append(_event_row_portfolio(profile=profile, event=event))
        for item in portfolio_engine.normalize_execution_events(
            execution_events=process_result.events,
            execution_engine=execution_engine,
        ):
            stats_engine.record_portfolio_event(item)
            portfolio_event_rows.append(_portfolio_event_row(profile=profile, event=item))
        for trade in process_result.closed_trades:
            stats_engine.record_trade_closed(trade)

    timezone_name = str(app_config.params.get("timezone", "Europe/Moscow"))
    all_trades = list(execution_engine.trade_records())
    trade_rows = [_trade_row_portfolio(profile=profile, trade=item, timezone_name=timezone_name) for item in all_trades]
    closed_trades = [trade for trade in all_trades if trade.closed_at is not None]

    point_stats = _point_stats(closed_trades)
    trading_days = _trading_days_count(
        candles=ordered_candles,
        report_start_utc=report_start_utc,
        report_end_utc=report_end_utc,
        timezone_name=timezone_name,
    )
    trades_per_day = _trades_per_day(closed=len(closed_trades), trading_days=trading_days)
    trade_breakdowns = _trade_breakdowns(
        closed_trades=closed_trades,
        timezone_name=timezone_name,
    )
    quality_analytics = _trade_quality_analytics(closed_trades=closed_trades)
    exposure_by_instrument, exposure_by_strategy, exposure_by_group = _portfolio_exposure_breakdown(closed_trades)

    summary_global = stats_engine.summary().get("global", {})
    wins = int(summary_global.get("wins", 0))
    losses = int(summary_global.get("losses", 0))
    closed = int(summary_global.get("closed", 0))
    win_rate = float(summary_global.get("win_rate", 0.0))
    loss_rate = 1.0 - win_rate if closed > 0 else 0.0

    metrics = {
        "profile": profile,
        "portfolio_mode": bool(portfolio_engine.enabled),
        "signals": int(summary_global.get("signals", 0)),
        "activated": int(summary_global.get("activated", 0)),
        "closed": closed,
        "open_trades": execution_engine.open_positions_count(),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100.0,
        "loss_rate_pct": loss_rate * 100.0,
        "trading_days": trading_days,
        "trades_per_day": trades_per_day,
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
        "long_closed": trade_breakdowns["long_closed"],
        "short_closed": trade_breakdowns["short_closed"],
        "hour_distribution": trade_breakdowns["hour_distribution"],
        "weekday_distribution": trade_breakdowns["weekday_distribution"],
        "exit_reason_breakdown": trade_breakdowns["exit_reason_breakdown"],
        "entry_mode_breakdown": trade_breakdowns["entry_mode_breakdown"],
        "reason_code_expectancy": quality_analytics["reason_code_expectancy"],
        "setup_quality_expectancy": quality_analytics["setup_quality_expectancy"],
        "signal_quality_expectancy": quality_analytics["signal_quality_expectancy"],
        "exposure_by_instrument": exposure_by_instrument,
        "exposure_by_strategy": exposure_by_strategy,
        "exposure_by_group": exposure_by_group,
        "portfolio_risk_reject_reasons": stats_engine.summary().get("portfolio", {}).get("risk_reject_reasons", {}),
        "signals_rows": len(signal_rows),
        "trades_rows": len(trade_rows),
        "events_rows": len(event_rows),
        "portfolio_events_rows": len(portfolio_event_rows),
    }

    return PortfolioRunResult(
        profile=profile,
        status="ok",
        error=None,
        metrics=metrics,
        signals=signal_rows,
        trades=trade_rows,
        events=event_rows,
        portfolio_events=portfolio_event_rows,
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
            f"max_dd={m.get('max_drawdown', 0.0):.4f} п., "
            f"trades/day={m.get('trades_per_day', 0.0):.2f}"
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
            f"tpd={m.get('trades_per_day', 0.0):.2f}, "
            f"long/short={m.get('long_closed', 0)}/{m.get('short_closed', 0)}, "
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


def _trade_row(*, task: ComboTask, trade: Trade, timezone_name: str) -> dict[str, Any]:
    zone = ZoneInfo(timezone_name)
    entry_time = trade.activated_at or trade.opened_at
    entry_local = entry_time.astimezone(zone)
    return {
        "profile": task.profile,
        "strategy": task.strategy,
        "instrument": task.instrument,
        "trade_id": trade.trade_id,
        "signal_id": trade.signal_id,
        "status": trade.status or "",
        "direction": trade.side.value,
        "created_at": (trade.created_at or trade.opened_at).isoformat(),
        "activated_at": (trade.activated_at or trade.opened_at).isoformat(),
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
        "entry_hour_local": entry_local.hour,
        "entry_weekday_local": entry_local.weekday(),
        "entry_mode": str(trade.metadata.get("entry_mode", "")),
        "entry": trade.entry_price,
        "entry_fill_price": trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry_price,
        "qty": trade.size,
        "planned_risk_money": _as_float_or_default(trade.metadata.get("planned_risk_money"), 0.0),
        "planned_risk_pct": _as_float_or_default(trade.metadata.get("planned_risk_pct"), 0.0),
        "stop_loss": trade.metadata.get("stop_loss", ""),
        "tp1": trade.metadata.get("tp1", ""),
        "tp2": trade.metadata.get("tp2", ""),
        "bars_waiting": trade.bars_waiting if trade.bars_waiting is not None else "",
        "bars_in_trade": trade.bars_in_trade if trade.bars_in_trade is not None else "",
        "gross_pnl": trade.gross_pnl if trade.gross_pnl is not None else trade.pnl,
        "net_pnl": trade.pnl,
        "fees_paid": trade.fees_paid if trade.fees_paid is not None else 0.0,
        "r_multiple": trade.r_multiple if trade.r_multiple is not None else 0.0,
        "exit_reason": trade.exit_reason or "",
        "remaining_qty": trade.remaining_qty if trade.remaining_qty is not None else "",
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


def _trade_row_portfolio(*, profile: str, trade: Trade, timezone_name: str) -> dict[str, Any]:
    zone = ZoneInfo(timezone_name)
    entry_time = trade.activated_at or trade.opened_at
    entry_local = entry_time.astimezone(zone)
    return {
        "profile": profile,
        "strategy": trade.strategy_id,
        "instrument": trade.instrument,
        "trade_id": trade.trade_id,
        "signal_id": trade.signal_id,
        "status": trade.status or "",
        "direction": trade.side.value,
        "created_at": (trade.created_at or trade.opened_at).isoformat(),
        "activated_at": (trade.activated_at or trade.opened_at).isoformat(),
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
        "entry_hour_local": entry_local.hour,
        "entry_weekday_local": entry_local.weekday(),
        "entry_mode": str(trade.metadata.get("entry_mode", "")),
        "entry": trade.entry_price,
        "entry_fill_price": trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry_price,
        "qty": trade.size,
        "planned_risk_money": _as_float_or_default(trade.metadata.get("planned_risk_money"), 0.0),
        "planned_risk_pct": _as_float_or_default(trade.metadata.get("planned_risk_pct"), 0.0),
        "stop_loss": trade.metadata.get("stop_loss", ""),
        "tp1": trade.metadata.get("tp1", ""),
        "tp2": trade.metadata.get("tp2", ""),
        "bars_waiting": trade.bars_waiting if trade.bars_waiting is not None else "",
        "bars_in_trade": trade.bars_in_trade if trade.bars_in_trade is not None else "",
        "gross_pnl": trade.gross_pnl if trade.gross_pnl is not None else trade.pnl,
        "net_pnl": trade.pnl,
        "fees_paid": trade.fees_paid if trade.fees_paid is not None else 0.0,
        "r_multiple": trade.r_multiple if trade.r_multiple is not None else 0.0,
        "exit_reason": trade.exit_reason or "",
        "remaining_qty": trade.remaining_qty if trade.remaining_qty is not None else "",
        "metadata_json": json.dumps(trade.metadata, ensure_ascii=False),
    }


def _event_row_portfolio(*, profile: str, event: TradeEvent) -> dict[str, Any]:
    return {
        "profile": profile,
        "strategy": event.strategy,
        "instrument": event.instrument,
        "trade_id": event.trade_id,
        "signal_id": event.signal_id,
        "event_type": event.event_type,
        "status": event.status,
        "event_time": event.event_time.isoformat(),
        "price": event.price if event.price is not None else "",
        "size": event.size if event.size is not None else "",
        "payload_json": json.dumps(event.payload, ensure_ascii=False),
    }


def _portfolio_event_row(*, profile: str, event: PortfolioEvent) -> dict[str, Any]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    return {
        "profile": profile,
        "event_type": event.event_type,
        "event_time": event.event_time.isoformat(),
        "instrument": event.instrument or "",
        "strategy": event.strategy or "",
        "signal_id": event.signal_id or "",
        "trade_id": event.trade_id or "",
        "reason": event.reason or "",
        "planned_risk_money": _as_float_or_default(payload.get("planned_risk_money"), 0.0),
        "planned_risk_pct": _as_float_or_default(payload.get("planned_risk_pct"), 0.0),
        "qty": _as_float_or_default(payload.get("qty", payload.get("position_qty")), 0.0),
        "payload_json": json.dumps(event.payload, ensure_ascii=False),
    }


def _portfolio_exposure_breakdown(closed_trades: list[Trade]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    by_instrument: dict[str, float] = {}
    by_strategy: dict[str, float] = {}
    by_group: dict[str, float] = {}
    for trade in closed_trades:
        raw = trade.metadata.get("planned_risk_money")
        try:
            risk_money = float(raw)
        except (TypeError, ValueError):
            raw_pct = trade.metadata.get("portfolio_risk_pct")
            stop_loss = trade.metadata.get("stop_loss")
            try:
                stop = float(stop_loss)
            except (TypeError, ValueError):
                stop = trade.entry_price
            risk_pct = abs(float(trade.entry_price) - stop) / max(abs(float(trade.entry_price)), 1e-9) * 100.0
            try:
                hinted_pct = float(raw_pct)
            except (TypeError, ValueError):
                hinted_pct = risk_pct
            risk_money = abs(hinted_pct)

        risk_money = max(0.0, risk_money)
        by_instrument[trade.instrument] = by_instrument.get(trade.instrument, 0.0) + risk_money
        by_strategy[trade.strategy_id] = by_strategy.get(trade.strategy_id, 0.0) + risk_money
        group_name = str(trade.metadata.get("correlation_group", "ungrouped")).strip() or "ungrouped"
        by_group[group_name] = by_group.get(group_name, 0.0) + risk_money

    sorted_instrument = dict(sorted(by_instrument.items(), key=lambda row: row[0]))
    sorted_strategy = dict(sorted(by_strategy.items(), key=lambda row: row[0]))
    sorted_group = dict(sorted(by_group.items(), key=lambda row: row[0]))
    return sorted_instrument, sorted_strategy, sorted_group


def _trading_days_count(
    *,
    candles: list[Candle],
    report_start_utc: datetime,
    report_end_utc: datetime,
    timezone_name: str,
) -> int:
    zone = ZoneInfo(timezone_name)
    trade_days: set[date] = set()
    for candle in candles:
        if candle.datetime < report_start_utc or candle.datetime > report_end_utc:
            continue
        trade_days.add(candle.datetime.astimezone(zone).date())
    return max(1, len(trade_days))


def _trades_per_day(*, closed: int, trading_days: int) -> float:
    return float(closed) / float(max(1, trading_days))


def _trade_breakdowns(*, closed_trades: list[Trade], timezone_name: str) -> dict[str, Any]:
    zone = ZoneInfo(timezone_name)
    long_closed = 0
    short_closed = 0
    hour_counter: Counter[int] = Counter()
    weekday_counter: Counter[int] = Counter()
    exit_reason_counter: Counter[str] = Counter()
    entry_mode_counter: Counter[str] = Counter()

    for trade in closed_trades:
        if trade.side.value == "LONG":
            long_closed += 1
        else:
            short_closed += 1

        local_entry = trade.opened_at.astimezone(zone)
        hour_counter[local_entry.hour] += 1
        weekday_counter[local_entry.weekday()] += 1

        exit_reason = (trade.exit_reason or "").strip() or "unknown"
        exit_reason_counter[exit_reason] += 1

        entry_mode = str(trade.metadata.get("entry_mode", "")).strip() or "unknown"
        entry_mode_counter[entry_mode] += 1

    return {
        "long_closed": long_closed,
        "short_closed": short_closed,
        "hour_distribution": {str(hour): int(hour_counter[hour]) for hour in sorted(hour_counter)},
        "weekday_distribution": {
            str(weekday): int(weekday_counter[weekday]) for weekday in sorted(weekday_counter)
        },
        "exit_reason_breakdown": {
            key: int(value) for key, value in sorted(exit_reason_counter.items(), key=lambda row: row[0])
        },
        "entry_mode_breakdown": {
            key: int(value) for key, value in sorted(entry_mode_counter.items(), key=lambda row: row[0])
        },
    }


def _point_stats(trades: list[Trade]) -> dict[str, float]:
    wins = [item.pnl for item in trades if item.pnl >= 0]
    losses = [item.pnl for item in trades if item.pnl < 0]
    gross_wins = float(sum(wins))
    gross_losses_abs = float(sum(abs(item) for item in losses))
    return {
        "avg_win_points": (gross_wins / len(wins)) if wins else 0.0,
        "avg_loss_points_abs": (gross_losses_abs / len(losses)) if losses else 0.0,
        "gross_wins_points": gross_wins,
        "gross_losses_points_abs": gross_losses_abs,
    }


def _trade_quality_analytics(*, closed_trades: list[Trade]) -> dict[str, dict[str, dict[str, Any]]]:
    reason_groups: dict[str, list[float]] = {}
    setup_quality_groups: dict[str, list[float]] = {}
    signal_quality_groups: dict[str, list[float]] = {}

    for trade in closed_trades:
        value = float(trade.pnl)
        reasons = _extract_reason_codes(trade.metadata)
        if not reasons:
            reasons = ("none",)
        for reason in set(reasons):
            reason_groups.setdefault(reason, []).append(value)

        setup_quality = _safe_quality_value(trade.metadata.get("setup_quality_score"))
        if setup_quality is not None:
            setup_bucket = _setup_quality_bucket(setup_quality)
            setup_quality_groups.setdefault(setup_bucket, []).append(value)

        signal_quality = _safe_quality_value(trade.metadata.get("signal_quality_score"))
        if signal_quality is not None:
            signal_bucket = _signal_quality_bucket(signal_quality)
            signal_quality_groups.setdefault(signal_bucket, []).append(value)

    return {
        "reason_code_expectancy": _expectancy_breakdown(reason_groups, _reason_sort_key),
        "setup_quality_expectancy": _expectancy_breakdown(setup_quality_groups, _setup_bucket_sort_key),
        "signal_quality_expectancy": _expectancy_breakdown(signal_quality_groups, _signal_bucket_sort_key),
    }


def _expectancy_breakdown(
    groups: dict[str, list[float]],
    sorter: Any,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(groups.keys(), key=sorter):
        values = groups.get(key, [])
        if not values:
            continue
        closed = len(values)
        wins = sum(1 for item in values if item >= 0.0)
        losses = closed - wins
        net = float(sum(values))
        gross_wins = float(sum(item for item in values if item >= 0.0))
        gross_losses_abs = float(sum(abs(item) for item in values if item < 0.0))
        out[key] = {
            "closed": closed,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": (wins / closed) * 100.0 if closed > 0 else 0.0,
            "net_pnl": net,
            "expectancy": net / closed if closed > 0 else 0.0,
            "profit_factor": (gross_wins / gross_losses_abs) if gross_losses_abs > 0 else 0.0,
        }
    return out


def _extract_reason_codes(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("reason_codes")
    if isinstance(raw, str):
        candidate = raw.strip()
        return (candidate,) if candidate else tuple()
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    return tuple()


def _safe_quality_value(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _setup_quality_bucket(value: float) -> str:
    if value < 0.45:
        return "<0.45"
    if value < 0.55:
        return "0.45-0.55"
    if value < 0.65:
        return "0.55-0.65"
    if value < 0.75:
        return "0.65-0.75"
    return ">=0.75"


def _signal_quality_bucket(value: float) -> str:
    if value < 0.55:
        return "<0.55"
    if value < 0.65:
        return "0.55-0.65"
    if value < 0.75:
        return "0.65-0.75"
    if value < 0.85:
        return "0.75-0.85"
    return ">=0.85"


def _setup_bucket_sort_key(value: str) -> tuple[int, str]:
    order = {
        "<0.45": 0,
        "0.45-0.55": 1,
        "0.55-0.65": 2,
        "0.65-0.75": 3,
        ">=0.75": 4,
    }
    return (order.get(value, 100), value)


def _signal_bucket_sort_key(value: str) -> tuple[int, str]:
    order = {
        "<0.55": 0,
        "0.55-0.65": 1,
        "0.65-0.75": 2,
        "0.75-0.85": 3,
        ">=0.85": 4,
    }
    return (order.get(value, 100), value)


def _reason_sort_key(value: str) -> str:
    return value


def _as_float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
