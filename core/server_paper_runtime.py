"""Server-oriented runtime for continuous paper-trading."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.config_loader import ConfigError, ConfigLoader
from core.execution_engine import ExecutionEngine
from core.history_preloader import preload_history
from core.instrument_registry import InstrumentRegistry
from core.logger_setup import setup_logging
from core.market_data import Candle, CandleValidationError, create_market_data_client
from core.news_filter import NewsBlackoutFilter
from core.portfolio_engine import PortfolioEngine
from core.session_manager import SessionManager
from core.signal_engine import SignalEngine
from core.stats_engine import StatsEngine
from core.telegram_notifier import TelegramNotifier, TelegramConfig
from core.trade_simulator import TradeSimulator
from core.trading_mode import resolve_primary_timeframe, resolve_trading_mode
from storage.memory_store import MemoryCandleStore
from storage.sqlite_store import SQLiteStore

LOGGER = logging.getLogger("server_runtime")
_STATE_KEY = "server_paper_runtime_state_v1"


@dataclass(frozen=True, slots=True)
class ServerRuntimeConfig:
    mode: str
    polling_interval_sec: int
    heartbeat_enabled: bool
    heartbeat_interval_min: int
    daily_report_enabled: bool
    daily_report_time: str
    timezone: str
    dedup_enabled: bool
    restart_recovery_enabled: bool
    weekly_report_enabled: bool
    debug_pipeline: bool = False
    debug_pipeline_every_updates: int = 50

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "ServerRuntimeConfig":
        section = params.get("runtime", {})
        if not isinstance(section, dict):
            section = {}
        tz_name = str(section.get("timezone", params.get("timezone", "Europe/Moscow"))).strip() or "Europe/Moscow"
        return cls(
            mode=str(section.get("mode", "server_paper")).strip() or "server_paper",
            polling_interval_sec=max(5, int(section.get("polling_interval_sec", 30))),
            heartbeat_enabled=_to_bool(section.get("heartbeat_enabled", True), default=True),
            heartbeat_interval_min=max(15, int(section.get("heartbeat_interval_min", 180))),
            daily_report_enabled=_to_bool(section.get("daily_report_enabled", True), default=True),
            daily_report_time=str(section.get("daily_report_time", "23:10")).strip() or "23:10",
            timezone=tz_name,
            dedup_enabled=_to_bool(section.get("dedup_enabled", True), default=True),
            restart_recovery_enabled=_to_bool(section.get("restart_recovery_enabled", True), default=True),
            weekly_report_enabled=_to_bool(section.get("weekly_report_enabled", False), default=False),
            debug_pipeline=_to_bool(section.get("debug_pipeline", False), default=False),
            debug_pipeline_every_updates=max(1, int(section.get("debug_pipeline_every_updates", 50))),
        )

    @property
    def report_time_local(self) -> time:
        return _parse_hhmm(self.daily_report_time)


@dataclass(slots=True)
class RuntimeState:
    last_processed_by_stream: dict[str, str] = field(default_factory=dict)
    daily_reports_sent: set[str] = field(default_factory=set)
    weekly_reports_sent: set[str] = field(default_factory=set)
    last_heartbeat_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "RuntimeState":
        if not isinstance(payload, dict):
            return cls()
        last_processed_raw = payload.get("last_processed_by_stream", {})
        if not isinstance(last_processed_raw, dict):
            last_processed_raw = {}
        daily_raw = payload.get("daily_reports_sent", [])
        weekly_raw = payload.get("weekly_reports_sent", [])
        return cls(
            last_processed_by_stream={
                str(key): str(value)
                for key, value in last_processed_raw.items()
                if str(key).strip() and str(value).strip()
            },
            daily_reports_sent={str(item) for item in daily_raw if str(item).strip()},
            weekly_reports_sent={str(item) for item in weekly_raw if str(item).strip()},
            last_heartbeat_at=_str_or_none(payload.get("last_heartbeat_at")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "last_processed_by_stream": dict(self.last_processed_by_stream),
            "daily_reports_sent": sorted(self.daily_reports_sent),
            "weekly_reports_sent": sorted(self.weekly_reports_sent),
            "last_heartbeat_at": self.last_heartbeat_at,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBootResult:
    exit_code: int
    reason: str


class ServerPaperRuntime:
    def __init__(
        self,
        *,
        app_config: Any,
        runtime_config: ServerRuntimeConfig,
        notifier: TelegramNotifier,
        sqlite_store: SQLiteStore,
        signal_engine: SignalEngine,
        execution_engine: ExecutionEngine,
        portfolio_engine: PortfolioEngine,
        stats_engine: StatsEngine,
        session_manager: SessionManager,
        blackout_filter: NewsBlackoutFilter,
        candle_store: MemoryCandleStore,
        registry: InstrumentRegistry,
        print_every: int,
    ):
        self._app_config = app_config
        self._runtime_config = runtime_config
        self._notifier = notifier
        self._sqlite = sqlite_store
        self._signal_engine = signal_engine
        self._execution_engine = execution_engine
        self._portfolio_engine = portfolio_engine
        self._stats_engine = stats_engine
        self._session_manager = session_manager
        self._blackout_filter = blackout_filter
        self._candle_store = candle_store
        self._registry = registry
        self._print_every = max(1, int(print_every))
        self._state = RuntimeState.from_payload(self._sqlite.load_runtime_state(state_key=_STATE_KEY))
        self._tz = ZoneInfo(self._runtime_config.timezone)
        self._updates_seen = 0
        self._recoverable_error_count = 0
        self._disconnect_count = 0
        self._recovery_count = 0
        self._last_market_data_error: str | None = None

    async def restore_state(self) -> int:
        if not self._runtime_config.restart_recovery_enabled:
            return 0
        open_rows = self._sqlite.load_open_trade_states()
        restored = self._execution_engine.restore_open_trades(open_rows)
        if restored > 0:
            LOGGER.info("Recovered open paper trades: %d", restored)
        return restored

    async def on_market_data_status(self, status: str, payload: dict[str, Any]) -> None:
        if status == "disconnect":
            self._disconnect_count += 1
            self._last_market_data_error = str(payload.get("error", "unknown"))
            LOGGER.warning("Market-data disconnect: %s", payload)
            await self._notify_system_text(
                category="runtime_disconnect",
                key=f"runtime_disconnect:{payload.get('attempt', 0)}:{self._last_market_data_error}",
                text=f"DATA DISCONNECT attempt={payload.get('attempt')} error={self._last_market_data_error}",
            )
            return
        if status == "connected" and payload.get("recovered"):
            self._recovery_count += 1
            LOGGER.info("Market-data stream recovered: %s", payload)
            await self._notify_system_text(
                category="runtime_recovered",
                key=f"runtime_recovered:{payload.get('attempt', 0)}",
                text=f"DATA RECOVERED attempt={payload.get('attempt')}",
            )

    async def on_candle(self, candle: Candle) -> None:
        try:
            upsert_result = self._candle_store.upsert(candle)
        except CandleValidationError as exc:
            self._recoverable_error_count += 1
            LOGGER.error("Malformed candle dropped: %s", exc)
            await self._notify_critical("Malformed candle", str(exc))
            return

        if upsert_result == "ignored":
            LOGGER.debug(
                "Ignored out-of-history candle %s %s %s",
                candle.instrument,
                candle.timeframe,
                candle.datetime.isoformat(),
            )
            return

        if not self._should_process_candle(candle):
            LOGGER.debug(
                "Skip duplicate candle %s %s %s",
                candle.instrument,
                candle.timeframe,
                candle.datetime.isoformat(),
            )
            return

        self._updates_seen += 1
        try:
            engine_result = self._signal_engine.process_candle(
                instrument=candle.instrument,
                timeframe=candle.timeframe,
            )
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.exception("Signal engine failure: %s", exc)
            await self._notify_critical("Signal engine failure", str(exc))
            return

        accepted_signals = self._filter_duplicate_signals(engine_result.accepted_signals)
        for signal_obj in accepted_signals:
            self._sqlite.save_signal(signal_obj)
            self._stats_engine.record_signal(instrument=signal_obj.instrument, strategy=signal_obj.strategy)

        selection = self._portfolio_engine.select_signals(
            signals=accepted_signals,
            open_positions=self._execution_engine.positions(),
        )
        for event in selection.events:
            self._stats_engine.record_portfolio_event(event)
            if event.event_type in {"risk_rejected", "allocation_rejected"}:
                await self._notify_portfolio_event_once(event)

        execution_open = self._portfolio_engine.submit_for_execution(
            signals=selection.accepted_signals,
            execution_engine=self._execution_engine,
            timeframe=candle.timeframe,
        )
        for event in execution_open.portfolio_events:
            self._stats_engine.record_portfolio_event(event)
            if event.event_type in {"position_opened", "position_closed", "trade_closed"}:
                await self._notify_portfolio_event_once(event)

        for signal_obj in selection.accepted_signals:
            await self._notify_signal_once(signal_obj)

        self._maybe_log_pipeline_debug(
            candle=candle,
            engine_result=engine_result,
            engine_accepted=len(engine_result.accepted_signals),
            dedup_accepted=len(accepted_signals),
            portfolio_accepted=len(selection.accepted_signals),
            portfolio_rejected=len(selection.rejected),
            executed=len(execution_open.opened_positions),
            filter_reasons=engine_result.rejected_reasons,
            portfolio_reasons=tuple(item.reason for item in selection.rejected),
        )

        instrument_meta = self._registry.get(candle.instrument)
        session_state = self._session_manager.get_state(instrument_meta, candle.datetime)
        blackout_active, blackout_reason = self._blackout_filter.is_blocked(candle.datetime)

        try:
            process_result = self._execution_engine.process_market(
                candle=candle,
                session_active=session_state.is_active,
                blackout_active=blackout_active,
                blackout_reason=blackout_reason,
            )
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.exception("Trade lifecycle failure: %s", exc)
            await self._notify_critical("Trade lifecycle failure", str(exc))
            return

        normalized_events = self._portfolio_engine.normalize_execution_events(
            execution_events=process_result.events,
            execution_engine=self._execution_engine,
        )
        for normalized in normalized_events:
            self._stats_engine.record_portfolio_event(normalized)
            if normalized.event_type in {"position_opened", "position_closed", "trade_closed"}:
                await self._notify_portfolio_event_once(normalized)
        for trade in process_result.closed_trades:
            self._stats_engine.record_trade_closed(trade)

        self._mark_candle_processed(candle)
        await self._maybe_send_heartbeat(now_utc=candle.datetime)
        await self._maybe_send_daily_report(now_utc=candle.datetime)

        if self._updates_seen % self._print_every == 0:
            summary = self._stats_engine.summary().get("global", {})
            LOGGER.info(
                "Runtime snapshot instrument=%s timeframe=%s updates=%d open=%d closed=%d net=%.5f",
                candle.instrument,
                candle.timeframe,
                self._updates_seen,
                self._execution_engine.open_positions_count(),
                int(summary.get("closed", 0)),
                float(summary.get("net_pnl", 0.0)),
            )

    async def periodic_tick(self) -> None:
        now = datetime.now(tz=timezone.utc)
        await self._maybe_send_heartbeat(now_utc=now)
        await self._maybe_send_daily_report(now_utc=now)

    def _should_process_candle(self, candle: Candle) -> bool:
        if not self._runtime_config.dedup_enabled:
            return True
        stream_key = self._stream_key(candle.instrument, candle.timeframe)
        last_iso = self._state.last_processed_by_stream.get(stream_key)
        if not last_iso:
            return True
        try:
            last_dt = _parse_dt(last_iso)
        except ValueError:
            return True
        return candle.datetime > last_dt

    def _mark_candle_processed(self, candle: Candle) -> None:
        stream_key = self._stream_key(candle.instrument, candle.timeframe)
        self._state.last_processed_by_stream[stream_key] = candle.datetime.isoformat()
        self._save_state()

    def _filter_duplicate_signals(self, signals: tuple[Any, ...]) -> tuple[Any, ...]:
        if not self._runtime_config.dedup_enabled:
            return tuple(signals)
        accepted: list[Any] = []
        for signal_obj in signals:
            if self._sqlite.signal_exists(signal_obj.signal_id):
                LOGGER.info(
                    "Skip duplicate signal by id signal_id=%s instrument=%s strategy=%s",
                    signal_obj.signal_id,
                    signal_obj.instrument,
                    signal_obj.strategy,
                )
                continue
            if self._sqlite.trade_origin_exists(
                instrument=signal_obj.instrument,
                strategy=signal_obj.strategy,
                created_at_iso=signal_obj.timestamp.isoformat(),
            ):
                LOGGER.info(
                    "Skip duplicate signal by origin instrument=%s strategy=%s ts=%s",
                    signal_obj.instrument,
                    signal_obj.strategy,
                    signal_obj.timestamp.isoformat(),
                )
                continue
            accepted.append(signal_obj)
        return tuple(accepted)

    def _maybe_log_pipeline_debug(
        self,
        *,
        candle: Candle,
        engine_result: Any,
        engine_accepted: int,
        dedup_accepted: int,
        portfolio_accepted: int,
        portfolio_rejected: int,
        executed: int,
        filter_reasons: tuple[str, ...],
        portfolio_reasons: tuple[str, ...],
    ) -> None:
        if not self._runtime_config.debug_pipeline:
            return
        should_log = (
            (self._updates_seen % self._runtime_config.debug_pipeline_every_updates) == 0
            or engine_result.raw_signals > 0
            or portfolio_rejected > 0
            or portfolio_accepted > 0
            or executed > 0
        )
        if not should_log:
            return
        LOGGER.info(
            "Pipeline debug instrument=%s timeframe=%s bar=%s bars_seen=%d raw=%d "
            "filter_rejected=%d engine_accepted=%d dedup_dropped=%d portfolio_rejected=%d "
            "portfolio_accepted=%d executed=%d filter_reasons=%s portfolio_reasons=%s",
            candle.instrument,
            candle.timeframe,
            candle.datetime.isoformat(),
            int(getattr(engine_result, "bars_seen", 0)),
            int(getattr(engine_result, "raw_signals", 0)),
            int(getattr(engine_result, "filter_rejected", 0)),
            engine_accepted,
            max(0, engine_accepted - dedup_accepted),
            portfolio_rejected,
            portfolio_accepted,
            executed,
            list(filter_reasons[:5]),
            list(portfolio_reasons[:5]),
        )

    async def _notify_signal_once(self, signal_obj: Any) -> None:
        key = f"signal:{signal_obj.instrument}:{signal_obj.strategy}:{signal_obj.timestamp.isoformat()}"
        if self._sqlite.runtime_notification_sent(notification_key=key):
            return
        try:
            await self._notifier.notify_signal(signal_obj)
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.warning("Notifier signal send failed key=%s error=%s", key, exc)
            return
        self._sqlite.mark_runtime_notification_sent(
            notification_key=key,
            category="signal",
            payload={"signal_id": signal_obj.signal_id},
        )

    async def _notify_portfolio_event_once(self, event: Any) -> None:
        key = (
            f"portfolio_event:{event.event_type}:{event.trade_id or ''}:"
            f"{event.signal_id or ''}:{event.event_time.isoformat()}"
        )
        if self._sqlite.runtime_notification_sent(notification_key=key):
            return
        try:
            await self._notifier.notify_portfolio_event(event)
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.warning("Notifier portfolio event send failed key=%s error=%s", key, exc)
            return
        self._sqlite.mark_runtime_notification_sent(
            notification_key=key,
            category="portfolio_event",
            payload={"event_type": event.event_type},
        )

    async def _notify_system_text(self, *, category: str, key: str, text: str) -> bool:
        if self._sqlite.runtime_notification_sent(notification_key=key):
            return True
        try:
            await self._notifier.notify_text(text, category=category)
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.warning("Notifier text send failed key=%s category=%s error=%s", key, category, exc)
            return False
        self._sqlite.mark_runtime_notification_sent(
            notification_key=key,
            category=category,
            payload={},
        )
        return True

    async def _notify_critical(self, title: str, details: str) -> None:
        try:
            await self._notifier.notify_critical(title, details)
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.warning("Notifier critical send failed title=%s error=%s", title, exc)

    async def _maybe_send_heartbeat(self, *, now_utc: datetime) -> None:
        if not self._runtime_config.heartbeat_enabled:
            return
        last_iso = self._state.last_heartbeat_at
        if last_iso:
            try:
                last = _parse_dt(last_iso)
            except ValueError:
                last = None
            if last is not None:
                min_delta = timedelta(minutes=self._runtime_config.heartbeat_interval_min)
                if now_utc < (last + min_delta):
                    return

        open_positions = self._execution_engine.open_positions_count()
        enabled_instruments = [item.symbol for item in self._registry.enabled()]
        msg = (
            "HEARTBEAT\n"
            f"Mode: {resolve_trading_mode(self._app_config.params).value}\n"
            f"Runtime: {self._runtime_config.mode}\n"
            f"Instruments: {', '.join(enabled_instruments)}\n"
            f"Open positions: {open_positions}\n"
            f"Last processed: {self._latest_processed_ts() or 'n/a'}\n"
            f"Recoverable errors: {self._recoverable_error_count}\n"
            f"Disconnects/recoveries: {self._disconnect_count}/{self._recovery_count}"
        )
        try:
            await self._notifier.notify_text(msg, category="heartbeat")
        except Exception as exc:
            self._recoverable_error_count += 1
            LOGGER.warning("Notifier heartbeat send failed error=%s", exc)
            return
        self._state.last_heartbeat_at = now_utc.isoformat()
        self._save_state()

    async def _maybe_send_daily_report(self, *, now_utc: datetime) -> None:
        if not self._runtime_config.daily_report_enabled:
            return
        now_local = now_utc.astimezone(self._tz)
        report_date = now_local.date()
        date_key = report_date.isoformat()
        if now_local.timetz().replace(tzinfo=None) < self._runtime_config.report_time_local:
            return
        if date_key in self._state.daily_reports_sent:
            return

        report = self._build_daily_report(report_date=report_date)
        messages = _format_daily_report_messages(report=report)
        sent_all = True
        for idx, text in enumerate(messages, start=1):
            sent = await self._notify_system_text(
                category=f"daily_report_{idx}",
                key=f"daily_report_part:{date_key}:{idx}",
                text=text,
            )
            if not sent:
                sent_all = False

        if not sent_all:
            return
        self._state.daily_reports_sent.add(date_key)
        self._save_state()
        self._sqlite.mark_runtime_notification_sent(
            notification_key=f"daily_report:{date_key}",
            category="daily_report",
            payload={"date": date_key},
        )

    def _build_daily_report(self, *, report_date: date) -> dict[str, Any]:
        start_local = datetime.combine(report_date, time(0, 0, 0), tzinfo=self._tz)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)
        day_trades = self._sqlite.load_trade_rows_closed_between(
            start_iso=start_utc.isoformat(),
            end_iso=end_utc.isoformat(),
        )
        day_signals = self._sqlite.load_signal_rows_between(
            start_iso=start_utc.isoformat(),
            end_iso=end_utc.isoformat(),
        )
        all_trades = self._sqlite.load_all_trade_rows()

        realized = sum(float(row.get("net_pnl", 0.0) or 0.0) for row in day_trades)
        new_trades = [
            row
            for row in all_trades
            if _is_in_range(row.get("created_at"), start_utc, end_utc)
        ]
        closed_trades = [
            row
            for row in day_trades
            if row.get("closed_at")
        ]
        open_positions = list(self._execution_engine.positions())
        unrealized = self._unrealized_pnl(open_positions)
        strategy_bucket = _bucket_day_trades(rows=closed_trades, key="strategy")
        instrument_bucket = _bucket_day_trades(rows=closed_trades, key="instrument")

        exposure = self._portfolio_engine.risk_manager.current_exposure(open_positions=open_positions)
        portfolio_stats = self._stats_engine.summary().get("portfolio", {})
        risk_by_group_pct = {
            key: (value / max(self._portfolio_engine.risk_manager.config.account_size, 1e-9)) * 100.0
            for key, value in exposure.risk_money_by_group.items()
        }
        return {
            "date": report_date.isoformat(),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "equity_change_proxy": realized + unrealized,
            "new_trades": len(new_trades),
            "closed_trades": len(closed_trades),
            "open_positions": [_position_row(item, self._candle_store) for item in open_positions],
            "closed_trade_rows": [_closed_trade_row(item) for item in closed_trades],
            "signals_by_strategy": _count_rows(day_signals, key="strategy"),
            "strategy_summary": strategy_bucket,
            "instrument_summary": instrument_bucket,
            "risk_snapshot": {
                "open_positions": exposure.total_positions,
                "total_risk_money": exposure.total_risk_money,
                "total_risk_pct": exposure.total_risk_pct,
                "risk_money_by_strategy": exposure.risk_money_by_strategy,
                "risk_money_by_instrument": exposure.risk_money_by_instrument,
                "risk_money_by_group": exposure.risk_money_by_group,
                "risk_by_strategy": exposure.risk_by_strategy,
                "risk_by_instrument": exposure.risk_by_instrument,
                "risk_by_group": risk_by_group_pct,
                "positions_by_strategy": exposure.positions_by_strategy,
                "positions_by_instrument": exposure.positions_by_instrument,
                "positions_by_group": exposure.positions_by_group,
                "risk_reject_reasons": portfolio_stats.get("risk_reject_reasons", {}),
            },
            "operational": {
                "recoverable_errors": self._recoverable_error_count,
                "disconnects": self._disconnect_count,
                "recoveries": self._recovery_count,
                "last_market_data_error": self._last_market_data_error or "",
                "last_processed": self._latest_processed_ts() or "",
            },
        }

    def _unrealized_pnl(self, positions: list[Any]) -> float:
        total = 0.0
        for position in positions:
            latest = self._candle_store.latest(position.instrument, position.timeframe or "")
            if latest is None:
                continue
            if position.side.value == "LONG":
                total += (latest.close - position.entry_price) * position.size
            else:
                total += (position.entry_price - latest.close) * position.size
        return total

    def _latest_processed_ts(self) -> str | None:
        if not self._state.last_processed_by_stream:
            return None
        values = sorted(self._state.last_processed_by_stream.values())
        return values[-1] if values else None

    def _save_state(self) -> None:
        self._sqlite.save_runtime_state(state_key=_STATE_KEY, payload=self._state.to_payload())

    @staticmethod
    def _stream_key(instrument: str, timeframe: str) -> str:
        return f"{instrument}:{timeframe}"


async def run_server_paper(
    *,
    config_dir: Path,
    log_dir: Path,
    run_seconds: int = 0,
    print_every: int = 10,
) -> RuntimeBootResult:
    setup_logging(log_dir)
    LOGGER.info("Server paper runtime start")
    _load_env_file(Path(__file__).resolve().parents[1] / ".env")

    try:
        app_config = ConfigLoader(config_dir).load()
    except ConfigError as exc:
        LOGGER.critical("Configuration error: %s", exc)
        notifier = await _build_notifier({})
        try:
            await notifier.notify_critical("Configuration error", str(exc))
        finally:
            await notifier.close()
        return RuntimeBootResult(exit_code=2, reason="config_error")

    runtime_cfg = ServerRuntimeConfig.from_params(app_config.params)
    notifier = await _build_notifier(app_config.params)
    registry = InstrumentRegistry.from_config(app_config)
    store = MemoryCandleStore(history_depth=app_config.history_depth)
    blackout_filter = NewsBlackoutFilter(app_config.blackout_windows)
    session_manager = SessionManager()

    try:
        sqlite_store = SQLiteStore(_resolve_db_path(app_config.params))
    except Exception as exc:
        LOGGER.exception("SQLite initialization failed: %s", exc)
        await notifier.notify_critical("SQLite initialization failed", str(exc))
        await notifier.close()
        return RuntimeBootResult(exit_code=3, reason="sqlite_error")

    stats_engine = StatsEngine()
    trade_simulator = TradeSimulator(
        params=app_config.params,
        logger=logging.getLogger("trade_simulator"),
        storage=sqlite_store,
    )
    execution_engine = ExecutionEngine(simulator=trade_simulator)
    portfolio_engine = PortfolioEngine(params=app_config.params)
    signal_engine = SignalEngine(
        registry=registry,
        store=store,
        params=app_config.params,
        blackout_filter=blackout_filter,
        logger=logging.getLogger("signal_engine"),
    )

    runtime = ServerPaperRuntime(
        app_config=app_config,
        runtime_config=runtime_cfg,
        notifier=notifier,
        sqlite_store=sqlite_store,
        signal_engine=signal_engine,
        execution_engine=execution_engine,
        portfolio_engine=portfolio_engine,
        stats_engine=stats_engine,
        session_manager=session_manager,
        blackout_filter=blackout_filter,
        candle_store=store,
        registry=registry,
        print_every=print_every,
    )

    restored = await runtime.restore_state()
    with contextlib.suppress(Exception):
        await notifier.notify_text(
            (
                "SERVER PAPER STARTED\n"
                f"Runtime mode: {runtime_cfg.mode}\n"
                f"Trading mode: {resolve_trading_mode(app_config.params).value}\n"
                f"Restored open trades: {restored}\n"
                f"Enabled instruments: {len(registry.enabled())}"
            ),
            category="startup",
        )

    mode = _resolve_market_data_mode(app_config.params)
    runtime_timeframe = resolve_primary_timeframe(
        params=app_config.params,
        default_timeframe=app_config.default_timeframe,
    )
    market_data_params = app_config.params.get("market_data", {})
    if not isinstance(market_data_params, dict):
        market_data_params = {}
    token = os.getenv("INVEST_TOKEN", "")
    if mode == "t_invest" and not token.strip():
        LOGGER.error("MARKET_DATA_MODE=t_invest but INVEST_TOKEN is empty; switching to demo")
        with contextlib.suppress(Exception):
            await notifier.notify_critical(
                "Live mode disabled",
                "MARKET_DATA_MODE=t_invest but INVEST_TOKEN is empty. Switched to demo mode.",
            )
        mode = "demo"

    if mode == "t_invest":
        try:
            preload_report = await preload_history(
                token=token,
                registry=registry,
                store=store,
                params=app_config.params,
                timeframe=runtime_timeframe,
                logger=LOGGER,
            )
            if preload_report.enabled:
                LOGGER.info(
                    "History preload requested=%d attempted=%d processed=%d",
                    preload_report.requested_bars,
                    preload_report.instruments_attempted,
                    preload_report.processed_candles,
                )
        except Exception as exc:
            LOGGER.exception("History preload failed: %s", exc)

    client = create_market_data_client(
        mode=mode,
        token=token,
        registry=registry,
        params=market_data_params,
        timeframe=runtime_timeframe,
        logger=logging.getLogger("market_data"),
        status_handler=runtime.on_market_data_status,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    async def _scheduler_loop() -> None:
        while not stop_event.is_set():
            try:
                await runtime.periodic_tick()
            except Exception as exc:
                LOGGER.exception("Scheduler tick failed: %s", exc)
                with contextlib.suppress(Exception):
                    await notifier.notify_critical("Scheduler failure", str(exc))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=runtime_cfg.polling_interval_sec)
            except asyncio.TimeoutError:
                continue

    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(client.run(on_candle=runtime.on_candle, stop_event=stop_event), name="market-data"),
        asyncio.create_task(_scheduler_loop(), name="runtime-scheduler"),
    ]

    if run_seconds > 0:
        async def _auto_stop() -> None:
            await asyncio.sleep(run_seconds)
            stop_event.set()
        tasks.append(asyncio.create_task(_auto_stop(), name="auto-stop"))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for finished in done:
        if finished.cancelled():
            continue
        exc = finished.exception()
        if exc:
            LOGGER.error(
                "Runtime task failed name=%s error=%s",
                finished.get_name(),
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            with contextlib.suppress(Exception):
                await notifier.notify_critical("Runtime task failed", f"{finished.get_name()}: {exc}")
            stop_event.set()

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    stats_summary = _with_portfolio_risk_snapshot(
        summary=stats_engine.summary(),
        portfolio_engine=portfolio_engine,
        execution_engine=execution_engine,
    )
    sqlite_store.save_stats_snapshot(datetime.now(tz=timezone.utc), stats_summary)
    sqlite_store.close()

    if notifier.enabled and notifier.send_shutdown_summary:
        with contextlib.suppress(Exception):
            await notifier.notify_daily_summary(stats_summary, execution_engine.open_positions_count())
    with contextlib.suppress(Exception):
        await notifier.notify_text("SERVER PAPER STOPPED", category="shutdown")
    with contextlib.suppress(Exception):
        await notifier.close()

    LOGGER.info("Server paper runtime stop")
    return RuntimeBootResult(exit_code=0, reason="ok")


async def _build_notifier(params: dict[str, Any]) -> TelegramNotifier:
    config = TelegramConfig.from_sources(env=os.environ, params=params)
    notifier = TelegramNotifier(config=config, logger=logging.getLogger("telegram"))
    await notifier.start()
    return notifier


def _load_env_file(path: Path) -> None:
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


def _resolve_market_data_mode(config_params: dict[str, Any]) -> str:
    env_value = os.getenv("MARKET_DATA_MODE", "").strip().lower()
    if env_value:
        return env_value
    market_data_cfg = config_params.get("market_data", {})
    if isinstance(market_data_cfg, dict):
        mode = str(market_data_cfg.get("mode", "demo")).strip().lower()
        return mode or "demo"
    return "demo"


def _resolve_db_path(config_params: dict[str, Any]) -> Path:
    env_db = os.getenv("DB_PATH", "").strip()
    if env_db:
        candidate = Path(env_db)
    else:
        storage_cfg = config_params.get("storage", {})
        if isinstance(storage_cfg, dict):
            value = str(storage_cfg.get("db_path", "signals.db")).strip() or "signals.db"
        else:
            value = "signals.db"
        candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def _to_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_hhmm(value: str) -> time:
    text = str(value).strip()
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.time()
        except ValueError:
            continue
    return time(23, 10, 0)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_in_range(value: Any, start_utc: datetime, end_utc: datetime) -> bool:
    if not value:
        return False
    try:
        point = _parse_dt(str(value))
    except ValueError:
        return False
    return start_utc <= point < end_utc


def _position_row(position: Any, store: MemoryCandleStore) -> dict[str, Any]:
    latest = store.latest(position.instrument, position.timeframe or "")
    mark = latest.close if latest is not None else position.entry_price
    if position.side.value == "LONG":
        pnl = (mark - position.entry_price) * position.size
    else:
        pnl = (position.entry_price - mark) * position.size
    holding_hours = (datetime.now(tz=timezone.utc) - position.opened_at).total_seconds() / 3600.0
    metadata = position.metadata if isinstance(position.metadata, dict) else {}
    return {
        "instrument": position.instrument,
        "strategy": position.strategy_id,
        "side": position.side.value,
        "entry_price": position.entry_price,
        "mark_price": mark,
        "size": position.size,
        "qty": position.size,
        "stop_loss": _as_float(position.stop_loss, 0.0),
        "take_profit": _as_float(position.take_profit, 0.0),
        "planned_risk_money": _as_float(metadata.get("planned_risk_money"), 0.0),
        "planned_risk_pct": _as_float(metadata.get("planned_risk_pct"), 0.0),
        "expected_rr": _as_float(metadata.get("post_fill_rr"), 0.0),
        "pnl": pnl,
        "holding_hours": max(0.0, holding_hours),
    }


def _exit_price_from_row(row: dict[str, Any]) -> float:
    metadata = row.get("metadata_json", {})
    if isinstance(metadata, dict):
        raw = metadata.get("last_exit_price")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    status = str(row.get("status", ""))
    if status == "tp2_hit":
        return float(row.get("tp2", 0.0))
    if status == "sl_hit":
        return float(row.get("current_stop", row.get("stop_loss", 0.0)))
    return float(row.get("entry_fill_price") or row.get("entry") or 0.0)


def _closed_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "instrument": str(row.get("instrument", "")),
        "strategy": str(row.get("strategy", "")),
        "side": str(row.get("direction", "")),
        "entry": float(row.get("entry_fill_price") or row.get("entry") or 0.0),
        "exit": _exit_price_from_row(row),
        "qty": float(row.get("quantity", 0.0) or 0.0),
        "planned_risk_money": _as_float(metadata.get("planned_risk_money"), 0.0),
        "planned_risk_pct": _as_float(metadata.get("planned_risk_pct"), 0.0),
        "gross_pnl": float(row.get("gross_pnl", 0.0) or 0.0),
        "fees": float(row.get("fees_paid", 0.0) or 0.0),
        "net_pnl": float(row.get("net_pnl", 0.0) or 0.0),
        "r_multiple": float(row.get("r_multiple", 0.0) or 0.0),
        "reason": str(row.get("exit_reason", "")),
    }


def _bucket_day_trades(*, rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = str(row.get(key, "")).strip() or "unknown"
        item = buckets.setdefault(bucket, {"trades": 0.0, "wins": 0.0, "losses": 0.0, "pnl": 0.0})
        pnl = float(row.get("net_pnl", 0.0) or 0.0)
        item["trades"] += 1.0
        item["pnl"] += pnl
        if pnl >= 0.0:
            item["wins"] += 1.0
        else:
            item["losses"] += 1.0
    for item in buckets.values():
        trades = max(1.0, item["trades"])
        item["win_rate"] = item["wins"] / trades
    return buckets


def _count_rows(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get(key, "")).strip() or "unknown"
        out[bucket] = out.get(bucket, 0) + 1
    return out


def _format_daily_report_messages(*, report: dict[str, Any]) -> list[str]:
    head = (
        "DAILY PAPER REPORT\n"
        f"Date: {report['date']}\n"
        f"Realized PnL: {report['realized_pnl']:.4f}\n"
        f"Unrealized PnL: {report['unrealized_pnl']:.4f}\n"
        f"Equity change (proxy): {report['equity_change_proxy']:.4f}\n"
        f"New trades: {report['new_trades']}\n"
        f"Closed trades: {report['closed_trades']}"
    )

    open_rows = report["open_positions"]
    if open_rows:
        lines = ["OPEN POSITIONS"]
        for row in open_rows[:12]:
            lines.append(
                f"{row['instrument']} {row['strategy']} {row['side']} "
                f"entry={row['entry_price']:.4f} mark={row['mark_price']:.4f} "
                f"stop={row['stop_loss']:.4f} take={row['take_profit']:.4f} "
                f"qty={row['qty']:.2f} risk={row['planned_risk_money']:.2f} ({row['planned_risk_pct']:.3f}%) "
                f"rr={row['expected_rr']:.2f} pnl={row['pnl']:.4f} hold_h={row['holding_hours']:.1f}"
            )
        open_block = "\n".join(lines)
    else:
        open_block = "OPEN POSITIONS\nnone"

    closed_rows = report["closed_trade_rows"]
    lines = ["CLOSED TRADES"]
    for row in closed_rows[:20]:
        lines.append(
            f"{row['instrument']} {row['strategy']} {row['side']} "
            f"entry={row['entry']:.4f} exit={row['exit']:.4f} qty={row['qty']:.2f} "
            f"risk={row['planned_risk_money']:.2f} ({row['planned_risk_pct']:.3f}%) "
            f"gross={row['gross_pnl']:.4f} fees={row['fees']:.4f} net={row['net_pnl']:.4f} "
            f"R={row['r_multiple']:.3f} reason={row['reason']}"
        )
    if len(closed_rows) > 20:
        lines.append(f"... and {len(closed_rows) - 20} more")
    closed_block = "\n".join(lines)

    strategy_summary = report["strategy_summary"]
    instrument_summary = report["instrument_summary"]
    risk = report["risk_snapshot"]
    operational = report["operational"]
    tail = (
        "SUMMARY\n"
        f"Signals by strategy: {report['signals_by_strategy']}\n"
        f"PnL by strategy: { {k: round(v['pnl'], 4) for k, v in strategy_summary.items()} }\n"
        f"PnL by instrument: { {k: round(v['pnl'], 4) for k, v in instrument_summary.items()} }\n"
        f"Risk total money: {risk['total_risk_money']:.4f}\n"
        f"Risk total pct: {risk['total_risk_pct']:.4f}\n"
        f"Risk by instrument: { {k: round(v, 4) for k, v in risk['risk_by_instrument'].items()} }\n"
        f"Risk by strategy: { {k: round(v, 4) for k, v in risk['risk_by_strategy'].items()} }\n"
        f"Risk by group: { {k: round(v, 4) for k, v in risk['risk_by_group'].items()} }\n"
        f"Risk rejects: {risk['risk_reject_reasons']}\n"
        f"Open positions: {risk['open_positions']}\n"
        f"Ops errors/disconnects/recoveries: "
        f"{operational['recoverable_errors']}/{operational['disconnects']}/{operational['recoveries']}\n"
        f"Last processed: {operational['last_processed'] or 'n/a'}"
    )
    return [head, open_block, closed_block, tail]


def _with_portfolio_risk_snapshot(
    *,
    summary: dict[str, Any],
    portfolio_engine: PortfolioEngine,
    execution_engine: ExecutionEngine,
) -> dict[str, Any]:
    payload = dict(summary)
    exposure = portfolio_engine.risk_manager.current_exposure(open_positions=execution_engine.positions())
    account_size = max(portfolio_engine.risk_manager.config.account_size, 1e-9)
    payload["portfolio_risk_snapshot"] = {
        "open_positions": exposure.total_positions,
        "total_risk_money": exposure.total_risk_money,
        "total_risk_pct": exposure.total_risk_pct,
        "risk_by_instrument": dict(exposure.risk_by_instrument),
        "risk_by_strategy": dict(exposure.risk_by_strategy),
        "risk_by_group": {
            key: (value / account_size) * 100.0 for key, value in exposure.risk_money_by_group.items()
        },
    }
    return payload


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
