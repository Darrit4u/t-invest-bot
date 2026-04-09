"""Entry point for Stage 4: simulator + stats + Telegram notifications."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config_loader import ConfigError, ConfigLoader
from core.instrument_registry import InstrumentRegistry
from core.logger_setup import setup_logging
from core.market_data import Candle, CandleValidationError, create_market_data_client
from core.news_filter import NewsBlackoutFilter
from core.session_manager import SessionManager
from core.signal_engine import SignalEngine
from core.stats_engine import StatsEngine
from core.telegram_notifier import TelegramConfig, TelegramNotifier
from core.trade_simulator import TradeSimulator
from storage.memory_store import MemoryCandleStore
from storage.sqlite_store import SQLiteStore


LOGGER = logging.getLogger("app")
CLOSE_EVENTS = {"tp2_hit", "sl_hit", "expired", "cancelled_by_news", "cancelled_by_session_end"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday futures signal engine - Stage 4")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent / "config",
        help="Path to YAML configuration directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).parent / "logs",
        help="Path for log files",
    )
    parser.add_argument(
        "--run-seconds",
        type=int,
        default=0,
        help="Optional auto-stop timeout. 0 means run forever.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="How often to log memory/regime snapshots (in candle updates)",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE entries from .env without external dependencies."""

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


def resolve_market_data_mode(config_params: dict[str, Any]) -> str:
    env_value = os.getenv("MARKET_DATA_MODE", "").strip().lower()
    if env_value:
        return env_value

    market_data_cfg = config_params.get("market_data", {})
    if isinstance(market_data_cfg, dict):
        mode = str(market_data_cfg.get("mode", "demo")).strip().lower()
        return mode or "demo"

    return "demo"


def resolve_db_path(config_params: dict[str, Any]) -> Path:
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
    return Path(__file__).parent / candidate


async def build_notifier(params: dict[str, Any]) -> TelegramNotifier:
    config = TelegramConfig.from_sources(env=os.environ, params=params)
    notifier = TelegramNotifier(config=config, logger=logging.getLogger("telegram"))
    await notifier.start()
    return notifier


async def run() -> int:
    args = parse_args()

    setup_logging(args.log_dir)
    LOGGER.info("Application start")

    load_env_file(Path(__file__).parent / ".env")

    try:
        app_config = ConfigLoader(args.config_dir).load()
    except ConfigError as exc:
        LOGGER.critical("Configuration error: %s", exc)
        bootstrap_notifier = await build_notifier({})
        try:
            await bootstrap_notifier.notify_critical("Configuration error", str(exc))
        finally:
            await bootstrap_notifier.close()
        return 2

    notifier = await build_notifier(app_config.params)

    LOGGER.info(
        "Configs loaded: instruments=%d enabled=%d blackout_windows=%d",
        len(app_config.instruments),
        sum(1 for item in app_config.instruments.values() if item.enabled),
        len(app_config.blackout_windows),
    )

    registry = InstrumentRegistry.from_config(app_config)
    store = MemoryCandleStore(history_depth=app_config.history_depth)
    blackout_filter = NewsBlackoutFilter(app_config.blackout_windows)
    session_manager = SessionManager()

    try:
        db_path = resolve_db_path(app_config.params)
        sqlite_store = SQLiteStore(db_path)
    except Exception as exc:
        LOGGER.exception("SQLite initialization failed: %s", exc)
        await notifier.notify_critical("SQLite initialization failed", str(exc))
        await notifier.close()
        return 3

    LOGGER.info("SQLite storage path: %s", sqlite_store.path)

    stats_engine = StatsEngine()
    trade_simulator = TradeSimulator(
        params=app_config.params,
        logger=logging.getLogger("trade_simulator"),
        storage=sqlite_store,
    )

    signal_engine = SignalEngine(
        registry=registry,
        store=store,
        params=app_config.params,
        blackout_filter=blackout_filter,
        logger=logging.getLogger("signal_engine"),
    )

    async def safe_db_write(action: str, fn: Any, *fn_args: Any) -> None:
        try:
            fn(*fn_args)
        except Exception as exc:
            LOGGER.exception("DB write failed action=%s error=%s", action, exc)
            await notifier.notify_critical("Database write failure", f"{action}: {exc}")

    async def on_market_data_status(status: str, payload: dict[str, Any]) -> None:
        if status == "disconnect":
            details = f"attempt={payload.get('attempt')} error={payload.get('error', 'unknown')}"
            await notifier.notify_critical("Market data disconnected", details)
            return

        if status == "connected" and payload.get("mode") == "t_invest":
            if payload.get("recovered"):
                await notifier.notify_text(
                    f"API feed recovered on attempt {payload.get('attempt')}",
                    category="feed_recovered",
                )
            return

    mode = resolve_market_data_mode(app_config.params)
    market_data_params = app_config.params.get("market_data", {})
    if not isinstance(market_data_params, dict):
        market_data_params = {}

    token = os.getenv("INVEST_TOKEN", "")
    if mode == "t_invest" and not token.strip():
        LOGGER.error("MARKET_DATA_MODE=t_invest but INVEST_TOKEN is empty; switching to demo mode")
        await notifier.notify_critical(
            "Live mode disabled",
            "MARKET_DATA_MODE=t_invest but INVEST_TOKEN is empty. Switched to demo mode.",
        )
        mode = "demo"

    client = create_market_data_client(
        mode=mode,
        token=token,
        registry=registry,
        params=market_data_params,
        timeframe=app_config.default_timeframe,
        logger=logging.getLogger("market_data"),
        status_handler=on_market_data_status,
    )

    LOGGER.info("Market data mode: %s", mode)

    if notifier.enabled and notifier.send_startup_message:
        await notifier.notify_text(
            f"Bot started. Mode={mode}. Enabled instruments={len(registry.enabled())}",
            category="startup",
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    updates_seen = 0

    async def on_candle(candle: Candle) -> None:
        nonlocal updates_seen
        try:
            upsert_result = store.upsert(candle)
        except CandleValidationError as exc:
            LOGGER.error("Rejected malformed candle: %s", exc)
            await notifier.notify_critical("Malformed candle rejected", str(exc))
            return

        if upsert_result == "ignored":
            LOGGER.warning(
                "Ignored out-of-history candle %s %s %s",
                candle.instrument,
                candle.timeframe,
                candle.datetime.isoformat(),
            )
            return

        updates_seen += 1

        try:
            engine_result = signal_engine.process_candle(
                instrument=candle.instrument,
                timeframe=candle.timeframe,
            )
        except Exception as exc:
            LOGGER.exception("Signal engine failed: %s", exc)
            await notifier.notify_critical("Signal engine failure", str(exc))
            return

        for signal_obj in engine_result.accepted_signals:
            await safe_db_write("save_signal", sqlite_store.save_signal, signal_obj)
            stats_engine.record_signal(instrument=signal_obj.instrument, strategy=signal_obj.strategy)
            registration_events = trade_simulator.register_signal(signal_obj, timeframe=candle.timeframe)
            for event in registration_events:
                stats_engine.record_event(event)
                trade_for_event = trade_simulator.get_trade(event.trade_id)
                if event.event_type != "new_signal":
                    await notifier.notify_trade_event(event, trade_for_event)

            await notifier.notify_signal(signal_obj)

            LOGGER.info(
                "Signal accepted | instrument=%s strategy=%s regime=%s direction=%s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f meta=%s",
                signal_obj.instrument,
                signal_obj.strategy,
                signal_obj.regime.value,
                signal_obj.direction.value,
                signal_obj.entry,
                signal_obj.stop_loss,
                signal_obj.tp1,
                signal_obj.tp2,
                signal_obj.metadata,
            )

        instrument_meta = registry.get(candle.instrument)
        session_state = session_manager.get_state(instrument_meta, candle.datetime)
        blackout_active, blackout_reason = blackout_filter.is_blocked(candle.datetime)

        try:
            trade_events = trade_simulator.process_candle(
                candle=candle,
                session_active=session_state.is_active,
                blackout_active=blackout_active,
                blackout_reason=blackout_reason,
            )
        except Exception as exc:
            LOGGER.exception("Trade simulator failure: %s", exc)
            await notifier.notify_critical("Trade simulator failure", str(exc))
            return

        for event in trade_events:
            stats_engine.record_event(event)
            trade = trade_simulator.get_trade(event.trade_id)
            await notifier.notify_trade_event(event, trade)
            if event.event_type in CLOSE_EVENTS and trade is not None:
                stats_engine.record_trade_closed(trade)

        if updates_seen % max(1, args.print_every) == 0:
            recent = store.get_recent(candle.instrument, candle.timeframe, limit=3)
            rows = [
                {
                    "time": item.datetime.isoformat(),
                    "o": round(item.open, 5),
                    "h": round(item.high, 5),
                    "l": round(item.low, 5),
                    "c": round(item.close, 5),
                    "v": round(item.volume, 2),
                }
                for item in recent
            ]
            summary = stats_engine.summary()["global"]
            LOGGER.info(
                "Recent candles %s/%s regime=%s rejected=%d open_trades=%d closed=%d net=%.5f rows=%s",
                candle.instrument,
                candle.timeframe,
                engine_result.regime.value if engine_result.regime else "N/A",
                len(engine_result.rejected_reasons),
                trade_simulator.open_trades_count(),
                int(summary["closed"]),
                float(summary["net_pnl"]),
                rows,
            )

    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(client.run(on_candle=on_candle, stop_event=stop_event), name="market-data")
    ]

    if notifier.enabled and notifier.summary_interval_seconds > 0:

        async def _summary_loop() -> None:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=notifier.summary_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    await notifier.notify_daily_summary(
                        stats_engine.summary(),
                        trade_simulator.open_trades_count(),
                    )

        tasks.append(asyncio.create_task(_summary_loop(), name="telegram-summary"))

    if args.run_seconds > 0:

        async def _auto_stop() -> None:
            await asyncio.sleep(args.run_seconds)
            LOGGER.info("Auto-stop timeout reached (%d seconds)", args.run_seconds)
            stop_event.set()

        tasks.append(asyncio.create_task(_auto_stop(), name="auto-stop"))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for finished in done:
        exc = finished.exception()
        if exc:
            LOGGER.error(
                "Task %s failed: %s",
                finished.get_name(),
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await notifier.notify_critical(
                "Runtime task failed",
                f"task={finished.get_name()} error={exc}",
            )
            stop_event.set()

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    stats_summary = stats_engine.summary()
    await safe_db_write("save_stats_snapshot", sqlite_store.save_stats_snapshot, datetime.now(tz=timezone.utc), stats_summary)
    sqlite_store.close()

    stats = store.stats()
    LOGGER.info(
        "Application stop: updates=%d instruments=%d streams=%d candles=%d",
        updates_seen,
        stats.instruments,
        stats.streams,
        stats.candles,
    )
    LOGGER.info("Final stats summary: %s", stats_summary)

    if notifier.enabled and notifier.send_shutdown_summary:
        await notifier.notify_daily_summary(stats_summary, trade_simulator.open_trades_count())

    with contextlib.suppress(Exception):
        await notifier.close()

    return 0


def main() -> None:
    exit_code = asyncio.run(run())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
