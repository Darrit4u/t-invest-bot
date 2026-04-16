from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.config_loader import ConfigLoader
from core.execution_engine import ExecutionEngine
from core.instrument_registry import InstrumentRegistry
from core.news_filter import NewsBlackoutFilter
from core.portfolio_engine import PortfolioEngine
from core.server_paper_runtime import ServerPaperRuntime, ServerRuntimeConfig, _format_daily_report_messages
from core.session_manager import SessionManager
from core.signal_engine import SignalEngine
from core.stats_engine import StatsEngine
from core.trade_simulator import TradeSimulator
from storage.memory_store import MemoryCandleStore
from storage.sqlite_store import SQLiteStore
from tests.helpers import build_signal, config_dir, make_candle


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def exception(self, *args, **kwargs) -> None:
        return None


class _StubNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.enabled = True
        self.send_shutdown_summary = True

    async def notify_text(self, text: str, category: str = "info") -> None:
        self.messages.append((category, text))

    async def notify_signal(self, signal) -> None:
        self.messages.append(("signal", signal.signal_id))

    async def notify_portfolio_event(self, event) -> None:
        self.messages.append(("portfolio", event.event_type))

    async def notify_critical(self, title: str, details: str) -> None:
        self.messages.append(("critical", f"{title}:{details}"))

    async def notify_daily_summary(self, summary, open_trades) -> None:
        self.messages.append(("daily_summary", str(open_trades)))

    async def close(self) -> None:
        return None


class _FailingNotifier(_StubNotifier):
    async def notify_text(self, text: str, category: str = "info") -> None:
        raise RuntimeError(f"telegram send failed: {category}")

    async def notify_signal(self, signal) -> None:
        raise RuntimeError("telegram signal send failed")

    async def notify_portfolio_event(self, event) -> None:
        raise RuntimeError("telegram portfolio send failed")

    async def notify_critical(self, title: str, details: str) -> None:
        raise RuntimeError("telegram critical send failed")


def _build_runtime(*, sqlite_store: SQLiteStore, notifier: _StubNotifier) -> ServerPaperRuntime:
    cfg = ConfigLoader(config_dir()).load()
    registry = InstrumentRegistry.from_config(cfg)
    store = MemoryCandleStore(history_depth=cfg.history_depth)
    blackout_filter = NewsBlackoutFilter(cfg.blackout_windows)
    signal_engine = SignalEngine(
        registry=registry,
        store=store,
        params=cfg.params,
        blackout_filter=blackout_filter,
        logger=_DummyLogger(),
    )
    simulator = TradeSimulator(params=cfg.params, logger=_DummyLogger(), storage=sqlite_store)
    execution_engine = ExecutionEngine(simulator=simulator)
    return ServerPaperRuntime(
        app_config=cfg,
        runtime_config=ServerRuntimeConfig(
            mode="server_paper",
            polling_interval_sec=10,
            heartbeat_enabled=False,
            heartbeat_interval_min=180,
            daily_report_enabled=True,
            daily_report_time="00:00",
            timezone="Europe/Moscow",
            dedup_enabled=True,
            restart_recovery_enabled=True,
            weekly_report_enabled=False,
        ),
        notifier=notifier,
        sqlite_store=sqlite_store,
        signal_engine=signal_engine,
        execution_engine=execution_engine,
        portfolio_engine=PortfolioEngine(params=cfg.params),
        stats_engine=StatsEngine(),
        session_manager=SessionManager(),
        blackout_filter=blackout_filter,
        candle_store=store,
        registry=registry,
        print_every=10,
    )


class ServerPaperRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_idempotent_candle_checkpoint_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "runtime_state.db")
            notifier = _StubNotifier()
            runtime = _build_runtime(sqlite_store=db, notifier=notifier)

            candle = make_candle(
                1,
                open_=100.0,
                high=101.0,
                low=99.5,
                close=100.5,
                instrument="ES",
                timeframe="1min",
            )
            self.assertTrue(runtime._should_process_candle(candle))  # type: ignore[attr-defined]
            runtime._mark_candle_processed(candle)  # type: ignore[attr-defined]
            self.assertFalse(runtime._should_process_candle(candle))  # type: ignore[attr-defined]

            runtime_after_restart = _build_runtime(sqlite_store=db, notifier=notifier)
            self.assertFalse(runtime_after_restart._should_process_candle(candle))  # type: ignore[attr-defined]
            newer = make_candle(
                2,
                open_=100.5,
                high=101.2,
                low=100.2,
                close=101.0,
                instrument="ES",
                timeframe="1min",
            )
            self.assertTrue(runtime_after_restart._should_process_candle(newer))  # type: ignore[attr-defined]
            db.close()

    async def test_trade_simulator_restore_open_trade_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "restore.db")
            params = {"trade_simulator": {"max_wait_bars": 5, "max_trade_bars": 20}}
            sim1 = TradeSimulator(params=params, logger=_DummyLogger(), storage=db)
            signal = build_signal(
                timestamp=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
                entry=100.0,
                stop_loss=99.0,
                tp1=101.0,
                tp2=102.0,
            )
            sim1.register_signal(signal, timeframe="1min")
            sim1.process_candle(
                candle=make_candle(
                    1,
                    open_=100.0,
                    high=100.6,
                    low=99.8,
                    close=100.4,
                    instrument="ES",
                    timeframe="1min",
                    base=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
                ),
                session_active=True,
                blackout_active=False,
                blackout_reason=None,
            )
            self.assertEqual(sim1.open_trades_count(), 1)

            restored_rows = db.load_open_trade_states()
            sim2 = TradeSimulator(params=params, logger=_DummyLogger(), storage=None)
            restored = sim2.restore_trade_states(restored_rows)
            self.assertEqual(restored, 1)
            self.assertEqual(sim2.open_trades_count(), 1)
            db.close()

    async def test_daily_report_is_sent_once_per_day(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "daily_report.db")
            notifier = _StubNotifier()
            runtime = _build_runtime(sqlite_store=db, notifier=notifier)

            now = datetime(2026, 4, 15, 21, 30, tzinfo=timezone.utc)
            await runtime._maybe_send_daily_report(now_utc=now)  # type: ignore[attr-defined]
            first_count = len([item for item in notifier.messages if item[0].startswith("daily_report_")])
            self.assertGreater(first_count, 0)

            await runtime._maybe_send_daily_report(now_utc=now)  # type: ignore[attr-defined]
            second_count = len([item for item in notifier.messages if item[0].startswith("daily_report_")])
            self.assertEqual(first_count, second_count)
            db.close()

    async def test_telegram_failures_do_not_break_runtime_loop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "daily_report_fail.db")
            notifier = _FailingNotifier()
            runtime = _build_runtime(sqlite_store=db, notifier=notifier)

            now = datetime(2026, 4, 15, 21, 30, tzinfo=timezone.utc)
            await runtime._maybe_send_daily_report(now_utc=now)  # type: ignore[attr-defined]
            self.assertGreater(runtime._recoverable_error_count, 0)  # type: ignore[attr-defined]
            self.assertNotIn("2026-04-16", runtime._state.daily_reports_sent)  # type: ignore[attr-defined]
            db.close()

    async def test_daily_report_format_includes_risk_and_qty_fields(self) -> None:
        report = {
            "date": "2026-04-16",
            "realized_pnl": 1.0,
            "unrealized_pnl": 2.0,
            "equity_change_proxy": 3.0,
            "new_trades": 1,
            "closed_trades": 1,
            "open_positions": [
                {
                    "instrument": "ES",
                    "strategy": "trend_pullback_vwap_ema",
                    "side": "LONG",
                    "entry_price": 100.0,
                    "mark_price": 101.0,
                    "stop_loss": 99.0,
                    "take_profit": 104.0,
                    "qty": 2.0,
                    "planned_risk_money": 500.0,
                    "planned_risk_pct": 0.5,
                    "expected_rr": 2.0,
                    "pnl": 2.0,
                    "holding_hours": 10.0,
                }
            ],
            "closed_trade_rows": [
                {
                    "instrument": "ES",
                    "strategy": "trend_pullback_vwap_ema",
                    "side": "LONG",
                    "entry": 100.0,
                    "exit": 102.0,
                    "qty": 2.0,
                    "planned_risk_money": 500.0,
                    "planned_risk_pct": 0.5,
                    "gross_pnl": 4.0,
                    "fees": 0.2,
                    "net_pnl": 3.8,
                    "r_multiple": 1.9,
                    "reason": "tp2_hit",
                }
            ],
            "signals_by_strategy": {"trend_pullback_vwap_ema": 1},
            "strategy_summary": {"trend_pullback_vwap_ema": {"pnl": 3.8}},
            "instrument_summary": {"ES": {"pnl": 3.8}},
            "risk_snapshot": {
                "total_risk_money": 500.0,
                "total_risk_pct": 0.5,
                "risk_by_instrument": {"ES": 0.5},
                "risk_by_strategy": {"trend_pullback_vwap_ema": 0.5},
                "risk_by_group": {"index": 0.5},
                "risk_reject_reasons": {"sizing_reject": 1},
                "open_positions": 1,
            },
            "operational": {
                "recoverable_errors": 0,
                "disconnects": 0,
                "recoveries": 0,
                "last_processed": "",
            },
        }
        messages = _format_daily_report_messages(report=report)
        merged = "\n".join(messages)
        self.assertIn("qty=", merged)
        self.assertIn("Risk total money:", merged)
        self.assertIn("Risk by group:", merged)
        self.assertIn("Risk rejects:", merged)


if __name__ == "__main__":
    unittest.main()
