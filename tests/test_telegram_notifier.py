from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from core.models import MarketRegime, SignalDirection
from core.telegram_notifier import (
    TelegramConfig,
    TelegramNotifier,
    _format_signal_message,
    _format_summary_message,
)
from tests.helpers import build_signal


class _DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args))

    def exception(self, *args, **kwargs):
        self.messages.append(("exception", args))


class TelegramNotifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_failures_do_not_raise_and_notifier_closes_cleanly(self) -> None:
        config = TelegramConfig(
            enabled=True,
            bot_token="fake",
            chat_id="123",
            retry_attempts=2,
            retry_delay_seconds=0.01,
            request_timeout_seconds=1.0,
            queue_maxsize=50,
            summary_interval_seconds=0,
            send_startup_message=True,
            send_shutdown_summary=True,
        )
        logger = _DummyLogger()
        notifier = TelegramNotifier(config=config, logger=logger)

        with patch("core.telegram_notifier._post_json", return_value=(0, "", "network blocked")):
            await notifier.start()
            self.assertTrue(notifier.enabled)
            await notifier.notify_text("hello", category="test")
            await notifier.close()

        self.assertTrue(any(level == "error" for level, _ in logger.messages))

    def test_signal_format_contains_required_fields(self) -> None:
        signal = build_signal(
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            regime=MarketRegime.TREND,
            direction=SignalDirection.LONG,
            timestamp=datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc),
            entry=100,
            stop_loss=99,
            tp1=101,
            tp2=102,
        )
        msg = _format_signal_message(signal)
        self.assertIn("NEW SIGNAL", msg)
        self.assertIn("Instrument: ES", msg)
        self.assertIn("Strategy: trend_pullback_vwap_ema", msg)
        self.assertIn("TP2", msg)

    def test_summary_format_includes_global_and_instrument_stats(self) -> None:
        summary = {
            "global": {
                "signals": 5,
                "activated": 4,
                "closed": 3,
                "net_pnl": 10.2,
                "win_rate": 0.5,
                "profit_factor": 1.8,
                "max_drawdown": 2.1,
            },
            "by_instrument": {
                "ES": {"signals": 3, "closed": 2, "net_pnl": 7.0},
                "NG": {"signals": 2, "closed": 1, "net_pnl": 3.2},
            },
        }
        msg = _format_summary_message(summary, open_trades=1)
        self.assertIn("DAILY SUMMARY", msg)
        self.assertIn("Signals: 5", msg)
        self.assertIn("Open trades: 1", msg)
        self.assertIn("ES: sig=3", msg)


if __name__ == "__main__":
    unittest.main()
