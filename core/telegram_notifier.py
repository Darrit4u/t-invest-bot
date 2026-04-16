"""Telegram notification delivery with retries and message templates."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

from core.models import StrategySignal, Trade
from core.portfolio_events import PortfolioEvent


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Telegram runtime configuration."""

    enabled: bool
    bot_token: str
    chat_id: str
    retry_attempts: int
    retry_delay_seconds: float
    request_timeout_seconds: float
    queue_maxsize: int
    summary_interval_seconds: int
    send_startup_message: bool
    send_shutdown_summary: bool
    send_signals: bool = True
    send_positions: bool = True
    send_daily_report: bool = True
    send_heartbeat: bool = True
    send_errors: bool = True

    @classmethod
    def from_sources(cls, *, env: dict[str, str], params: dict[str, Any]) -> "TelegramConfig":
        section = params.get("telegram", {}) if isinstance(params.get("telegram", {}), dict) else {}

        enabled = bool(section.get("enabled", True))
        bot_token = str(env.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(env.get("TELEGRAM_CHAT_ID", "")).strip()

        return cls(
            enabled=enabled,
            bot_token=bot_token,
            chat_id=chat_id,
            retry_attempts=max(1, int(section.get("retry_attempts", 3))),
            retry_delay_seconds=max(0.1, float(section.get("retry_delay_seconds", 1.5))),
            request_timeout_seconds=max(1.0, float(section.get("request_timeout_seconds", 10.0))),
            queue_maxsize=max(10, int(section.get("queue_maxsize", 200))),
            summary_interval_seconds=max(0, int(section.get("summary_interval_seconds", 0))),
            send_startup_message=bool(section.get("send_startup_message", True)),
            send_shutdown_summary=bool(section.get("send_shutdown_summary", True)),
            send_signals=_as_bool(section.get("send_signals", True), default=True),
            send_positions=_as_bool(section.get("send_positions", True), default=True),
            send_daily_report=_as_bool(section.get("send_daily_report", True), default=True),
            send_heartbeat=_as_bool(section.get("send_heartbeat", True), default=True),
            send_errors=_as_bool(section.get("send_errors", True), default=True),
        )


class TelegramNotifier:
    """Asynchronous safe-send Telegram notifier with in-process queue."""

    def __init__(self, config: TelegramConfig, logger: Any):
        self._config = config
        self._logger = logger
        self._queue: asyncio.Queue[tuple[str, str] | None] | None = None
        self._worker_task: asyncio.Task[Any] | None = None
        self._ready = False

    @property
    def enabled(self) -> bool:
        return self._ready

    @property
    def summary_interval_seconds(self) -> int:
        return self._config.summary_interval_seconds

    @property
    def send_startup_message(self) -> bool:
        return self._config.send_startup_message

    @property
    def send_shutdown_summary(self) -> bool:
        return self._config.send_shutdown_summary

    async def start(self) -> None:
        if not self._config.enabled:
            self._logger.info("Telegram disabled by config")
            return

        if not self._config.bot_token or not self._config.chat_id:
            self._logger.warning("Telegram disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty")
            return

        self._queue = asyncio.Queue(maxsize=self._config.queue_maxsize)
        self._worker_task = asyncio.create_task(self._worker(), name="telegram-notifier")
        self._ready = True
        self._logger.info("Telegram notifier started")

    async def close(self) -> None:
        if not self._ready:
            return

        assert self._queue is not None
        assert self._worker_task is not None

        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._worker_task, timeout=20)
        except asyncio.TimeoutError:
            self._logger.error("Telegram worker shutdown timeout; cancelling worker task")
            self._worker_task.cancel()
            with contextlib.suppress(Exception):
                await self._worker_task

        self._ready = False
        self._logger.info("Telegram notifier stopped")

    async def notify_text(self, text: str, category: str = "info") -> None:
        if not self._ready:
            return

        assert self._queue is not None
        try:
            self._queue.put_nowait((category, text))
        except asyncio.QueueFull:
            self._logger.error("Telegram queue overflow: dropping message category=%s", category)

    async def notify_signal(self, signal: StrategySignal) -> None:
        if not self._config.send_signals:
            return
        await self.notify_text(_format_signal_message(signal), category="new_signal")

    async def notify_trade_event(self, event: Any, trade: Trade | None = None) -> None:
        """Legacy adapter for low-level simulator events."""
        text = _format_trade_event_message(event, trade)
        await self.notify_text(text, category=event.event_type)

    async def notify_portfolio_event(self, event: PortfolioEvent) -> None:
        if not self._config.send_positions:
            return
        text = _format_portfolio_event_message(event)
        await self.notify_text(text, category=f"portfolio_{event.event_type}")

    async def notify_daily_summary(self, summary: dict[str, Any], open_trades: int) -> None:
        if not self._config.send_daily_report:
            return
        await self.notify_text(_format_summary_message(summary, open_trades), category="daily_summary")

    async def notify_critical(self, title: str, details: str) -> None:
        if not self._config.send_errors:
            return
        await self.notify_text(_format_critical_message(title, details), category="critical")

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            category, text = item
            try:
                await self._send_with_retry(text=text, category=category)
            except Exception:
                self._logger.exception("Telegram send failed unexpectedly")
            finally:
                self._queue.task_done()

    async def _send_with_retry(self, *, text: str, category: str) -> None:
        url = f"https://api.telegram.org/bot{self._config.bot_token}/sendMessage"
        payload = {
            "chat_id": self._config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        for attempt in range(1, self._config.retry_attempts + 1):
            status, body, error_text = await asyncio.to_thread(
                _post_json,
                url,
                payload,
                self._config.request_timeout_seconds,
            )

            if status == 200:
                self._logger.info("Telegram delivered category=%s attempt=%d", category, attempt)
                return

            if error_text:
                self._logger.warning(
                    "Telegram send exception category=%s attempt=%d error=%s",
                    category,
                    attempt,
                    error_text,
                )
            else:
                self._logger.warning(
                    "Telegram API status=%s category=%s attempt=%d body=%s",
                    status,
                    category,
                    attempt,
                    body[:300],
                )

            if attempt < self._config.retry_attempts:
                await asyncio.sleep(self._config.retry_delay_seconds * attempt)

        self._logger.error("Telegram message dropped after retries category=%s", category)


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> tuple[int, str, str | None]:
    request_payload = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=request_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", errors="replace")
            return int(status), body, None
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return int(getattr(exc, "code", 0)), body, None
    except Exception as exc:
        return 0, "", str(exc)


def _as_bool(value: Any, *, default: bool) -> bool:
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


def _format_signal_message(signal: StrategySignal) -> str:
    return (
        "NEW SIGNAL\n"
        f"Instrument: {signal.instrument}\n"
        f"Strategy: {signal.strategy}\n"
        f"Regime: {signal.regime.value}\n"
        f"Direction: {signal.direction.value}\n"
        f"Entry mode: {signal.entry_mode}\n"
        f"Entry: {_fmt_price(signal.entry)}\n"
        f"SL: {_fmt_price(signal.stop_loss)}\n"
        f"TP1: {_fmt_price(signal.tp1)}\n"
        f"TP2: {_fmt_price(signal.tp2)}\n"
        f"Time: {signal.timestamp.isoformat()}"
    )


def _format_trade_event_message(event: Any, trade: Trade | None) -> str:
    header = {
        "new_signal": "TRADE REGISTERED",
        "activated": "TRADE ACTIVATED",
        "tp1_hit": "TP1 HIT",
        "tp2_hit": "TP2 HIT",
        "sl_hit": "STOP LOSS HIT",
        "expired": "TRADE EXPIRED",
        "cancelled_by_news": "TRADE CANCELLED BY NEWS",
        "cancelled_by_session_end": "TRADE CANCELLED BY SESSION",
    }.get(event.event_type, "TRADE EVENT")

    lines = [
        header,
        f"Instrument: {event.instrument}",
        f"Strategy: {event.strategy}",
        f"Status: {event.status}",
        f"Time: {event.event_time.isoformat()}",
    ]

    if event.price is not None:
        lines.append(f"Price: {_fmt_price(event.price)}")

    if trade is not None:
        entry_fill = trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry_price
        lines.extend(
            [
                f"Entry fill: {_fmt_price(entry_fill)}",
                f"Gross PnL: {_fmt_price(trade.gross_pnl if trade.gross_pnl is not None else trade.pnl)}",
                f"Net PnL: {_fmt_price(trade.pnl)}",
                f"R: {(trade.r_multiple if trade.r_multiple is not None else 0.0):.4f}",
            ]
        )

    if event.payload:
        lines.append(f"Details: {event.payload}")

    return "\n".join(lines)


def _format_summary_message(summary: dict[str, Any], open_trades: int) -> str:
    global_stats = summary.get("global", {})
    portfolio_stats = summary.get("portfolio", {})
    risk_snapshot = summary.get("portfolio_risk_snapshot", {})
    lines = [
        "DAILY SUMMARY",
        f"Signals: {int(global_stats.get('signals', 0))}",
        f"Activated: {int(global_stats.get('activated', 0))}",
        f"Closed: {int(global_stats.get('closed', 0))}",
        f"Open trades: {open_trades}",
        f"Net PnL: {_fmt_price(float(global_stats.get('net_pnl', 0.0)))}",
        f"Win rate: {float(global_stats.get('win_rate', 0.0)):.2%}",
        f"Profit factor: {float(global_stats.get('profit_factor', 0.0)):.4f}",
        f"Drawdown: {_fmt_price(float(global_stats.get('max_drawdown', 0.0)))}",
    ]
    risk_reject_reasons = portfolio_stats.get("risk_reject_reasons", {})
    if isinstance(risk_reject_reasons, dict) and risk_reject_reasons:
        lines.append(f"Risk rejects: {risk_reject_reasons}")
    if isinstance(risk_snapshot, dict) and risk_snapshot:
        lines.append(f"Open risk pct: {float(risk_snapshot.get('total_risk_pct', 0.0)):.4f}")
        lines.append(f"Open risk money: {_fmt_price(float(risk_snapshot.get('total_risk_money', 0.0)))}")
        lines.append(f"Risk by instrument: {risk_snapshot.get('risk_by_instrument', {})}")
        lines.append(f"Risk by strategy: {risk_snapshot.get('risk_by_strategy', {})}")
        lines.append(f"Risk by group: {risk_snapshot.get('risk_by_group', {})}")

    by_instrument = summary.get("by_instrument", {})
    if by_instrument:
        lines.append("By instrument:")
        for instrument, stats in by_instrument.items():
            lines.append(
                f"{instrument}: sig={int(stats.get('signals', 0))} closed={int(stats.get('closed', 0))} net={_fmt_price(float(stats.get('net_pnl', 0.0)))}"
            )

    return "\n".join(lines)


def _format_portfolio_event_message(event: PortfolioEvent) -> str:
    if event.event_type == "position_opened":
        return _format_position_opened_message(event)
    if event.event_type in {"risk_rejected", "allocation_rejected", "signal_rejected"}:
        return _format_rejection_message(event)

    lines = [
        "PORTFOLIO EVENT",
        f"Type: {event.kind.value}",
        f"Time: {event.event_time.isoformat()}",
    ]
    if event.instrument:
        lines.append(f"Instrument: {event.instrument}")
    if event.strategy:
        lines.append(f"Strategy: {event.strategy}")
    if event.signal_id:
        lines.append(f"Signal: {event.signal_id}")
    if event.trade_id:
        lines.append(f"Trade: {event.trade_id}")
    if event.reason:
        lines.append(f"Reason: {event.reason}")
    if event.payload:
        lines.append(f"Details: {event.payload}")
    return "\n".join(lines)


def _format_position_opened_message(event: PortfolioEvent) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    qty = _fmt_optional_float(payload.get("qty"))
    entry = _fmt_optional_float(payload.get("entry_fill_price"), fallback=payload.get("entry_price"))
    stop = _fmt_optional_float(payload.get("stop_loss"))
    take = _fmt_optional_float(payload.get("take_profit"))
    planned_risk_money = _fmt_optional_float(payload.get("planned_risk_money"))
    planned_risk_pct = _fmt_optional_pct(payload.get("planned_risk_pct"))
    expected_rr = _fmt_optional_ratio(payload.get("expected_rr"))

    lines = [
        "POSITION OPENED",
        f"Time: {event.event_time.isoformat()}",
        f"Instrument: {event.instrument or '-'}",
        f"Strategy: {event.strategy or '-'}",
    ]
    side = str(payload.get("side", "")).strip()
    if side:
        lines.append(f"Side: {side}")
    if entry is not None:
        lines.append(f"Entry: {entry}")
    if stop is not None:
        lines.append(f"Stop: {stop}")
    if take is not None:
        lines.append(f"Take: {take}")
    if qty is not None:
        lines.append(f"Qty: {qty}")
    if planned_risk_money is not None:
        lines.append(f"Planned risk: {planned_risk_money}")
    if planned_risk_pct is not None:
        lines.append(f"Planned risk %: {planned_risk_pct}")
    if expected_rr is not None:
        lines.append(f"Expected RR: {expected_rr}")
    if event.trade_id:
        lines.append(f"Trade: {event.trade_id}")
    if event.signal_id:
        lines.append(f"Signal: {event.signal_id}")
    return "\n".join(lines)


def _format_rejection_message(event: PortfolioEvent) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    lines = [
        "SIGNAL REJECTED",
        f"Type: {event.event_type}",
        f"Time: {event.event_time.isoformat()}",
        f"Instrument: {event.instrument or '-'}",
        f"Strategy: {event.strategy or '-'}",
        f"Reason: {event.reason or 'unknown'}",
    ]
    sizing_reason = payload.get("sizing_reject_reason")
    if sizing_reason:
        lines.append(f"Sizing reject: {sizing_reason}")
    category = payload.get("rejection_category")
    if category:
        lines.append(f"Category: {category}")
    if payload:
        lines.append(f"Details: {payload}")
    return "\n".join(lines)


def _format_critical_message(title: str, details: str) -> str:
    now = datetime.now(tz=timezone.utc).isoformat()
    return f"CRITICAL ALERT\nTitle: {title}\nTime: {now}\nDetails: {details}"


def _fmt_price(value: float) -> str:
    return f"{value:.5f}"


def _fmt_optional_float(value: Any, *, fallback: Any = None) -> str | None:
    raw = value if value is not None else fallback
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None
    return f"{numeric:.5f}"


def _fmt_optional_pct(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return f"{numeric:.4f}%"


def _fmt_optional_ratio(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return f"{numeric:.3f}"
