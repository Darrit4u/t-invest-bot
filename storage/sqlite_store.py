"""SQLite persistence for signals, trades, lifecycle events and stats snapshots."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.models import StrategySignal


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class SQLiteStore:
    """Lightweight SQLite wrapper for runtime persistence.

    Notes:
    - `trades`/`trade_events` store simulator-level lifecycle state for auditability.
    - Domain-level analytics should consume `Position`/`Trade`/`DomainEvent` via adapters.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                instrument TEXT NOT NULL,
                strategy TEXT NOT NULL,
                regime TEXT NOT NULL,
                direction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                entry_mode TEXT NOT NULL,
                entry REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,
                instrument TEXT NOT NULL,
                strategy TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                activated_at TEXT,
                closed_at TEXT,
                entry REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL NOT NULL,
                tp1_size REAL NOT NULL,
                quantity REAL NOT NULL,
                remaining_qty REAL NOT NULL,
                entry_fill_price REAL,
                current_stop REAL NOT NULL,
                tp1_hit_at TEXT,
                tp2_hit_at TEXT,
                bars_waiting INTEGER NOT NULL,
                bars_in_trade INTEGER NOT NULL,
                max_wait_bars INTEGER NOT NULL,
                max_trade_bars INTEGER NOT NULL,
                gross_pnl REAL NOT NULL,
                fees_paid REAL NOT NULL,
                net_pnl REAL NOT NULL,
                r_multiple REAL NOT NULL,
                exit_reason TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                instrument TEXT NOT NULL,
                strategy TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                event_time TEXT NOT NULL,
                price REAL,
                size REAL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stats_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_state (
                state_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_notifications (
                notification_key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                sent_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_instrument_time
                ON signals(instrument, timestamp);

            CREATE INDEX IF NOT EXISTS idx_trades_status_updated
                ON trades(status, updated_at);

            CREATE INDEX IF NOT EXISTS idx_trade_events_trade_time
                ON trade_events(trade_id, event_time);

            CREATE INDEX IF NOT EXISTS idx_runtime_notifications_sent_at
                ON runtime_notifications(sent_at);
            """
        )
        self._conn.commit()

    def save_signal(self, signal: StrategySignal) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO signals (
                signal_id, instrument, strategy, regime, direction, timestamp,
                entry_mode, entry, stop_loss, tp1, tp2, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.signal_id,
                signal.instrument,
                signal.strategy,
                signal.regime.value,
                signal.direction.value,
                signal.timestamp.isoformat(),
                signal.entry_mode,
                signal.entry,
                signal.stop_loss,
                signal.tp1,
                signal.tp2,
                json.dumps(signal.metadata, default=_json_default),
                now,
            ),
        )
        self._conn.commit()

    def save_trade_state_snapshot(self, trade_state: Any) -> None:
        """Persist simulator trade state snapshot into `trades`."""
        row = self._trade_row(trade_state)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trades (
                trade_id, signal_id, instrument, strategy, timeframe, direction, status,
                created_at, updated_at, activated_at, closed_at,
                entry, stop_loss, tp1, tp2, tp1_size,
                quantity, remaining_qty, entry_fill_price, current_stop,
                tp1_hit_at, tp2_hit_at,
                bars_waiting, bars_in_trade, max_wait_bars, max_trade_bars,
                gross_pnl, fees_paid, net_pnl, r_multiple, exit_reason, metadata_json
            ) VALUES (
                :trade_id, :signal_id, :instrument, :strategy, :timeframe, :direction, :status,
                :created_at, :updated_at, :activated_at, :closed_at,
                :entry, :stop_loss, :tp1, :tp2, :tp1_size,
                :quantity, :remaining_qty, :entry_fill_price, :current_stop,
                :tp1_hit_at, :tp2_hit_at,
                :bars_waiting, :bars_in_trade, :max_wait_bars, :max_trade_bars,
                :gross_pnl, :fees_paid, :net_pnl, :r_multiple, :exit_reason, :metadata_json
            )
            """,
            row,
        )
        self._conn.commit()

    def save_trade(self, trade: Any) -> None:
        """Backward-compatible alias for simulator trade-state persistence."""
        self.save_trade_state_snapshot(trade)

    def save_trade_lifecycle_event(self, event: Any) -> None:
        """Persist low-level simulator lifecycle event into `trade_events`."""
        payload = getattr(event, "payload", {})
        self._conn.execute(
            """
            INSERT INTO trade_events (
                trade_id, signal_id, instrument, strategy,
                event_type, status, event_time, price, size, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.trade_id,
                event.signal_id,
                event.instrument,
                event.strategy,
                event.event_type,
                event.status,
                event.event_time.isoformat(),
                event.price,
                event.size,
                json.dumps(payload, default=_json_default),
            ),
        )
        self._conn.commit()

    def save_trade_event(self, event: Any) -> None:
        """Backward-compatible alias for low-level lifecycle event persistence."""
        self.save_trade_lifecycle_event(event)

    def save_stats_snapshot(self, snapshot_time: datetime, payload: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO stats_snapshots (snapshot_time, payload_json)
            VALUES (?, ?)
            """,
            (snapshot_time.isoformat(), json.dumps(payload, default=_json_default)),
        )
        self._conn.commit()

    def signal_exists(self, signal_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM signals WHERE signal_id = ? LIMIT 1",
            (signal_id,),
        ).fetchone()
        return row is not None

    def trade_origin_exists(self, *, instrument: str, strategy: str, created_at_iso: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM trades
            WHERE instrument = ? AND strategy = ? AND created_at = ?
            LIMIT 1
            """,
            (instrument, strategy, created_at_iso),
        ).fetchone()
        return row is not None

    def load_open_trade_states(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM trades
            WHERE closed_at IS NULL
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [self._trade_row_to_dict(row) for row in rows]

    def load_all_trade_rows(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM trades
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [self._trade_row_to_dict(row) for row in rows]

    def load_trade_rows_closed_between(self, *, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM trades
            WHERE closed_at IS NOT NULL
              AND closed_at >= ?
              AND closed_at < ?
            ORDER BY closed_at ASC
            """,
            (start_iso, end_iso),
        ).fetchall()
        return [self._trade_row_to_dict(row) for row in rows]

    def load_signal_rows_between(self, *, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM signals
            WHERE timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp ASC
            """,
            (start_iso, end_iso),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_runtime_state(self, *, state_key: str, payload: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO runtime_state (state_key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (state_key, json.dumps(payload, default=_json_default), now),
        )
        self._conn.commit()

    def load_runtime_state(self, *, state_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT value_json
            FROM runtime_state
            WHERE state_key = ?
            LIMIT 1
            """,
            (state_key,),
        ).fetchone()
        if row is None:
            return None
        raw = row["value_json"]
        if not isinstance(raw, str):
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            return value
        return None

    def mark_runtime_notification_sent(
        self,
        *,
        notification_key: str,
        category: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        if self.runtime_notification_sent(notification_key=notification_key):
            return False

        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO runtime_notifications (notification_key, category, payload_json, sent_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                notification_key,
                category,
                json.dumps(payload or {}, default=_json_default),
                now,
            ),
        )
        self._conn.commit()
        return True

    def runtime_notification_sent(self, *, notification_key: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM runtime_notifications
            WHERE notification_key = ?
            LIMIT 1
            """,
            (notification_key,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _trade_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        raw_meta = payload.get("metadata_json")
        if isinstance(raw_meta, str):
            try:
                value = json.loads(raw_meta)
            except json.JSONDecodeError:
                value = {}
            payload["metadata_json"] = value if isinstance(value, dict) else {}
        else:
            payload["metadata_json"] = {}
        return payload

    @staticmethod
    def _trade_row(trade: Any) -> dict[str, Any]:
        return {
            "trade_id": trade.trade_id,
            "signal_id": trade.signal_id,
            "instrument": trade.instrument,
            "strategy": trade.strategy,
            "timeframe": trade.timeframe,
            "direction": trade.direction.value,
            "status": trade.status.value,
            "created_at": trade.created_at.isoformat(),
            "updated_at": trade.updated_at.isoformat(),
            "activated_at": trade.activated_at.isoformat() if trade.activated_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            "entry": trade.entry,
            "stop_loss": trade.stop_loss,
            "tp1": trade.tp1,
            "tp2": trade.tp2,
            "tp1_size": trade.tp1_size,
            "quantity": trade.quantity,
            "remaining_qty": trade.remaining_qty,
            "entry_fill_price": trade.entry_fill_price,
            "current_stop": trade.current_stop,
            "tp1_hit_at": trade.tp1_hit_at.isoformat() if trade.tp1_hit_at else None,
            "tp2_hit_at": trade.tp2_hit_at.isoformat() if trade.tp2_hit_at else None,
            "bars_waiting": trade.bars_waiting,
            "bars_in_trade": trade.bars_in_trade,
            "max_wait_bars": trade.max_wait_bars,
            "max_trade_bars": trade.max_trade_bars,
            "gross_pnl": trade.gross_pnl,
            "fees_paid": trade.fees_paid,
            "net_pnl": trade.net_pnl,
            "r_multiple": trade.r_multiple,
            "exit_reason": trade.exit_reason,
            "metadata_json": json.dumps(trade.metadata, default=_json_default),
        }
