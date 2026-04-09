from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from storage.sqlite_store import SQLiteStore


class SQLiteSchemaRegressionTests(unittest.TestCase):
    def test_expected_tables_and_columns_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "schema.db")
            conn = db._conn

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("signals", tables)
            self.assertIn("trades", tables)
            self.assertIn("trade_events", tables)
            self.assertIn("stats_snapshots", tables)

            trade_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(trades)").fetchall()
            }
            for required in (
                "trade_id",
                "signal_id",
                "status",
                "entry",
                "stop_loss",
                "tp1",
                "tp2",
                "gross_pnl",
                "fees_paid",
                "net_pnl",
                "r_multiple",
            ):
                self.assertIn(required, trade_cols)

            db.close()

    def test_stats_snapshot_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "stats.db")
            payload = {"global": {"signals": 10, "net_pnl": 1.23}}
            db.save_stats_snapshot(datetime.now(tz=timezone.utc), payload)

            row = db._conn.execute("SELECT payload_json FROM stats_snapshots LIMIT 1").fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertIn("signals", row[0])
            db.close()


if __name__ == "__main__":
    unittest.main()
