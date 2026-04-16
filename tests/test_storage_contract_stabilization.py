from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.trade_simulator import TradeSimulator
from storage.sqlite_store import SQLiteStore
from tests.helpers import build_signal


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        return None


class StorageContractStabilizationTests(unittest.TestCase):
    def test_explicit_trade_state_and_event_methods_persist_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "storage_contract.db")
            sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)

            signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0)
            events = sim.register_signal(signal, timeframe="1min")
            event = events[0]
            trade_state = sim.get_trade(event.trade_id)
            self.assertIsNotNone(trade_state)
            assert trade_state is not None

            db.save_trade_state_snapshot(trade_state)
            db.save_trade_lifecycle_event(event)

            trades = db._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            lifecycle_events = db._conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()[0]
            self.assertEqual(trades, 1)
            self.assertEqual(lifecycle_events, 1)
            db.close()

    def test_legacy_aliases_remain_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "storage_contract_legacy.db")
            sim = TradeSimulator(params={"trade_simulator": {}}, logger=_DummyLogger(), storage=None)

            signal = build_signal(entry=100.0, stop_loss=99.0, tp1=101.0, tp2=102.0)
            events = sim.register_signal(signal, timeframe="1min")
            event = events[0]
            trade_state = sim.get_trade(event.trade_id)
            self.assertIsNotNone(trade_state)
            assert trade_state is not None

            db.save_trade(trade_state)
            db.save_trade_event(event)

            trades = db._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            lifecycle_events = db._conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()[0]
            self.assertEqual(trades, 1)
            self.assertEqual(lifecycle_events, 1)
            db.close()

    def test_runtime_notification_dedup_and_state_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = SQLiteStore(Path(td) / "runtime_state.db")
            inserted = db.mark_runtime_notification_sent(
                notification_key="signal:ES:trend:2026-01-05T10:00:00+00:00",
                category="signal",
                payload={"signal_id": "s1"},
            )
            self.assertTrue(inserted)
            self.assertTrue(
                db.runtime_notification_sent(
                    notification_key="signal:ES:trend:2026-01-05T10:00:00+00:00"
                )
            )
            inserted_again = db.mark_runtime_notification_sent(
                notification_key="signal:ES:trend:2026-01-05T10:00:00+00:00",
                category="signal",
                payload={"signal_id": "s1"},
            )
            self.assertFalse(inserted_again)

            payload = {
                "last_processed_by_stream": {"ES:1min": "2026-01-05T10:00:00+00:00"},
                "daily_reports_sent": ["2026-01-05"],
            }
            db.save_runtime_state(state_key="server_paper_runtime_state_v1", payload=payload)
            loaded = db.load_runtime_state(state_key="server_paper_runtime_state_v1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["last_processed_by_stream"]["ES:1min"], "2026-01-05T10:00:00+00:00")
            db.close()


if __name__ == "__main__":
    unittest.main()
