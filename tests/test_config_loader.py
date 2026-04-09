from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.config_loader import ConfigError, ConfigLoader
from tests.helpers import config_dir


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_project_config(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        self.assertGreaterEqual(len(cfg.instruments), 1)
        self.assertGreaterEqual(sum(1 for i in cfg.instruments.values() if i.enabled), 1)
        self.assertIn("ES", cfg.strategies_by_instrument)

    def test_blackout_file_path_is_read_from_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "instruments.yaml").write_text(
                """
                history_depth: 100
                default_timeframe: "1min"
                session_rules:
                  X:
                    timezone: "UTC"
                    start: "00:00"
                    end: "23:59"
                instruments:
                  ES:
                    enabled: true
                    tick_size: 0.25
                    tick_value: 12.5
                    lot: 1
                    sessions: [X]
                """,
                encoding="utf-8",
            )
            (root / "strategies.yaml").write_text(
                "strategies:\n  ES: [trend_pullback_vwap_ema]\n",
                encoding="utf-8",
            )
            (root / "params.yaml").write_text(
                "news_blackout_file: custom_blackout.yaml\ntimezone: UTC\n",
                encoding="utf-8",
            )
            (root / "custom_blackout.yaml").write_text(
                "- start: '2026-04-10 15:25'\n  end: '2026-04-10 15:40'\n  description: CPI\n",
                encoding="utf-8",
            )

            cfg = ConfigLoader(root).load()
            self.assertEqual(len(cfg.blackout_windows), 1)
            self.assertEqual(cfg.blackout_windows[0].description, "CPI")

    def test_invalid_history_depth_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "instruments.yaml").write_text(
                """
                history_depth: 2
                default_timeframe: "1min"
                session_rules: {}
                instruments: {}
                """,
                encoding="utf-8",
            )
            (root / "strategies.yaml").write_text("strategies: {}\n", encoding="utf-8")
            (root / "params.yaml").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ConfigError):
                ConfigLoader(root).load()

    def test_unknown_session_reference_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "instruments.yaml").write_text(
                """
                history_depth: 100
                default_timeframe: "1min"
                session_rules:
                  A:
                    timezone: "UTC"
                    start: "00:00"
                    end: "23:59"
                instruments:
                  ES:
                    enabled: true
                    tick_size: 0.25
                    tick_value: 12.5
                    lot: 1
                    sessions: [B]
                """,
                encoding="utf-8",
            )
            (root / "strategies.yaml").write_text("strategies:\n  ES: [trend_pullback_vwap_ema]\n", encoding="utf-8")
            (root / "params.yaml").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ConfigError):
                ConfigLoader(root).load()


if __name__ == "__main__":
    unittest.main()
