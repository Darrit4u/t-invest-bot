from __future__ import annotations

import unittest

from core.strategy_params import iter_strategy_param_variants, resolve_strategy_params


class StrategyParamsTests(unittest.TestCase):
    def test_resolve_strategy_params_supports_legacy_flat_structure(self) -> None:
        section = {
            "trend_pullback_vwap_ema": {
                "impulse_bars": 3,
                "tp1_r": 1.0,
            }
        }

        resolved = resolve_strategy_params(
            section=section,
            strategy_name="trend_pullback_vwap_ema",
            instrument_symbol="ES",
        )

        self.assertEqual(resolved["impulse_bars"], 3)
        self.assertEqual(resolved["tp1_r"], 1.0)

    def test_resolve_strategy_params_merges_defaults_and_instrument_override(self) -> None:
        section = {
            "defaults": {
                "trend_pullback_vwap_ema": {
                    "impulse_bars": 3,
                    "tp1_r": 1.0,
                    "tp2_r": 2.2,
                }
            },
            "by_instrument": {
                "ES": {
                    "trend_pullback_vwap_ema": {
                        "tp1_r": 1.5,
                    }
                }
            },
        }

        resolved_es = resolve_strategy_params(
            section=section,
            strategy_name="trend_pullback_vwap_ema",
            instrument_symbol="es",
        )
        resolved_ng = resolve_strategy_params(
            section=section,
            strategy_name="trend_pullback_vwap_ema",
            instrument_symbol="NG",
        )

        self.assertEqual(resolved_es["impulse_bars"], 3)
        self.assertEqual(resolved_es["tp1_r"], 1.5)
        self.assertEqual(resolved_es["tp2_r"], 2.2)
        self.assertEqual(resolved_ng["tp1_r"], 1.0)

    def test_iter_strategy_param_variants_returns_unique_effective_variants(self) -> None:
        section = {
            "defaults": {
                "compression_breakout": {
                    "compression_window_bars": 12,
                }
            },
            "instruments": {
                "ES": {
                    "compression_breakout": {
                        "compression_window_bars": 16,
                    }
                },
                "NG": {
                    "compression_breakout": {
                        "compression_window_bars": 16,
                    }
                },
            },
        }

        variants = iter_strategy_param_variants(
            section=section,
            strategy_name="compression_breakout",
        )

        self.assertEqual(len(variants), 2)
        windows = sorted(int(item.get("compression_window_bars", 0)) for item in variants)
        self.assertEqual(windows, [12, 16])

    def test_mode_specific_defaults_are_applied(self) -> None:
        section = {
            "defaults": {
                "trend_pullback_vwap_ema": {
                    "impulse_bars": 3,
                    "tp1_r": 1.0,
                }
            },
            "by_mode": {
                "intraday": {
                    "defaults": {
                        "trend_pullback_vwap_ema": {
                            "tp1_r": 0.9,
                            "trend_timeframe": "1hour",
                        }
                    }
                },
                "swing": {
                    "defaults": {
                        "trend_pullback_vwap_ema": {
                            "tp1_r": 1.3,
                            "trend_timeframe": "4hour",
                            "setup_timeframe": "1hour",
                        }
                    }
                },
            },
        }

        intraday = resolve_strategy_params(
            section=section,
            strategy_name="trend_pullback_vwap_ema",
            trading_mode="intraday",
        )
        swing = resolve_strategy_params(
            section=section,
            strategy_name="trend_pullback_vwap_ema",
            trading_mode="swing",
        )

        self.assertEqual(intraday["tp1_r"], 0.9)
        self.assertEqual(intraday["trend_timeframe"], "1hour")
        self.assertEqual(swing["tp1_r"], 1.3)
        self.assertEqual(swing["trend_timeframe"], "4hour")
        self.assertEqual(swing["setup_timeframe"], "1hour")


if __name__ == "__main__":
    unittest.main()
