from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from core.history_preloader import estimate_required_bars, preload_history
from core.instrument_registry import InstrumentRegistry
from storage.memory_store import MemoryCandleStore
from tests.helpers import build_instrument_meta


class _FakeCandle:
    def __init__(self, dt: datetime, open_: float, high: float, low: float, close: float, volume: float):
        self.time = dt
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class _FakeResponse:
    def __init__(self, candles):
        self.candles = candles


class _FakeMarketData:
    def __init__(self, candles):
        self._candles = candles

    async def get_candles(self, **kwargs):  # noqa: ARG002
        return _FakeResponse(self._candles)


class _FakeAsyncClient:
    def __init__(self, token: str, candles):
        self.token = token
        self.market_data = _FakeMarketData(candles=candles)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
        return False


class _SilentLogger:
    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class _SequenceMarketData:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def get_candles(self, **kwargs):  # noqa: ARG002
        idx = min(self.calls, max(0, len(self._responses) - 1))
        self.calls += 1
        return _FakeResponse(self._responses[idx])


class _SequenceAsyncClient:
    def __init__(self, token: str, market_data):
        self.token = token
        self.market_data = market_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
        return False


class HistoryPreloaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_preload_history_inserts_candles_into_store(self) -> None:
        meta = replace(build_instrument_meta(symbol="ES"), uid="test-uid")
        registry = InstrumentRegistry(items={"ES": meta})
        store = MemoryCandleStore(history_depth=200)
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        candles = [
            _FakeCandle(now - timedelta(minutes=10), 100.0, 100.3, 99.8, 100.1, 1200.0),
            _FakeCandle(now - timedelta(minutes=5), 100.1, 100.5, 100.0, 100.4, 1300.0),
            _FakeCandle(now, 100.4, 100.7, 100.2, 100.6, 1250.0),
        ]

        class _Interval:
            CANDLE_INTERVAL_5_MIN = object()

        sdk = {
            "AsyncClient": lambda token: _FakeAsyncClient(token=token, candles=candles),
            "CandleInterval": _Interval,
        }
        params = {
            "history_preload": {
                "enabled": True,
                "bars": 20,
                "extra_bars": 0,
            }
        }

        with patch("core.history_preloader._load_tinvest_sdk", return_value=sdk):
            report = await preload_history(
                token="test-token",
                registry=registry,
                store=store,
                params=params,
                timeframe="5min",
                logger=_SilentLogger(),
            )

        self.assertTrue(report.enabled)
        self.assertGreaterEqual(report.inserted, 3)
        self.assertEqual(report.instruments_with_data, 1)
        self.assertIsNotNone(store.latest("ES", "5min"))

    async def test_preload_history_expands_lookback_until_target_or_max_available(self) -> None:
        meta = replace(build_instrument_meta(symbol="ES"), uid="test-uid")
        registry = InstrumentRegistry(items={"ES": meta})
        store = MemoryCandleStore(history_depth=500)
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

        short_rows = [
            _FakeCandle(
                now - timedelta(minutes=(50 - i) * 5),
                100.0 + i * 0.1,
                100.2 + i * 0.1,
                99.8 + i * 0.1,
                100.1 + i * 0.1,
                1000.0 + i,
            )
            for i in range(10)
        ]
        long_rows = [
            _FakeCandle(
                now - timedelta(minutes=(70 - i) * 5),
                99.0 + i * 0.1,
                99.3 + i * 0.1,
                98.7 + i * 0.1,
                99.1 + i * 0.1,
                900.0 + i,
            )
            for i in range(70)
        ]
        sequence_market_data = _SequenceMarketData([short_rows, long_rows])

        class _Interval:
            CANDLE_INTERVAL_5_MIN = object()

        sdk = {
            "AsyncClient": lambda token: _SequenceAsyncClient(token=token, market_data=sequence_market_data),
            "CandleInterval": _Interval,
        }
        params = {
            "history_preload": {
                "enabled": True,
                "bars": 50,
                "extra_bars": 0,
                "max_fetch_rounds": 3,
                "lookback_start_multiplier": 1.0,
                "lookback_growth_factor": 2.0,
                "max_lookback_multiplier": 4.0,
            }
        }

        with patch("core.history_preloader._load_tinvest_sdk", return_value=sdk):
            report = await preload_history(
                token="test-token",
                registry=registry,
                store=store,
                params=params,
                timeframe="5min",
                logger=_SilentLogger(),
            )

        self.assertTrue(report.enabled)
        self.assertGreaterEqual(report.processed_candles, 50)
        self.assertGreaterEqual(sequence_market_data.calls, 2)
        self.assertIsNotNone(store.latest("ES", "5min"))

    def test_estimate_required_bars_accounts_for_mtf(self) -> None:
        params = {
            "indicator_engine": {
                "atr_period": 14,
                "volume_period": 20,
                "slope_period": 5,
                "overlap_window": 12,
                "swing_window": 5,
                "crossing_lookback": 30,
            },
            "strategy_params": {
                "trend_pullback_vwap_ema": {
                    "impulse_bars": 3,
                    "use_mtf_filter": True,
                    "trend_timeframe": "1hour",
                    "setup_timeframe": "15min",
                    "mtf_slow_ema": 6,
                    "mtf_slope_bars": 2,
                }
            },
        }
        bars = estimate_required_bars(params=params, timeframe="5min")
        # 5min -> 1hour ratio is 12; with slow=6,slope=2 => (6+2+1)*12 = 108 bars.
        self.assertGreaterEqual(bars, 108)

    def test_estimate_required_bars_uses_instrument_overrides(self) -> None:
        params = {
            "strategy_params": {
                "defaults": {
                    "trend_pullback_vwap_ema": {
                        "impulse_bars": 3,
                        "use_mtf_filter": False,
                    }
                },
                "by_instrument": {
                    "NG": {
                        "trend_pullback_vwap_ema": {
                            "use_mtf_filter": True,
                            "trend_timeframe": "1hour",
                            "setup_timeframe": "15min",
                            "mtf_slow_ema": 7,
                            "mtf_slope_bars": 3,
                        }
                    }
                },
            }
        }

        bars = estimate_required_bars(params=params, timeframe="5min")
        # 5min -> 1hour ratio is 12; with slow=7,slope=3 => (7+3+1)*12 = 132 bars.
        self.assertGreaterEqual(bars, 132)

    def test_estimate_required_bars_supports_4hour_trend_timeframe(self) -> None:
        params = {
            "strategy_params": {
                "trend_pullback_vwap_ema": {
                    "use_mtf_filter": True,
                    "trend_timeframe": "4hour",
                    "setup_timeframe": "1hour",
                    "mtf_slow_ema": 6,
                    "mtf_slope_bars": 2,
                }
            },
        }

        bars = estimate_required_bars(params=params, timeframe="1hour")
        # 1hour -> 4hour ratio is 4; with slow=6,slope=2 => (6+2+1)*4 = 36 bars.
        self.assertGreaterEqual(bars, 36)


if __name__ == "__main__":
    unittest.main()
