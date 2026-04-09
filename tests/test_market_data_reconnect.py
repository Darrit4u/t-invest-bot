from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from core.config_loader import ConfigLoader
from core.instrument_registry import InstrumentRegistry
from core.market_data import Candle, create_market_data_client
from tests.helpers import config_dir


class MarketDataReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_status_emitted_when_stream_cancelled(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)

        statuses: list[tuple[str, dict]] = []
        stop_event = asyncio.Event()

        async def on_status(status: str, payload: dict) -> None:
            statuses.append((status, payload))
            if status == "disconnect":
                stop_event.set()

        async def on_candle(_: Candle) -> None:
            return None

        class _SilentLogger:
            def info(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

            def exception(self, *args, **kwargs):
                return None

            def debug(self, *args, **kwargs):
                return None

        client = create_market_data_client(
            mode="t_invest",
            token="test-token",
            registry=registry,
            params={"reconnect_delay_seconds": 0.1},
            timeframe="1min",
            logger=_SilentLogger(),
            status_handler=on_status,
        )

        with patch.object(client, "_run_once", side_effect=asyncio.CancelledError()):
            await asyncio.wait_for(client.run(on_candle=on_candle, stop_event=stop_event), timeout=2)

        status_names = [name for name, _ in statuses]
        self.assertIn("connecting", status_names)
        self.assertIn("disconnect", status_names)

    async def test_disconnect_status_emitted_when_sdk_missing(self) -> None:
        cfg = ConfigLoader(config_dir()).load()
        registry = InstrumentRegistry.from_config(cfg)

        statuses: list[tuple[str, dict]] = []
        stop_event = asyncio.Event()

        async def on_status(status: str, payload: dict) -> None:
            statuses.append((status, payload))
            if status == "disconnect":
                stop_event.set()

        async def on_candle(_: Candle) -> None:
            return None

        class _SilentLogger:
            def info(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

            def exception(self, *args, **kwargs):
                return None

            def debug(self, *args, **kwargs):
                return None

        client = create_market_data_client(
            mode="t_invest",
            token="test-token",
            registry=registry,
            params={"reconnect_delay_seconds": 0.1},
            timeframe="1min",
            logger=_SilentLogger(),
            status_handler=on_status,
        )

        with patch("core.market_data._load_tinvest_sdk", return_value=None):
            await asyncio.wait_for(client.run(on_candle=on_candle, stop_event=stop_event), timeout=2)

        status_names = [name for name, _ in statuses]
        self.assertIn("connecting", status_names)
        self.assertIn("disconnect", status_names)


if __name__ == "__main__":
    unittest.main()
