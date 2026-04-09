"""Core market data models and ingestion clients."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from core.instrument_registry import InstrumentMeta, InstrumentRegistry


CandleHandler = Callable[["Candle"], Awaitable[None]]
StatusHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


class CandleValidationError(ValueError):
    """Raised when incoming market data cannot be normalized."""


@dataclass(frozen=True, slots=True)
class Candle:
    """Normalized internal candle model."""

    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str
    timeframe: str

    @classmethod
    def validated(
        cls,
        *,
        dt: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        instrument: str,
        timeframe: str,
    ) -> "Candle":
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)

        if not instrument:
            raise CandleValidationError("instrument must not be empty")

        if not timeframe:
            raise CandleValidationError("timeframe must not be empty")

        values = {
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }

        if values["volume"] < 0:
            raise CandleValidationError("volume must be >= 0")

        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise CandleValidationError("low exceeds candle body/high")

        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise CandleValidationError("high is below candle body/low")

        return cls(
            datetime=dt,
            open=values["open"],
            high=values["high"],
            low=values["low"],
            close=values["close"],
            volume=values["volume"],
            instrument=instrument,
            timeframe=timeframe,
        )


class BaseMarketDataClient(ABC):
    """Abstract candle stream provider."""

    def __init__(self, logger: logging.Logger, status_handler: StatusHandler | None = None):
        self._logger = logger
        self._status_handler = status_handler

    @abstractmethod
    async def run(self, on_candle: CandleHandler, stop_event: asyncio.Event) -> None:
        """Start candle stream until stop_event is set."""

    async def _emit_status(self, status: str, payload: dict[str, Any]) -> None:
        if self._status_handler is None:
            return
        try:
            await self._status_handler(status, payload)
        except Exception:
            self._logger.exception("Market-data status handler failed")


class DemoMarketDataClient(BaseMarketDataClient):
    """Deterministic pseudo-feed to keep the app operational without API access."""

    def __init__(
        self,
        *,
        instruments: tuple[InstrumentMeta, ...],
        timeframe: str,
        interval_seconds: float,
        base_prices: dict[str, float],
        logger: logging.Logger,
        status_handler: StatusHandler | None = None,
    ):
        super().__init__(logger=logger, status_handler=status_handler)
        self._instruments = instruments
        self._timeframe = timeframe
        self._interval_seconds = max(interval_seconds, 0.25)
        self._rng = random.Random(42)
        self._last_prices = {
            item.symbol: base_prices.get(item.symbol, 100.0 + idx * 10)
            for idx, item in enumerate(instruments)
        }

    async def run(self, on_candle: CandleHandler, stop_event: asyncio.Event) -> None:
        self._logger.info("Demo market data mode started for %d instruments", len(self._instruments))
        await self._emit_status("connected", {"mode": "demo", "instruments": len(self._instruments)})
        while not stop_event.is_set():
            timestamp = datetime.now(tz=timezone.utc).replace(microsecond=0)
            for instrument in self._instruments:
                candle = self._build_candle(instrument.symbol, timestamp)
                await on_candle(candle)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval_seconds)
            except asyncio.TimeoutError:
                continue

    def _build_candle(self, instrument: str, timestamp: datetime) -> Candle:
        previous = self._last_prices[instrument]
        drift = self._rng.uniform(-0.6, 0.6)
        close = max(0.001, previous + drift)
        high = max(previous, close) + self._rng.uniform(0.0, 0.25)
        low = max(0.001, min(previous, close) - self._rng.uniform(0.0, 0.25))
        volume = abs(self._rng.gauss(1000.0, 250.0))
        self._last_prices[instrument] = close

        return Candle.validated(
            dt=timestamp,
            open_=previous,
            high=high,
            low=low,
            close=close,
            volume=volume,
            instrument=instrument,
            timeframe=self._timeframe,
        )


class TInvestMarketDataClient(BaseMarketDataClient):
    """T-Invest SDK stream wrapper with reconnect behavior."""

    def __init__(
        self,
        *,
        token: str,
        instruments: tuple[InstrumentMeta, ...],
        timeframe: str,
        reconnect_delay_seconds: float,
        logger: logging.Logger,
        status_handler: StatusHandler | None = None,
    ):
        super().__init__(logger=logger, status_handler=status_handler)
        self._token = token
        self._instruments = instruments
        self._timeframe = timeframe
        self._reconnect_delay_seconds = max(reconnect_delay_seconds, 1.0)

    async def run(self, on_candle: CandleHandler, stop_event: asyncio.Event) -> None:
        if not self._token.strip():
            raise RuntimeError("INVEST_TOKEN is empty; cannot start T-Invest client")

        if not self._instruments:
            self._logger.warning("No enabled instruments available for T-Invest stream")
            return

        reconnect_attempt = 0
        had_disconnect = False
        while not stop_event.is_set():
            reconnect_attempt += 1
            try:
                self._logger.info("Connecting to T-Invest stream (attempt #%d)", reconnect_attempt)
                await self._emit_status(
                    "connecting",
                    {"mode": "t_invest", "attempt": reconnect_attempt},
                )
                await self._run_once(
                    on_candle=on_candle,
                    stop_event=stop_event,
                    reconnect_attempt=reconnect_attempt,
                    recovered=had_disconnect,
                )
                reconnect_attempt = 0
                had_disconnect = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception("T-Invest stream error: %s", exc)
                had_disconnect = True
                await self._emit_status(
                    "disconnect",
                    {
                        "mode": "t_invest",
                        "attempt": reconnect_attempt,
                        "error": str(exc),
                    },
                )
                self._logger.warning(
                    "Reconnect scheduled in %.1f seconds", self._reconnect_delay_seconds
                )
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=self._reconnect_delay_seconds,
                    )
                except asyncio.TimeoutError:
                    continue

    async def _run_once(
        self,
        on_candle: CandleHandler,
        stop_event: asyncio.Event,
        reconnect_attempt: int,
        recovered: bool,
    ) -> None:
        sdk = _load_tinvest_sdk()
        if sdk is None:
            raise RuntimeError(
                "T-Invest SDK is not available. Install it as described in requirements.txt"
            )

        async_client = sdk["AsyncClient"]
        candle_interval = sdk["CandleInterval"]
        market_data_request = sdk["MarketDataRequest"]
        subscribe_candles_request = sdk["SubscribeCandlesRequest"]
        candle_instrument = sdk["CandleInstrument"]
        subscription_action = sdk["SubscriptionAction"]

        instruments_for_subscribe = []
        for instrument in self._instruments:
            figi = instrument.figi
            if not figi:
                self._logger.warning(
                    "Instrument %s has no FIGI in config and will be skipped for live stream",
                    instrument.symbol,
                )
                continue

            instruments_for_subscribe.append(
                candle_instrument(figi=figi, interval=_map_timeframe(self._timeframe, candle_interval))
            )

        if not instruments_for_subscribe:
            raise RuntimeError("No valid FIGI found for live stream subscription")

        subscribe_request = market_data_request(
            subscribe_candles_request=subscribe_candles_request(
                waiting_close=False,
                subscription_action=subscription_action.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=instruments_for_subscribe,
            )
        )

        figi_to_symbol = {
            item.figi: item.symbol for item in self._instruments if item.figi
        }

        async with async_client(self._token) as client:
            await self._emit_status(
                "connected",
                {
                    "mode": "t_invest",
                    "attempt": reconnect_attempt,
                    "recovered": recovered,
                },
            )
            stream = client.market_data_stream.market_data_stream([subscribe_request])
            async for packet in stream:
                if stop_event.is_set():
                    return

                candle = getattr(packet, "candle", None)
                if candle is None:
                    continue

                symbol = figi_to_symbol.get(getattr(candle, "figi", ""))
                if symbol is None:
                    self._logger.debug("Received candle for unknown FIGI, skipping")
                    continue

                try:
                    normalized = Candle.validated(
                        dt=getattr(candle, "time"),
                        open_=_quotation_to_float(getattr(candle, "open")),
                        high=_quotation_to_float(getattr(candle, "high")),
                        low=_quotation_to_float(getattr(candle, "low")),
                        close=_quotation_to_float(getattr(candle, "close")),
                        volume=float(getattr(candle, "volume", 0.0)),
                        instrument=symbol,
                        timeframe=self._timeframe,
                    )
                except CandleValidationError as exc:
                    self._logger.error("Invalid candle received from stream: %s", exc)
                    continue

                await on_candle(normalized)


def _map_timeframe(timeframe: str, candle_interval_cls: Any) -> Any:
    normalized = timeframe.lower().strip()
    mapping = {
        "1min": "CANDLE_INTERVAL_1_MIN",
        "2min": "CANDLE_INTERVAL_2_MIN",
        "3min": "CANDLE_INTERVAL_3_MIN",
        "5min": "CANDLE_INTERVAL_5_MIN",
        "10min": "CANDLE_INTERVAL_10_MIN",
        "15min": "CANDLE_INTERVAL_15_MIN",
        "30min": "CANDLE_INTERVAL_30_MIN",
        "1hour": "CANDLE_INTERVAL_HOUR",
    }
    attr_name = mapping.get(normalized, "CANDLE_INTERVAL_1_MIN")
    return getattr(candle_interval_cls, attr_name)


def _quotation_to_float(value: Any) -> float:
    if value is None:
        raise CandleValidationError("quotation value is None")

    if isinstance(value, (int, float)):
        return float(value)

    units = getattr(value, "units", None)
    nano = getattr(value, "nano", None)
    if units is not None and nano is not None:
        return float(units) + float(nano) / 1_000_000_000

    as_float = getattr(value, "to_float", None)
    if callable(as_float):
        return float(as_float())

    raise CandleValidationError(f"Unsupported quotation object: {type(value)!r}")


def _load_tinvest_sdk() -> dict[str, Any] | None:
    """Try importing known T-Invest SDK namespaces."""

    providers = [
        "tinkoff.invest",
        "tbank.invest",
        "invest",
    ]

    for module_name in providers:
        try:
            module = __import__(module_name, fromlist=["dummy"])
        except ImportError:
            continue

        required = [
            "AsyncClient",
            "CandleInterval",
            "MarketDataRequest",
            "SubscribeCandlesRequest",
            "CandleInstrument",
            "SubscriptionAction",
        ]

        if all(hasattr(module, attr) for attr in required):
            return {attr: getattr(module, attr) for attr in required}

    return None


def create_market_data_client(
    *,
    mode: str,
    token: str,
    registry: InstrumentRegistry,
    params: dict[str, Any],
    timeframe: str,
    logger: logging.Logger,
    status_handler: StatusHandler | None = None,
) -> BaseMarketDataClient:
    """Factory that creates the configured market data client."""

    enabled = registry.enabled()
    if mode == "t_invest":
        reconnect_delay_seconds = float(params.get("reconnect_delay_seconds", 5.0))
        return TInvestMarketDataClient(
            token=token,
            instruments=enabled,
            timeframe=timeframe,
            reconnect_delay_seconds=reconnect_delay_seconds,
            logger=logger,
            status_handler=status_handler,
        )

    demo_interval = float(params.get("candle_interval_seconds", 1.0))
    base_prices = params.get("base_prices", {})
    if not isinstance(base_prices, dict):
        base_prices = {}

    return DemoMarketDataClient(
        instruments=enabled,
        timeframe=timeframe,
        interval_seconds=demo_interval,
        base_prices={str(key): float(value) for key, value in base_prices.items()},
        logger=logger,
        status_handler=status_handler,
    )
