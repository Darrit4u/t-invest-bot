"""Core market data models and ingestion clients."""

from __future__ import annotations

import asyncio
import importlib
import logging
import random
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from core.instrument_registry import InstrumentMeta, InstrumentRegistry
from core.timeframes import map_tinvest_subscription_interval


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
        wait_for_close: bool,
        reconnect_delay_seconds: float,
        logger: logging.Logger,
        status_handler: StatusHandler | None = None,
    ):
        super().__init__(logger=logger, status_handler=status_handler)
        self._token = token
        self._instruments = instruments
        self._timeframe = timeframe
        self._wait_for_close = wait_for_close
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
            except asyncio.CancelledError as exc:
                if stop_event.is_set():
                    return

                self._logger.warning(
                    "T-Invest stream cancelled unexpectedly (attempt #%d): %s",
                    reconnect_attempt,
                    exc,
                )
                had_disconnect = True
                await self._emit_status(
                    "disconnect",
                    {
                        "mode": "t_invest",
                        "attempt": reconnect_attempt,
                        "error": "stream_cancelled",
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
                "T-Invest SDK is not available. Install package 't-tech-investments' "
                "and ensure import 't_tech.invest' works in the current Python environment."
            )

        async_client = sdk["AsyncClient"]
        candle_instrument = sdk["CandleInstrument"]
        grpc_helpers = sdk["grpc_helpers"]
        market_data_pb2 = sdk["market_data_pb2"]
        market_data_response = sdk["MarketDataResponse"]
        market_data_server_side_stream_request = sdk["MarketDataServerSideStreamRequest"]
        subscribe_candles_request = sdk["SubscribeCandlesRequest"]
        subscription_action = sdk["SubscriptionAction"]
        subscription_interval = sdk["SubscriptionInterval"]

        instruments_for_subscribe = []
        uid_to_symbol: dict[str, str] = {}
        figi_to_symbol: dict[str, str] = {}
        for instrument in self._instruments:
            instrument_id = (instrument.uid or "").strip() or (instrument.figi or "").strip()
            if not instrument_id:
                self._logger.warning(
                    "Instrument %s has no UID/FIGI in config and will be skipped for live stream",
                    instrument.symbol,
                )
                continue

            figi = instrument.figi
            uid = instrument.uid

            instruments_for_subscribe.append(
                candle_instrument(
                    instrument_id=instrument_id,
                    interval=_map_timeframe(self._timeframe, subscription_interval),
                )
            )
            if uid:
                uid_to_symbol[uid] = instrument.symbol
            if figi:
                figi_to_symbol[figi] = instrument.symbol

        if not instruments_for_subscribe:
            raise RuntimeError("No valid UID/FIGI found for live stream subscription")

        async with async_client(self._token) as client:
            await self._emit_status(
                "connected",
                {
                    "mode": "t_invest",
                    "attempt": reconnect_attempt,
                    "recovered": recovered,
                },
            )
            async def _read_stream() -> None:
                request = market_data_server_side_stream_request(
                    subscribe_candles_request=subscribe_candles_request(
                        subscription_action=subscription_action.SUBSCRIPTION_ACTION_SUBSCRIBE,
                        instruments=instruments_for_subscribe,
                        waiting_close=self._wait_for_close,
                    )
                )
                protobuf_request = grpc_helpers.dataclass_to_protobuff(
                    request,
                    market_data_pb2.MarketDataServerSideStreamRequest(),
                )
                stream = _open_market_data_server_side_stream(
                    rpc=client.market_data_stream.stub.MarketDataServerSideStream,
                    request=protobuf_request,
                    metadata=client.market_data_stream.metadata,
                )
                async for protobuf_response in stream:
                    packet = grpc_helpers.protobuf_to_dataclass(
                        protobuf_response,
                        market_data_response,
                    )
                    candle = getattr(packet, "candle", None)
                    if candle is None:
                        continue

                    symbol = uid_to_symbol.get(getattr(candle, "instrument_uid", ""))
                    if symbol is None:
                        symbol = figi_to_symbol.get(getattr(candle, "figi", ""))
                    if symbol is None:
                        self._logger.debug("Received candle for unknown instrument id, skipping")
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

            read_task = asyncio.create_task(_read_stream(), name="market-data-read-loop")
            stop_waiter = asyncio.create_task(stop_event.wait(), name="market-data-stop-waiter")
            try:
                done, pending = await asyncio.wait(
                    {read_task, stop_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_waiter in done:
                    read_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await read_task
                    return

                if read_task in done:
                    exc = read_task.exception()
                    if exc is not None:
                        raise exc
                    return
            finally:
                for task in (read_task, stop_waiter):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(read_task, stop_waiter, return_exceptions=True)


def _open_market_data_server_side_stream(*, rpc: Any, request: Any, metadata: Any) -> Any:
    """
    Open market-data stream across SDK/gRPC variants.

    Preferred path is unary->stream RPC with `request`.
    Some environments require positional request, while legacy variants can require
    request iterator style.
    """

    attempts = (
        lambda: rpc(request=request, metadata=metadata),
        lambda: rpc(request, metadata=metadata),
        lambda: rpc(
            request_iterator=_single_request_iterator(request=request),
            metadata=metadata,
        ),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to open market data stream")


async def _single_request_iterator(*, request: Any):
    yield request


def _map_timeframe(timeframe: str, subscription_interval_cls: Any) -> Any:
    return map_tinvest_subscription_interval(
        timeframe=timeframe,
        subscription_interval_cls=subscription_interval_cls,
    )


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
    """Load supported T-Invest SDK namespace."""

    try:
        module = importlib.import_module("t_tech.invest")
        grpc_helpers = importlib.import_module("t_tech.invest._grpc_helpers")
        market_data_pb2 = importlib.import_module("t_tech.invest.grpc.marketdata_pb2")
        schemas = importlib.import_module("t_tech.invest.schemas")
    except ImportError:
        return None

    required = [
        "AsyncClient",
        "CandleInterval",
        "SubscriptionInterval",
        "CandleInstrument",
        "MarketDataServerSideStreamRequest",
        "SubscribeCandlesRequest",
        "SubscriptionAction",
    ]

    if all(hasattr(module, attr) for attr in required):
        result = {attr: getattr(module, attr) for attr in required}
        result["grpc_helpers"] = grpc_helpers
        result["market_data_pb2"] = market_data_pb2
        result["MarketDataResponse"] = getattr(schemas, "MarketDataResponse")
        return result
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
        wait_for_close = _to_bool(params.get("wait_for_close", True), default=True)
        return TInvestMarketDataClient(
            token=token,
            instruments=enabled,
            timeframe=timeframe,
            wait_for_close=wait_for_close,
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


def _to_bool(value: Any, *, default: bool) -> bool:
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
