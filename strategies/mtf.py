"""Multi-timeframe helpers for strategy direction alignment."""

from __future__ import annotations

from typing import Any

from core.market_data import Candle
from core.models import SignalDirection


_TIMEFRAME_TO_MINUTES = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
}


def mtf_alignment(
    *,
    enabled: bool,
    candles: list[Candle],
    source_timeframe: str,
    direction: SignalDirection,
    trend_timeframe: str = "1hour",
    setup_timeframe: str = "15min",
    fast_ema: int = 8,
    slow_ema: int = 21,
    slope_bars: int = 2,
) -> tuple[bool, dict[str, Any]]:
    if not enabled:
        return True, {
            "mtf_filter_enabled": False,
        }

    trend_direction = _timeframe_direction(
        candles=candles,
        source_timeframe=source_timeframe,
        target_timeframe=trend_timeframe,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        slope_bars=slope_bars,
    )
    setup_direction = _timeframe_direction(
        candles=candles,
        source_timeframe=source_timeframe,
        target_timeframe=setup_timeframe,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        slope_bars=slope_bars,
    )

    metadata = {
        "mtf_filter_enabled": True,
        "mtf_trend_tf": trend_timeframe,
        "mtf_setup_tf": setup_timeframe,
        "mtf_trend_direction": trend_direction.value if trend_direction is not None else "NONE",
        "mtf_setup_direction": setup_direction.value if setup_direction is not None else "NONE",
    }

    if trend_direction is None or setup_direction is None:
        return False, metadata

    aligned = trend_direction == direction and setup_direction == direction
    return aligned, metadata


def _timeframe_direction(
    *,
    candles: list[Candle],
    source_timeframe: str,
    target_timeframe: str,
    fast_ema: int,
    slow_ema: int,
    slope_bars: int,
) -> SignalDirection | None:
    aggregated = _aggregate_timeframe(
        candles=candles,
        source_timeframe=source_timeframe,
        target_timeframe=target_timeframe,
    )
    if not aggregated:
        return None

    closes = [item.close for item in aggregated]
    if len(closes) < slow_ema + slope_bars + 1:
        return None

    fast_series = _ema_series(closes, period=fast_ema)
    slow_series = _ema_series(closes, period=slow_ema)
    fast_last = fast_series[-1]
    slow_last = slow_series[-1]
    slope = fast_series[-1] - fast_series[-1 - slope_bars]

    if fast_last > slow_last and slope > 0:
        return SignalDirection.LONG
    if fast_last < slow_last and slope < 0:
        return SignalDirection.SHORT
    return None


def _aggregate_timeframe(
    *,
    candles: list[Candle],
    source_timeframe: str,
    target_timeframe: str,
) -> list[Candle]:
    source_minutes = _TIMEFRAME_TO_MINUTES.get(source_timeframe.lower().strip())
    target_minutes = _TIMEFRAME_TO_MINUTES.get(target_timeframe.lower().strip())
    if source_minutes is None or target_minutes is None:
        return []
    if target_minutes < source_minutes:
        return []
    if target_minutes == source_minutes:
        return candles
    if target_minutes % source_minutes != 0:
        return []

    bucket_seconds = target_minutes * 60
    items: list[Candle] = []
    current_bucket: int | None = None
    current: dict[str, Any] | None = None

    for candle in candles:
        bucket = int(candle.datetime.timestamp()) // bucket_seconds
        if current_bucket is None or bucket != current_bucket:
            if current is not None:
                items.append(
                    Candle.validated(
                        dt=current["dt"],
                        open_=current["open"],
                        high=current["high"],
                        low=current["low"],
                        close=current["close"],
                        volume=current["volume"],
                        instrument=current["instrument"],
                        timeframe=target_timeframe,
                    )
                )
            current_bucket = bucket
            current = {
                "dt": candle.datetime,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "instrument": candle.instrument,
            }
            continue

        assert current is not None
        current["dt"] = candle.datetime
        current["high"] = max(current["high"], candle.high)
        current["low"] = min(current["low"], candle.low)
        current["close"] = candle.close
        current["volume"] += candle.volume

    if current is not None:
        items.append(
            Candle.validated(
                dt=current["dt"],
                open_=current["open"],
                high=current["high"],
                low=current["low"],
                close=current["close"],
                volume=current["volume"],
                instrument=current["instrument"],
                timeframe=target_timeframe,
            )
        )
    return items


def _ema_series(values: list[float], *, period: int) -> list[float]:
    alpha = 2.0 / (max(period, 2) + 1.0)
    result: list[float] = []
    ema_value = values[0]
    result.append(ema_value)
    for value in values[1:]:
        ema_value = (alpha * value) + ((1.0 - alpha) * ema_value)
        result.append(ema_value)
    return result
