"""Shared timeframe definitions and T-Invest mappings."""

from __future__ import annotations

from typing import Any

SUPPORTED_TIMEFRAMES = (
    "1min",
    "2min",
    "3min",
    "5min",
    "10min",
    "15min",
    "30min",
    "1hour",
    "4hour",
)

TIMEFRAME_TO_MINUTES = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "4hour": 240,
}

_TINVEST_HISTORY_INTERVAL_ATTR = {
    "1min": "CANDLE_INTERVAL_1_MIN",
    "2min": "CANDLE_INTERVAL_2_MIN",
    "3min": "CANDLE_INTERVAL_3_MIN",
    "5min": "CANDLE_INTERVAL_5_MIN",
    "10min": "CANDLE_INTERVAL_10_MIN",
    "15min": "CANDLE_INTERVAL_15_MIN",
    "30min": "CANDLE_INTERVAL_30_MIN",
    "1hour": "CANDLE_INTERVAL_HOUR",
    "4hour": "CANDLE_INTERVAL_4_HOUR",
}

_TINVEST_SUBSCRIPTION_INTERVAL_ATTR = {
    "1min": "SUBSCRIPTION_INTERVAL_ONE_MINUTE",
    "2min": "SUBSCRIPTION_INTERVAL_2_MIN",
    "3min": "SUBSCRIPTION_INTERVAL_3_MIN",
    "5min": "SUBSCRIPTION_INTERVAL_FIVE_MINUTES",
    "10min": "SUBSCRIPTION_INTERVAL_10_MIN",
    "15min": "SUBSCRIPTION_INTERVAL_FIFTEEN_MINUTES",
    "30min": "SUBSCRIPTION_INTERVAL_30_MIN",
    "1hour": "SUBSCRIPTION_INTERVAL_ONE_HOUR",
    "4hour": "SUBSCRIPTION_INTERVAL_4_HOUR",
}


def normalize_timeframe(timeframe: str) -> str:
    normalized = str(timeframe).strip().lower()
    if not normalized:
        raise ValueError("timeframe must not be empty")
    return normalized


def supported_timeframes() -> tuple[str, ...]:
    return tuple(SUPPORTED_TIMEFRAMES)


def ensure_supported_timeframe(timeframe: str) -> str:
    normalized = normalize_timeframe(timeframe)
    if normalized not in TIMEFRAME_TO_MINUTES:
        raise ValueError(
            "Unsupported timeframe "
            f"{timeframe!r}. Supported values: {', '.join(sorted(SUPPORTED_TIMEFRAMES))}"
        )
    return normalized


def timeframe_minutes(timeframe: str) -> int:
    normalized = ensure_supported_timeframe(timeframe)
    return TIMEFRAME_TO_MINUTES[normalized]


def map_tinvest_history_interval(*, timeframe: str, candle_interval_cls: Any) -> Any:
    normalized = ensure_supported_timeframe(timeframe)
    attr_name = _TINVEST_HISTORY_INTERVAL_ATTR.get(normalized)
    if attr_name is None:
        raise ValueError(
            "Unsupported historical timeframe "
            f"{timeframe!r}. Supported values: {', '.join(sorted(_TINVEST_HISTORY_INTERVAL_ATTR))}"
        )
    if not hasattr(candle_interval_cls, attr_name):
        raise ValueError(
            f"Historical timeframe {timeframe!r} is not supported by installed T-Invest SDK "
            f"(missing {attr_name})"
        )
    return getattr(candle_interval_cls, attr_name)


def map_tinvest_subscription_interval(*, timeframe: str, subscription_interval_cls: Any) -> Any:
    normalized = ensure_supported_timeframe(timeframe)
    attr_name = _TINVEST_SUBSCRIPTION_INTERVAL_ATTR.get(normalized)
    if attr_name is None:
        raise ValueError(
            "Unsupported live timeframe "
            f"{timeframe!r}. Supported values: {', '.join(sorted(_TINVEST_SUBSCRIPTION_INTERVAL_ATTR))}"
        )
    if not hasattr(subscription_interval_cls, attr_name):
        raise ValueError(
            f"Live timeframe {timeframe!r} is not supported by installed T-Invest SDK "
            f"(missing {attr_name})"
        )
    return getattr(subscription_interval_cls, attr_name)
