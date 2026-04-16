"""Trading mode helpers shared across runtime components."""

from __future__ import annotations

from enum import Enum
from typing import Any


class TradingMode(str, Enum):
    """Supported high-level operating modes."""

    INTRADAY = "intraday"
    SWING = "swing"


def resolve_trading_mode(params: dict[str, Any]) -> TradingMode:
    section = params.get("trading", {})
    if not isinstance(section, dict):
        section = {}
    raw = str(section.get("mode", TradingMode.SWING.value)).strip().lower()
    if raw == TradingMode.SWING.value:
        return TradingMode.SWING
    return TradingMode.INTRADAY


def resolve_primary_timeframe(*, params: dict[str, Any], default_timeframe: str) -> str:
    mode = resolve_trading_mode(params)
    if mode != TradingMode.SWING:
        return default_timeframe

    section = params.get("timeframes", {})
    if not isinstance(section, dict):
        return default_timeframe
    value = str(section.get("primary", default_timeframe)).strip()
    return value or default_timeframe


def should_enforce_session_filter(params: dict[str, Any]) -> bool:
    section = params.get("trading", {})
    if not isinstance(section, dict):
        section = {}
    explicit = section.get("enforce_session_filter")
    if explicit is not None:
        return _to_bool(explicit, default=True)

    mode = resolve_trading_mode(params)
    if mode == TradingMode.SWING:
        swing_section = params.get("swing", {})
        if isinstance(swing_section, dict):
            if "enforce_session_filter" in swing_section:
                return _to_bool(swing_section.get("enforce_session_filter"), default=False)
        return False
    return True


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
