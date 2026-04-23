"""Shared boolean parsing helpers for config and metadata values."""

from __future__ import annotations

from typing import Any


def to_bool(value: Any, *, default: bool) -> bool:
    """Parse booleans safely from bool/int/str with fallback to default."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
