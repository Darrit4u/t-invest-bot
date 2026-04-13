"""Helpers for resolving strategy parameters from config."""

from __future__ import annotations

from typing import Any

_INSTRUMENT_SECTION_KEYS = ("by_instrument", "instruments")


def resolve_strategy_params(
    *,
    section: dict[str, Any],
    strategy_name: str,
    instrument_symbol: str | None = None,
) -> dict[str, Any]:
    """Return effective params for one strategy (optionally per instrument)."""
    defaults = _strategy_defaults(section=section, strategy_name=strategy_name)
    if not instrument_symbol:
        return defaults

    instrument_overrides = _instrument_overrides(section=section, instrument_symbol=instrument_symbol)
    override = instrument_overrides.get(strategy_name, {})
    if not isinstance(override, dict):
        return defaults
    return dict(defaults) | dict(override)


def iter_strategy_param_variants(
    *,
    section: dict[str, Any],
    strategy_name: str,
) -> tuple[dict[str, Any], ...]:
    """Return unique effective configs across default + instrument overrides."""
    variants: list[dict[str, Any]] = [resolve_strategy_params(section=section, strategy_name=strategy_name)]
    for symbol in _instrument_symbols(section=section):
        resolved = resolve_strategy_params(
            section=section,
            strategy_name=strategy_name,
            instrument_symbol=symbol,
        )
        if resolved not in variants:
            variants.append(resolved)
    return tuple(variants)


def _strategy_defaults(*, section: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    defaults_section = section.get("defaults", {})
    if isinstance(defaults_section, dict) and strategy_name in defaults_section:
        candidate = defaults_section.get(strategy_name)
        if isinstance(candidate, dict):
            return dict(candidate)

    # Backward compatibility with legacy flat structure:
    # strategy_params.<strategy_name>: {...}
    legacy = section.get(strategy_name, {})
    if isinstance(legacy, dict):
        return dict(legacy)
    return {}


def _instrument_overrides(*, section: dict[str, Any], instrument_symbol: str) -> dict[str, Any]:
    normalized = instrument_symbol.strip().lower()
    if not normalized:
        return {}

    for symbol, payload in _instrument_sections(section=section).items():
        if str(symbol).strip().lower() != normalized:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _instrument_symbols(*, section: dict[str, Any]) -> tuple[str, ...]:
    symbols = [str(symbol).strip() for symbol in _instrument_sections(section=section)]
    return tuple(symbol for symbol in symbols if symbol)


def _instrument_sections(*, section: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in _INSTRUMENT_SECTION_KEYS:
        raw = section.get(key, {})
        if not isinstance(raw, dict):
            continue
        for symbol, payload in raw.items():
            merged[str(symbol)] = payload
    return merged
