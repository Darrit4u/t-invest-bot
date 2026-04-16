"""Helpers for resolving strategy parameters from config."""

from __future__ import annotations

from typing import Any

_INSTRUMENT_SECTION_KEYS = ("by_instrument", "instruments")


def resolve_strategy_params(
    *,
    section: dict[str, Any],
    strategy_name: str,
    instrument_symbol: str | None = None,
    trading_mode: str | None = None,
) -> dict[str, Any]:
    """Return effective params for one strategy (optionally per instrument)."""
    defaults = _strategy_defaults(section=section, strategy_name=strategy_name)
    mode_section = _mode_section(section=section, trading_mode=trading_mode)
    if mode_section:
        defaults = dict(defaults) | _strategy_defaults(section=mode_section, strategy_name=strategy_name)
    if not instrument_symbol:
        return defaults

    instrument_overrides = _instrument_overrides(section=section, instrument_symbol=instrument_symbol)
    override = instrument_overrides.get(strategy_name, {})
    merged = dict(defaults)
    if isinstance(override, dict):
        merged |= dict(override)

    if mode_section:
        mode_instrument_overrides = _instrument_overrides(
            section=mode_section,
            instrument_symbol=instrument_symbol,
        )
        mode_override = mode_instrument_overrides.get(strategy_name, {})
        if isinstance(mode_override, dict):
            merged |= dict(mode_override)
    return merged


def iter_strategy_param_variants(
    *,
    section: dict[str, Any],
    strategy_name: str,
    trading_mode: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return unique effective configs across default + instrument overrides."""
    variants: list[dict[str, Any]] = [
        resolve_strategy_params(
            section=section,
            strategy_name=strategy_name,
            trading_mode=trading_mode,
        )
    ]
    for symbol in _instrument_symbols(section=section):
        resolved = resolve_strategy_params(
            section=section,
            strategy_name=strategy_name,
            instrument_symbol=symbol,
            trading_mode=trading_mode,
        )
        if resolved not in variants:
            variants.append(resolved)
    mode_section = _mode_section(section=section, trading_mode=trading_mode)
    if mode_section:
        for symbol in _instrument_symbols(section=mode_section):
            resolved = resolve_strategy_params(
                section=section,
                strategy_name=strategy_name,
                instrument_symbol=symbol,
                trading_mode=trading_mode,
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


def _mode_section(*, section: dict[str, Any], trading_mode: str | None) -> dict[str, Any]:
    normalized_mode = (trading_mode or "").strip().lower()
    if not normalized_mode:
        return {}

    by_mode = section.get("by_mode", {})
    if isinstance(by_mode, dict):
        candidate = by_mode.get(normalized_mode)
        if isinstance(candidate, dict):
            return candidate

    modes = section.get("modes", {})
    if isinstance(modes, dict):
        candidate = modes.get(normalized_mode)
        if isinstance(candidate, dict):
            return candidate

    return {}
