"""Instrument metadata registry used by the entire application."""

from __future__ import annotations

from core.config_loader import AppConfig, InstrumentConfig, SessionRule
from domain.models import Instrument

InstrumentMeta = Instrument


class InstrumentRegistry:
    """Single source of truth for instrument-level metadata."""

    def __init__(self, items: dict[str, Instrument]):
        self._items = dict(items)

    @classmethod
    def from_config(cls, config: AppConfig) -> "InstrumentRegistry":
        items: dict[str, Instrument] = {}
        for symbol, instrument in config.instruments.items():
            session_objects = tuple(config.session_rules[name] for name in instrument.sessions)
            strategies = config.strategies_by_instrument.get(symbol, tuple())
            items[symbol] = cls._build_meta(instrument, session_objects, strategies)
        return cls(items)

    @staticmethod
    def _build_meta(
        instrument: InstrumentConfig,
        sessions: tuple[SessionRule, ...],
        allowed_strategies: tuple[str, ...],
    ) -> Instrument:
        return Instrument(
            symbol=instrument.symbol,
            enabled=instrument.enabled,
            uid=instrument.uid,
            figi=instrument.figi,
            ticker=instrument.ticker or instrument.symbol,
            class_code=instrument.class_code,
            tick_size=instrument.tick_size,
            tick_value=instrument.tick_value,
            lot=instrument.lot,
            sessions=sessions,
            allowed_strategies=allowed_strategies,
        )

    def all(self) -> tuple[Instrument, ...]:
        return tuple(self._items.values())

    def enabled(self) -> tuple[Instrument, ...]:
        return tuple(item for item in self._items.values() if item.enabled)

    def get(self, symbol: str) -> Instrument:
        return self._items[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._items
