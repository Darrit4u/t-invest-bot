"""Instrument metadata registry used by the entire application."""

from __future__ import annotations

from dataclasses import dataclass

from core.config_loader import AppConfig, InstrumentConfig, SessionRule


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    """Resolved instrument metadata and strategy permissions."""

    symbol: str
    enabled: bool
    uid: str | None
    figi: str | None
    ticker: str
    class_code: str | None
    tick_size: float
    tick_value: float
    lot: int
    sessions: tuple[SessionRule, ...]
    allowed_strategies: tuple[str, ...]


class InstrumentRegistry:
    """Single source of truth for instrument-level metadata."""

    def __init__(self, items: dict[str, InstrumentMeta]):
        self._items = dict(items)

    @classmethod
    def from_config(cls, config: AppConfig) -> "InstrumentRegistry":
        items: dict[str, InstrumentMeta] = {}
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
    ) -> InstrumentMeta:
        return InstrumentMeta(
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

    def all(self) -> tuple[InstrumentMeta, ...]:
        return tuple(self._items.values())

    def enabled(self) -> tuple[InstrumentMeta, ...]:
        return tuple(item for item in self._items.values() if item.enabled)

    def get(self, symbol: str) -> InstrumentMeta:
        return self._items[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._items
