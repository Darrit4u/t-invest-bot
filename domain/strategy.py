"""Domain-level strategy contracts."""

from __future__ import annotations

from typing import Protocol, Sequence

from domain.models import Signal, StrategyContext


class Strategy(Protocol):
    """Minimal strategy interface for phased strategy/execution separation."""

    strategy_id: str

    def generate_signals(self, context: StrategyContext) -> Sequence[Signal]:
        ...
