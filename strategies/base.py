"""Base strategy abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from core.models import MarketRegime, SignalDirection, StrategyContext, StrategySignal


class BaseStrategy(ABC):
    """Base class for all signal strategies."""

    name: str
    allowed_regime: MarketRegime

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        """Return a signal or None when setup is invalid."""

    def _float(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    def _int(self, key: str, default: int) -> int:
        return int(self.params.get(key, default))

    def _str(self, key: str, default: str) -> str:
        value = self.params.get(key, default)
        return str(value)

    def _bool(self, key: str, default: bool) -> bool:
        value = self.params.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def build_signal(
        self,
        *,
        context: StrategyContext,
        direction: SignalDirection,
        entry_mode: str,
        entry: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        metadata: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> StrategySignal:
        ts = timestamp or context.candles[-1].datetime
        return StrategySignal(
            signal_id=str(uuid4()),
            instrument=context.instrument.symbol,
            strategy=self.name,
            regime=context.regime,
            direction=direction,
            timestamp=ts,
            entry_mode=entry_mode,
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            metadata=metadata,
        )
