"""Execution layer wrapper around trade lifecycle simulator."""

from __future__ import annotations

from dataclasses import dataclass

from core.market_data import Candle
from core.models import Position, StrategySignal, Trade
from core.trade_simulator import TradeEvent, TradeSimulator

_CLOSE_EVENTS = {"tp2_hit", "sl_hit", "expired", "cancelled_by_news", "cancelled_by_session_end"}


@dataclass(frozen=True, slots=True)
class ExecutionOpenResult:
    """Result of opening a position from a signal."""

    position: Position | None
    events: tuple[TradeEvent, ...]


@dataclass(frozen=True, slots=True)
class ExecutionProcessResult:
    """Result of processing market data for already opened positions."""

    events: tuple[TradeEvent, ...]
    closed_trades: tuple[Trade, ...]


class ExecutionEngine:
    """Separated execution/lifecycle layer for strategy signals."""

    def __init__(
        self,
        *,
        simulator: TradeSimulator,
    ):
        self._simulator = simulator

    def open_from_signal(self, *, signal: StrategySignal, timeframe: str) -> ExecutionOpenResult:
        events = self._simulator.register_signal(signal, timeframe=timeframe)
        position: Position | None = None
        if events:
            position = self._simulator.get_position(events[0].trade_id)
        return ExecutionOpenResult(position=position, events=events)

    def process_market(
        self,
        *,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> ExecutionProcessResult:
        events = self._simulator.process_candle(
            candle=candle,
            session_active=session_active,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason,
        )
        closed_trades: list[Trade] = []
        for event in events:
            if event.event_type not in _CLOSE_EVENTS:
                continue
            trade = self._simulator.get_trade_record(event.trade_id)
            if trade is None:
                continue
            closed_trades.append(trade)
        return ExecutionProcessResult(events=events, closed_trades=tuple(closed_trades))

    def open_positions_count(self) -> int:
        return self._simulator.open_trades_count()

    def positions(self) -> tuple[Position, ...]:
        return self._simulator.positions()

    def trade_records(self) -> tuple[Trade, ...]:
        return self._simulator.trade_records()

    def get_trade_record(self, trade_id: str) -> Trade | None:
        return self._simulator.get_trade_record(trade_id)

    def restore_open_trades(self, rows: list[dict[str, object]]) -> int:
        return self._simulator.restore_trade_states(rows)
