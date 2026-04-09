"""Paper-trade lifecycle simulator for accepted signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from core.market_data import Candle
from core.models import SignalDirection, StrategySignal


class TradeStatus(str, Enum):
    NEW_SIGNAL = "new_signal"
    WAITING_ACTIVATION = "waiting_activation"
    ACTIVATED = "activated"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    SL_HIT = "sl_hit"
    EXPIRED = "expired"
    CANCELLED_BY_NEWS = "cancelled_by_news"
    CANCELLED_BY_SESSION_END = "cancelled_by_session_end"


@dataclass(slots=True)
class TradeEvent:
    """Lifecycle event emitted by simulator."""

    trade_id: str
    signal_id: str
    instrument: str
    strategy: str
    event_type: str
    status: str
    event_time: datetime
    price: float | None = None
    size: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulatedTrade:
    """Mutable trade state used by simulator."""

    trade_id: str
    signal_id: str
    instrument: str
    strategy: str
    timeframe: str
    direction: SignalDirection
    status: TradeStatus
    created_at: datetime
    updated_at: datetime
    activated_at: datetime | None
    closed_at: datetime | None
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp1_size: float
    quantity: float
    remaining_qty: float
    entry_fill_price: float | None
    current_stop: float
    tp1_hit_at: datetime | None
    tp2_hit_at: datetime | None
    bars_waiting: int
    bars_in_trade: int
    max_wait_bars: int
    max_trade_bars: int
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    r_multiple: float
    exit_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_closed(self) -> bool:
        return self.closed_at is not None


class TradeSimulator:
    """Maintains signal -> activation -> exit lifecycle."""

    def __init__(self, *, params: dict[str, Any], logger: Any, storage: Any | None = None):
        sim_cfg = params.get("trade_simulator", {}) if isinstance(params.get("trade_simulator", {}), dict) else {}
        self._logger = logger
        self._storage = storage
        self._commission_per_side = float(sim_cfg.get("commission_per_side", 0.0004))
        self._tp1_size = float(sim_cfg.get("tp1_size", 0.5))
        self._max_wait_bars = int(sim_cfg.get("max_wait_bars", 6))
        self._max_trade_bars = int(sim_cfg.get("max_trade_bars", 20))
        self._move_stop_to_breakeven = bool(sim_cfg.get("move_stop_to_breakeven", True))
        self._close_active_on_blackout = bool(sim_cfg.get("close_active_on_blackout", False))
        self._intrabar_stop_priority = bool(sim_cfg.get("intrabar_stop_priority", True))

        self._trades: dict[str, SimulatedTrade] = {}
        self._open_by_instrument: dict[str, set[str]] = {}

    def register_signal(self, signal: StrategySignal, timeframe: str) -> tuple[TradeEvent, ...]:
        trade_id = str(uuid4())
        now = signal.timestamp
        trade = SimulatedTrade(
            trade_id=trade_id,
            signal_id=signal.signal_id,
            instrument=signal.instrument,
            strategy=signal.strategy,
            timeframe=timeframe,
            direction=signal.direction,
            status=TradeStatus.WAITING_ACTIVATION,
            created_at=now,
            updated_at=now,
            activated_at=None,
            closed_at=None,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp1_size=min(max(self._tp1_size, 0.0), 1.0),
            quantity=1.0,
            remaining_qty=1.0,
            entry_fill_price=None,
            current_stop=signal.stop_loss,
            tp1_hit_at=None,
            tp2_hit_at=None,
            bars_waiting=0,
            bars_in_trade=0,
            max_wait_bars=self._max_wait_bars,
            max_trade_bars=self._max_trade_bars,
            gross_pnl=0.0,
            fees_paid=0.0,
            net_pnl=0.0,
            r_multiple=0.0,
            exit_reason=None,
            metadata=dict(signal.metadata) | {"entry_mode": signal.entry_mode},
        )
        self._trades[trade_id] = trade
        self._open_by_instrument.setdefault(trade.instrument, set()).add(trade_id)

        if self._storage is not None:
            self._storage.save_trade(trade)

        event = TradeEvent(
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            strategy=trade.strategy,
            event_type="new_signal",
            status=trade.status.value,
            event_time=now,
            payload={"entry": trade.entry, "stop_loss": trade.stop_loss, "tp1": trade.tp1, "tp2": trade.tp2},
        )
        self._persist_event(event)
        return (event,)

    def process_candle(
        self,
        *,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> tuple[TradeEvent, ...]:
        trade_ids = sorted(self._open_by_instrument.get(candle.instrument, set()))
        if not trade_ids:
            return tuple()

        events: list[TradeEvent] = []
        for trade_id in trade_ids:
            trade = self._trades.get(trade_id)
            if trade is None or trade.is_closed():
                continue

            if trade.status == TradeStatus.WAITING_ACTIVATION:
                events.extend(
                    self._process_waiting_trade(
                        trade=trade,
                        candle=candle,
                        session_active=session_active,
                        blackout_active=blackout_active,
                        blackout_reason=blackout_reason,
                    )
                )
            else:
                events.extend(
                    self._process_active_trade(
                        trade=trade,
                        candle=candle,
                        session_active=session_active,
                        blackout_active=blackout_active,
                        blackout_reason=blackout_reason,
                    )
                )

            if self._storage is not None:
                self._storage.save_trade(trade)

            if trade.is_closed():
                self._open_by_instrument.get(trade.instrument, set()).discard(trade.trade_id)

        return tuple(events)

    def _process_waiting_trade(
        self,
        *,
        trade: SimulatedTrade,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> list[TradeEvent]:
        trade.bars_waiting += 1
        trade.updated_at = candle.datetime

        if blackout_active:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.CANCELLED_BY_NEWS,
                reason=blackout_reason or "blackout",
                event_type="cancelled_by_news",
            )

        if not session_active:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.CANCELLED_BY_SESSION_END,
                reason="session_inactive_before_activation",
                event_type="cancelled_by_session_end",
            )

        if trade.bars_waiting > trade.max_wait_bars:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.EXPIRED,
                reason="activation_timeout",
                event_type="expired",
            )

        fill_price = self._entry_fill_price(trade=trade, candle=candle)
        if fill_price is None:
            return []

        trade.entry_fill_price = fill_price
        trade.activated_at = candle.datetime
        trade.updated_at = candle.datetime
        trade.status = TradeStatus.ACTIVATED
        trade.fees_paid += fill_price * trade.quantity * self._commission_per_side
        trade.net_pnl = trade.gross_pnl - trade.fees_paid

        event = TradeEvent(
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            strategy=trade.strategy,
            event_type="activated",
            status=trade.status.value,
            event_time=candle.datetime,
            price=fill_price,
            size=trade.quantity,
            payload={"bars_waiting": trade.bars_waiting},
        )
        self._persist_event(event)
        return [event]

    def _process_active_trade(
        self,
        *,
        trade: SimulatedTrade,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> list[TradeEvent]:
        trade.bars_in_trade += 1
        trade.updated_at = candle.datetime

        if not session_active:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.CANCELLED_BY_SESSION_END,
                reason="session_inactive",
                event_type="cancelled_by_session_end",
            )

        if blackout_active and self._close_active_on_blackout:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.CANCELLED_BY_NEWS,
                reason=blackout_reason or "blackout",
                event_type="cancelled_by_news",
            )

        events: list[TradeEvent] = []
        if self._stop_hit(trade=trade, candle=candle):
            # Conservative assumption when both SL and TP touched in same candle.
            if self._intrabar_stop_priority:
                events.extend(self._close_at_stop(trade=trade, candle=candle))
                return events

        if self._tp1_hit(trade=trade, candle=candle):
            tp1_size = min(trade.remaining_qty, trade.quantity * trade.tp1_size)
            if tp1_size > 0:
                self._realize_partial_exit(
                    trade=trade,
                    when=candle.datetime,
                    price=trade.tp1,
                    size=tp1_size,
                )
                trade.tp1_hit_at = candle.datetime
                if trade.remaining_qty > 0:
                    trade.status = TradeStatus.TP1_HIT
                if self._move_stop_to_breakeven and trade.entry_fill_price is not None:
                    trade.current_stop = trade.entry_fill_price

                event = TradeEvent(
                    trade_id=trade.trade_id,
                    signal_id=trade.signal_id,
                    instrument=trade.instrument,
                    strategy=trade.strategy,
                    event_type="tp1_hit",
                    status=trade.status.value,
                    event_time=candle.datetime,
                    price=trade.tp1,
                    size=tp1_size,
                    payload={"remaining_qty": trade.remaining_qty},
                )
                self._persist_event(event)
                events.append(event)

        if self._tp2_hit(trade=trade, candle=candle) and trade.remaining_qty > 0:
            self._realize_partial_exit(
                trade=trade,
                when=candle.datetime,
                price=trade.tp2,
                size=trade.remaining_qty,
            )
            trade.tp2_hit_at = candle.datetime
            trade.status = TradeStatus.TP2_HIT
            trade.closed_at = candle.datetime
            trade.exit_reason = "tp2_hit"
            self._refresh_performance_metrics(trade)

            event = TradeEvent(
                trade_id=trade.trade_id,
                signal_id=trade.signal_id,
                instrument=trade.instrument,
                strategy=trade.strategy,
                event_type="tp2_hit",
                status=trade.status.value,
                event_time=candle.datetime,
                price=trade.tp2,
                size=0.0,
                payload={"gross_pnl": trade.gross_pnl, "net_pnl": trade.net_pnl},
            )
            self._persist_event(event)
            events.append(event)
            return events

        if self._stop_hit(trade=trade, candle=candle) and trade.remaining_qty > 0:
            events.extend(self._close_at_stop(trade=trade, candle=candle))
            return events

        if trade.bars_in_trade > trade.max_trade_bars and trade.remaining_qty > 0:
            return self._force_close(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                status=TradeStatus.EXPIRED,
                reason="max_bars_in_trade",
                event_type="expired",
            )

        return events

    def _close_at_stop(self, *, trade: SimulatedTrade, candle: Candle) -> list[TradeEvent]:
        self._realize_partial_exit(
            trade=trade,
            when=candle.datetime,
            price=trade.current_stop,
            size=trade.remaining_qty,
        )
        trade.status = TradeStatus.SL_HIT
        trade.closed_at = candle.datetime
        trade.exit_reason = "stop_hit"
        self._refresh_performance_metrics(trade)

        event = TradeEvent(
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            strategy=trade.strategy,
            event_type="sl_hit",
            status=trade.status.value,
            event_time=candle.datetime,
            price=trade.current_stop,
            size=0.0,
            payload={"gross_pnl": trade.gross_pnl, "net_pnl": trade.net_pnl},
        )
        self._persist_event(event)
        return [event]

    def _force_close(
        self,
        *,
        trade: SimulatedTrade,
        when: datetime,
        price: float,
        status: TradeStatus,
        reason: str,
        event_type: str,
    ) -> list[TradeEvent]:
        if trade.entry_fill_price is None:
            # No fill happened; no PnL or fees.
            trade.status = status
            trade.closed_at = when
            trade.updated_at = when
            trade.exit_reason = reason
        else:
            if trade.remaining_qty > 0:
                self._realize_partial_exit(trade=trade, when=when, price=price, size=trade.remaining_qty)
            trade.status = status
            trade.closed_at = when
            trade.updated_at = when
            trade.exit_reason = reason
            self._refresh_performance_metrics(trade)

        event = TradeEvent(
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            strategy=trade.strategy,
            event_type=event_type,
            status=trade.status.value,
            event_time=when,
            price=price,
            size=0.0,
            payload={"reason": reason, "gross_pnl": trade.gross_pnl, "net_pnl": trade.net_pnl},
        )
        self._persist_event(event)
        return [event]

    def _realize_partial_exit(self, *, trade: SimulatedTrade, when: datetime, price: float, size: float) -> None:
        if size <= 0:
            return

        entry_price = trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry
        sign = 1.0 if trade.direction == SignalDirection.LONG else -1.0
        trade.gross_pnl += sign * (price - entry_price) * size
        trade.fees_paid += price * size * self._commission_per_side
        trade.remaining_qty = max(0.0, trade.remaining_qty - size)
        trade.updated_at = when
        self._refresh_performance_metrics(trade)

    def _refresh_performance_metrics(self, trade: SimulatedTrade) -> None:
        trade.net_pnl = trade.gross_pnl - trade.fees_paid
        reference_entry = trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry
        risk = abs(reference_entry - trade.stop_loss)
        trade.r_multiple = trade.net_pnl / max(risk, 1e-9)

    @staticmethod
    def _tp1_hit(*, trade: SimulatedTrade, candle: Candle) -> bool:
        if trade.tp1_hit_at is not None or trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.high >= trade.tp1
        return candle.low <= trade.tp1

    @staticmethod
    def _tp2_hit(*, trade: SimulatedTrade, candle: Candle) -> bool:
        if trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.high >= trade.tp2
        return candle.low <= trade.tp2

    @staticmethod
    def _stop_hit(*, trade: SimulatedTrade, candle: Candle) -> bool:
        if trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.low <= trade.current_stop
        return candle.high >= trade.current_stop

    @staticmethod
    def _entry_fill_price(*, trade: SimulatedTrade, candle: Candle) -> float | None:
        entry_mode = str(trade.metadata.get("entry_mode", "")).strip().upper()
        if entry_mode == "NEXT_BAR_OPEN":
            return candle.open

        if candle.low <= trade.entry <= candle.high:
            return trade.entry

        if trade.direction == SignalDirection.LONG and candle.open > trade.entry and candle.low > trade.entry:
            return candle.open

        if trade.direction == SignalDirection.SHORT and candle.open < trade.entry and candle.high < trade.entry:
            return candle.open

        return None

    def _persist_event(self, event: TradeEvent) -> None:
        if self._storage is not None:
            self._storage.save_trade_event(event)
        self._logger.info(
            "Trade event | trade_id=%s instrument=%s strategy=%s event=%s status=%s price=%s size=%s payload=%s",
            event.trade_id,
            event.instrument,
            event.strategy,
            event.event_type,
            event.status,
            f"{event.price:.5f}" if event.price is not None else "-",
            f"{event.size:.4f}" if event.size is not None else "-",
            event.payload,
        )

    def open_trades_count(self) -> int:
        return sum(len(items) for items in self._open_by_instrument.values())

    def trades(self) -> tuple[SimulatedTrade, ...]:
        return tuple(self._trades.values())

    def get_trade(self, trade_id: str) -> SimulatedTrade | None:
        return self._trades.get(trade_id)
