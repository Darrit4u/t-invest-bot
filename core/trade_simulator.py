"""Paper-trade lifecycle simulator for accepted signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any
from uuid import uuid4

from core.bool_parser import to_bool
from core.lifecycle_policy import BaseLifecyclePolicy, LifecycleCloseAction, build_lifecycle_policy
from core.market_data import Candle
from core.models import Position, SignalDirection, StrategySignal, Trade
from core.post_fill_validation import PostFillValidationConfig, validate_post_fill


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
class TradeState:
    """Mutable trade lifecycle state used by simulator internals."""

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


@dataclass(frozen=True, slots=True)
class SlippageModelConfig:
    enabled: bool
    model: str
    entry_ticks: float
    stop_exit_ticks: float
    target_exit_ticks: float
    forced_exit_ticks: float
    by_instrument: dict[str, dict[str, float]]


class TradeSimulator:
    """Maintains signal -> activation -> exit lifecycle."""

    def __init__(self, *, params: dict[str, Any], logger: Any, storage: Any | None = None):
        sim_cfg = params.get("trade_simulator", {}) if isinstance(params.get("trade_simulator", {}), dict) else {}
        execution_cfg = params.get("execution", {}) if isinstance(params.get("execution", {}), dict) else {}
        filter_cfg = params.get("signal_filter", {}) if isinstance(params.get("signal_filter", {}), dict) else {}
        self._logger = logger
        self._storage = storage
        self._commission_per_side = float(sim_cfg.get("commission_per_side", 0.0004))
        self._tp1_size = float(sim_cfg.get("tp1_size", 0.5))
        self._max_wait_bars = int(sim_cfg.get("max_wait_bars", 6))
        self._max_trade_bars = int(sim_cfg.get("max_trade_bars", 20))
        self._move_stop_to_breakeven = to_bool(sim_cfg.get("move_stop_to_breakeven", True), default=True)
        self._close_active_on_blackout = to_bool(
            sim_cfg.get("close_active_on_blackout", False),
            default=False,
        )
        self._fill_model = _parse_fill_model(execution_cfg=execution_cfg, sim_cfg=sim_cfg)
        self._intrabar_stop_priority = _parse_intrabar_conflict_policy(
            execution_cfg=execution_cfg,
            sim_cfg=sim_cfg,
        )
        self._close_profitable_on_session_end = to_bool(
            sim_cfg.get("close_profitable_on_session_end", False),
            default=False,
        )
        self._revalidate_after_fill = to_bool(sim_cfg.get("revalidate_after_fill", True), default=True)
        self._min_rr_after_fill = float(sim_cfg.get("min_rr_after_fill", 0.50))
        self._min_expected_edge_after_fees = float(
            sim_cfg.get("min_expected_edge_after_fees", 0.0)
        )
        commission_roundtrip = float(
            sim_cfg.get(
                "commission_roundtrip",
                filter_cfg.get("commission_roundtrip", self._commission_per_side * 2.0),
            )
        )
        safety_multiplier = float(
            sim_cfg.get("safety_multiplier", filter_cfg.get("safety_multiplier", 1.5))
        )
        self._post_fill_cfg = PostFillValidationConfig(
            commission_roundtrip=commission_roundtrip,
            safety_multiplier=safety_multiplier,
            min_rr_after_fill=self._min_rr_after_fill,
            min_expected_edge_after_fees=self._min_expected_edge_after_fees,
        )
        self._lifecycle_policy: BaseLifecyclePolicy = build_lifecycle_policy(
            params=params,
            max_trade_bars=self._max_trade_bars,
            close_profitable_on_session_end=self._close_profitable_on_session_end,
        )
        self._slippage = _parse_slippage_config(params=params, sim_cfg=sim_cfg)

        self._trades: dict[str, TradeState] = {}
        self._open_by_instrument: dict[str, set[str]] = {}

    def register_signal(self, signal: StrategySignal, timeframe: str) -> tuple[TradeEvent, ...]:
        trade_id = str(uuid4())
        now = signal.timestamp
        signal_meta = dict(signal.metadata) if isinstance(signal.metadata, dict) else {}
        quantity = _extract_position_qty(signal_meta)
        trade = TradeState(
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
            quantity=quantity,
            remaining_qty=quantity,
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
            metadata=signal_meta | {"entry_mode": signal.entry_mode},
        )
        self._normalize_trade_levels(trade)
        self._trades[trade_id] = trade
        self._open_by_instrument.setdefault(trade.instrument, set()).add(trade_id)

        if self._storage is not None:
            self._persist_trade_state(trade)

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
                self._persist_trade_state(trade)

            if trade.is_closed():
                self._open_by_instrument.get(trade.instrument, set()).discard(trade.trade_id)

        return tuple(events)

    def _process_waiting_trade(
        self,
        *,
        trade: TradeState,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> list[TradeEvent]:
        if candle.datetime <= trade.created_at:
            return []

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

        session_action = self._lifecycle_policy.waiting_session_action(session_active=session_active)
        if session_action is not None:
            return self._force_close_with_action(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                action=session_action,
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
        fill_price = self._finalize_fill_price(
            trade=trade,
            raw_price=fill_price,
            execution_kind="entry",
        )

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
        events = [event]

        validation_ok, validation_reason = self._validate_after_fill(trade=trade)
        if self._revalidate_after_fill and not validation_ok:
            events.extend(
                self._force_close(
                    trade=trade,
                    when=candle.datetime,
                    price=fill_price,
                    status=TradeStatus.EXPIRED,
                    reason=validation_reason or "poor_rr_after_fill",
                    event_type="expired",
                )
            )
        return events

    def _process_active_trade(
        self,
        *,
        trade: TradeState,
        candle: Candle,
        session_active: bool,
        blackout_active: bool,
        blackout_reason: str | None,
    ) -> list[TradeEvent]:
        trade.bars_in_trade += 1
        trade.updated_at = candle.datetime

        session_action = self._lifecycle_policy.active_session_action(
            session_active=session_active,
            profitable=self._would_close_in_profit(trade=trade, price=candle.close),
        )
        if session_action is not None:
            return self._force_close_with_action(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                action=session_action,
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
                tp1_price = self._target_exit_price(trade=trade, candle=candle, level=trade.tp1)
                tp1_price = self._finalize_fill_price(
                    trade=trade,
                    raw_price=tp1_price,
                    execution_kind="target_exit",
                )
                self._realize_partial_exit(
                    trade=trade,
                    when=candle.datetime,
                    price=tp1_price,
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
                    price=tp1_price,
                    size=tp1_size,
                    payload={"remaining_qty": trade.remaining_qty},
                )
                self._persist_event(event)
                events.append(event)

        if self._tp2_hit(trade=trade, candle=candle) and trade.remaining_qty > 0:
            tp2_price = self._target_exit_price(trade=trade, candle=candle, level=trade.tp2)
            tp2_price = self._finalize_fill_price(
                trade=trade,
                raw_price=tp2_price,
                execution_kind="target_exit",
            )
            self._realize_partial_exit(
                trade=trade,
                when=candle.datetime,
                price=tp2_price,
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
                price=tp2_price,
                size=0.0,
                payload={"gross_pnl": trade.gross_pnl, "net_pnl": trade.net_pnl},
            )
            self._persist_event(event)
            events.append(event)
            return events

        if self._stop_hit(trade=trade, candle=candle) and trade.remaining_qty > 0:
            events.extend(self._close_at_stop(trade=trade, candle=candle))
            return events

        expiry_action = self._lifecycle_policy.holding_expiry_action(
            opened_at=trade.activated_at or trade.created_at,
            now=candle.datetime,
            bars_in_trade=trade.bars_in_trade,
        )
        if expiry_action is not None and trade.remaining_qty > 0:
            return self._force_close_with_action(
                trade=trade,
                when=candle.datetime,
                price=candle.close,
                action=expiry_action,
            )

        return events

    def _close_at_stop(self, *, trade: TradeState, candle: Candle) -> list[TradeEvent]:
        stop_price = self._stop_exit_price(trade=trade, candle=candle)
        stop_price = self._finalize_fill_price(
            trade=trade,
            raw_price=stop_price,
            execution_kind="stop_exit",
        )
        self._realize_partial_exit(
            trade=trade,
            when=candle.datetime,
            price=stop_price,
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
            price=stop_price,
            size=0.0,
            payload={"gross_pnl": trade.gross_pnl, "net_pnl": trade.net_pnl},
        )
        self._persist_event(event)
        return [event]

    def _force_close(
        self,
        *,
        trade: TradeState,
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
            exit_price = self._finalize_fill_price(
                trade=trade,
                raw_price=price,
                execution_kind="forced_exit",
            )
            if trade.remaining_qty > 0:
                self._realize_partial_exit(trade=trade, when=when, price=exit_price, size=trade.remaining_qty)
            trade.status = status
            trade.closed_at = when
            trade.updated_at = when
            trade.exit_reason = reason
            self._refresh_performance_metrics(trade)
            price = exit_price

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

    def _force_close_with_action(
        self,
        *,
        trade: TradeState,
        when: datetime,
        price: float,
        action: LifecycleCloseAction,
    ) -> list[TradeEvent]:
        return self._force_close(
            trade=trade,
            when=when,
            price=price,
            status=self._status_from_value(action.status),
            reason=action.reason,
            event_type=action.event_type,
        )

    def _realize_partial_exit(self, *, trade: TradeState, when: datetime, price: float, size: float) -> None:
        if size <= 0:
            return

        entry_price = trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry
        sign = 1.0 if trade.direction == SignalDirection.LONG else -1.0
        trade.gross_pnl += sign * (price - entry_price) * size
        trade.fees_paid += price * size * self._commission_per_side
        trade.remaining_qty = max(0.0, trade.remaining_qty - size)
        trade.updated_at = when
        trade.metadata["last_exit_price"] = float(price)
        self._refresh_performance_metrics(trade)

    def _refresh_performance_metrics(self, trade: TradeState) -> None:
        trade.net_pnl = trade.gross_pnl - trade.fees_paid
        reference_entry = trade.entry_fill_price if trade.entry_fill_price is not None else trade.entry
        risk_per_contract = abs(reference_entry - trade.stop_loss)
        total_position_risk = risk_per_contract * max(float(trade.quantity), 1e-9)
        trade.r_multiple = trade.net_pnl / max(total_position_risk, 1e-9)

    def _would_close_in_profit(self, *, trade: TradeState, price: float) -> bool:
        if trade.entry_fill_price is None or trade.remaining_qty <= 0:
            return False
        sign = 1.0 if trade.direction == SignalDirection.LONG else -1.0
        unrealized = sign * (price - trade.entry_fill_price) * trade.remaining_qty
        projected_gross = trade.gross_pnl + unrealized
        projected_fees = trade.fees_paid + (price * trade.remaining_qty * self._commission_per_side)
        return (projected_gross - projected_fees) > 0.0

    @staticmethod
    def _tp1_hit(*, trade: TradeState, candle: Candle) -> bool:
        if trade.tp1_hit_at is not None or trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.high >= trade.tp1
        return candle.low <= trade.tp1

    @staticmethod
    def _tp2_hit(*, trade: TradeState, candle: Candle) -> bool:
        if trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.high >= trade.tp2
        return candle.low <= trade.tp2

    @staticmethod
    def _stop_hit(*, trade: TradeState, candle: Candle) -> bool:
        if trade.remaining_qty <= 0:
            return False
        if trade.direction == SignalDirection.LONG:
            return candle.low <= trade.current_stop
        return candle.high >= trade.current_stop

    def _entry_fill_price(self, *, trade: TradeState, candle: Candle) -> float | None:
        if self._fill_model == "conservative_next_bar_open":
            return candle.open

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

    def _normalize_trade_levels(self, trade: TradeState) -> None:
        tick_size = self._tick_size_for_trade(trade)
        trade.entry = self._round_to_tick(price=trade.entry, tick_size=tick_size, mode="nearest")
        trade.stop_loss = self._round_to_tick(price=trade.stop_loss, tick_size=tick_size, mode="nearest")
        trade.tp1 = self._round_to_tick(price=trade.tp1, tick_size=tick_size, mode="nearest")
        trade.tp2 = self._round_to_tick(price=trade.tp2, tick_size=tick_size, mode="nearest")

        if trade.direction == SignalDirection.LONG:
            trade.stop_loss = min(trade.stop_loss, trade.entry - tick_size)
            trade.tp1 = max(trade.tp1, trade.entry + tick_size)
            trade.tp2 = max(trade.tp2, trade.tp1 + tick_size)
        else:
            trade.stop_loss = max(trade.stop_loss, trade.entry + tick_size)
            trade.tp1 = min(trade.tp1, trade.entry - tick_size)
            trade.tp2 = min(trade.tp2, trade.tp1 - tick_size)
        trade.current_stop = trade.stop_loss

    def _tick_size_for_trade(self, trade: TradeState) -> float:
        meta = trade.metadata if isinstance(trade.metadata, dict) else {}
        raw = meta.get("tick_size")
        try:
            tick = float(raw)
        except (TypeError, ValueError):
            tick = 0.01
        return max(tick, 1e-9)

    @staticmethod
    def _round_to_tick(*, price: float, tick_size: float, mode: str) -> float:
        tick = max(float(tick_size), 1e-9)
        value = float(price) / tick
        if mode == "up":
            return max(tick, math.ceil(value) * tick)
        if mode == "down":
            return max(tick, math.floor(value) * tick)
        return max(tick, round(value) * tick)

    def _finalize_fill_price(
        self,
        *,
        trade: TradeState,
        raw_price: float,
        execution_kind: str,
    ) -> float:
        tick_size = self._tick_size_for_trade(trade)
        is_buy = self._is_buy_action(trade=trade, execution_kind=execution_kind)
        price = float(raw_price)
        slip_ticks = self._slippage_ticks(trade=trade, execution_kind=execution_kind)
        if slip_ticks > 0.0:
            delta = slip_ticks * tick_size
            price = price + delta if is_buy else price - delta
        return self._round_to_tick(
            price=price,
            tick_size=tick_size,
            mode="up" if is_buy else "down",
        )

    @staticmethod
    def _is_buy_action(*, trade: TradeState, execution_kind: str) -> bool:
        if execution_kind == "entry":
            return trade.direction == SignalDirection.LONG
        return trade.direction == SignalDirection.SHORT

    def _slippage_ticks(self, *, trade: TradeState, execution_kind: str) -> float:
        if not self._slippage.enabled:
            return 0.0
        if self._slippage.model != "fixed_ticks":
            return 0.0

        overrides = self._slippage.by_instrument.get(trade.instrument, {})
        if execution_kind == "entry":
            return float(overrides.get("entry_ticks", self._slippage.entry_ticks))
        if execution_kind == "stop_exit":
            return float(overrides.get("stop_exit_ticks", self._slippage.stop_exit_ticks))
        if execution_kind == "target_exit":
            return float(overrides.get("target_exit_ticks", self._slippage.target_exit_ticks))
        if execution_kind == "forced_exit":
            return float(overrides.get("forced_exit_ticks", self._slippage.forced_exit_ticks))
        return 0.0

    def _persist_event(self, event: TradeEvent) -> None:
        if self._storage is not None:
            self._persist_trade_event(event)
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

    def _persist_trade_state(self, trade: TradeState) -> None:
        if self._storage is None:
            return
        if hasattr(self._storage, "save_trade_state_snapshot"):
            self._storage.save_trade_state_snapshot(trade)
            return
        self._storage.save_trade(trade)

    def _persist_trade_event(self, event: TradeEvent) -> None:
        if self._storage is None:
            return
        if hasattr(self._storage, "save_trade_lifecycle_event"):
            self._storage.save_trade_lifecycle_event(event)
            return
        self._storage.save_trade_event(event)

    def open_trades_count(self) -> int:
        return sum(len(items) for items in self._open_by_instrument.values())

    def restore_trade_states(self, rows: list[dict[str, Any]]) -> int:
        restored = 0
        for row in rows:
            trade = self._trade_state_from_row(row)
            if trade is None:
                continue
            if trade.trade_id in self._trades:
                continue
            self._trades[trade.trade_id] = trade
            if not trade.is_closed():
                self._open_by_instrument.setdefault(trade.instrument, set()).add(trade.trade_id)
            restored += 1
        return restored

    def trades(self) -> tuple[TradeState, ...]:
        return tuple(self._trades.values())

    def get_trade(self, trade_id: str) -> TradeState | None:
        return self._trades.get(trade_id)

    def get_position(self, trade_id: str) -> Position | None:
        trade = self._trades.get(trade_id)
        if trade is None or trade.is_closed():
            return None
        return self._to_position(trade)

    def get_trade_record(self, trade_id: str) -> Trade | None:
        trade = self._trades.get(trade_id)
        if trade is None:
            return None
        return self._to_trade(trade)

    def positions(self) -> tuple[Position, ...]:
        open_trades = [
            trade
            for trade in self._trades.values()
            if not trade.is_closed()
        ]
        return tuple(self._to_position(item) for item in open_trades)

    def trade_records(self) -> tuple[Trade, ...]:
        return tuple(self._to_trade(item) for item in self._trades.values())

    def _validate_after_fill(self, *, trade: TradeState) -> tuple[bool, str | None]:
        if trade.entry_fill_price is None:
            return False, "poor_rr_after_fill"

        validation = validate_post_fill(
            direction=trade.direction,
            stop_loss=trade.stop_loss,
            tp1=trade.tp1,
            entry_price=float(trade.entry_fill_price),
            config=self._post_fill_cfg,
        )
        trade.metadata["post_fill_risk"] = validation.metrics.risk
        trade.metadata["post_fill_reward"] = validation.metrics.reward
        trade.metadata["post_fill_rr"] = validation.metrics.post_fill_rr
        trade.metadata["expected_edge_after_fees"] = validation.metrics.expected_edge_after_fees
        trade.metadata["post_fill_validation_passed"] = bool(validation.accepted)
        trade.metadata["post_fill_validation_reason"] = validation.reason or ""

        if not validation.accepted:
            return False, validation.reason or "poor_rr_after_fill"
        return True, None

    def _stop_exit_price(self, *, trade: TradeState, candle: Candle) -> float:
        if self._lifecycle_policy.gap_risk_handling != "conservative":
            return trade.current_stop

        if trade.direction == SignalDirection.LONG and candle.open < trade.current_stop:
            return candle.open
        if trade.direction == SignalDirection.SHORT and candle.open > trade.current_stop:
            return candle.open
        return trade.current_stop

    def _target_exit_price(self, *, trade: TradeState, candle: Candle, level: float) -> float:
        if self._lifecycle_policy.gap_risk_handling != "conservative":
            return level

        if trade.direction == SignalDirection.LONG and candle.open > level:
            return candle.open
        if trade.direction == SignalDirection.SHORT and candle.open < level:
            return candle.open
        return level

    @staticmethod
    def _status_from_value(value: str) -> TradeStatus:
        for status in TradeStatus:
            if status.value == value:
                return status
        return TradeStatus.EXPIRED

    @staticmethod
    def _to_position(trade: TradeState) -> Position:
        entry_price = float(trade.entry_fill_price) if trade.entry_fill_price is not None else float(trade.entry)
        opened_at = trade.activated_at or trade.created_at
        return Position(
            position_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            side=trade.direction,
            entry_price=entry_price,
            size=float(trade.remaining_qty),
            opened_at=opened_at,
            stop_loss=float(trade.current_stop),
            take_profit=float(trade.tp2),
            strategy_id=trade.strategy,
            status=trade.status.value,
            timeframe=trade.timeframe,
            metadata=dict(trade.metadata),
        )

    @staticmethod
    def _to_trade(trade: TradeState) -> Trade:
        entry_price = float(trade.entry_fill_price) if trade.entry_fill_price is not None else float(trade.entry)
        qty = max(float(trade.quantity), 1e-9)
        pnl_pct = trade.net_pnl / (abs(entry_price) * qty)
        metadata = dict(trade.metadata) | {
            "stop_loss": float(trade.stop_loss),
            "tp1": float(trade.tp1),
            "tp2": float(trade.tp2),
        }
        return Trade(
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            instrument=trade.instrument,
            side=trade.direction,
            entry_price=entry_price,
            exit_price=TradeSimulator._infer_exit_price(trade),
            size=float(trade.quantity),
            opened_at=trade.activated_at or trade.created_at,
            closed_at=trade.closed_at,
            pnl=float(trade.net_pnl),
            pnl_pct=float(pnl_pct),
            strategy_id=trade.strategy,
            status=trade.status.value,
            timeframe=trade.timeframe,
            created_at=trade.created_at,
            activated_at=trade.activated_at,
            gross_pnl=float(trade.gross_pnl),
            fees_paid=float(trade.fees_paid),
            r_multiple=float(trade.r_multiple),
            exit_reason=trade.exit_reason,
            entry_fill_price=(
                float(trade.entry_fill_price) if trade.entry_fill_price is not None else None
            ),
            remaining_qty=float(trade.remaining_qty),
            bars_waiting=int(trade.bars_waiting),
            bars_in_trade=int(trade.bars_in_trade),
            metadata=metadata,
        )

    @staticmethod
    def _infer_exit_price(trade: TradeState) -> float | None:
        if trade.closed_at is None:
            return None
        raw = (trade.metadata or {}).get("last_exit_price")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        status_value = str(getattr(trade.status, "value", trade.status))
        if status_value == "tp2_hit":
            return float(trade.tp2)
        if status_value == "sl_hit":
            return float(trade.current_stop)
        return None

    @staticmethod
    def _trade_state_from_row(row: dict[str, Any]) -> TradeState | None:
        try:
            trade_id = str(row.get("trade_id", "")).strip()
            signal_id = str(row.get("signal_id", "")).strip()
            instrument = str(row.get("instrument", "")).strip()
            strategy = str(row.get("strategy", "")).strip()
            timeframe = str(row.get("timeframe", "")).strip() or "1min"
            direction = SignalDirection(str(row.get("direction", "LONG")).strip().upper())
            status = TradeSimulator._status_from_value(str(row.get("status", "expired")).strip())
            created_at = _parse_dt(row.get("created_at"))
            updated_at = _parse_dt(row.get("updated_at"))
            activated_at = _parse_dt_nullable(row.get("activated_at"))
            closed_at = _parse_dt_nullable(row.get("closed_at"))
            metadata = row.get("metadata_json", {})
            if not isinstance(metadata, dict):
                metadata = {}

            if not trade_id or not signal_id or not instrument or not strategy:
                return None

            return TradeState(
                trade_id=trade_id,
                signal_id=signal_id,
                instrument=instrument,
                strategy=strategy,
                timeframe=timeframe,
                direction=direction,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                activated_at=activated_at,
                closed_at=closed_at,
                entry=float(row.get("entry", 0.0)),
                stop_loss=float(row.get("stop_loss", 0.0)),
                tp1=float(row.get("tp1", 0.0)),
                tp2=float(row.get("tp2", 0.0)),
                tp1_size=float(row.get("tp1_size", 0.5)),
                quantity=float(row.get("quantity", 1.0)),
                remaining_qty=float(row.get("remaining_qty", 1.0)),
                entry_fill_price=_as_float_nullable(row.get("entry_fill_price")),
                current_stop=float(row.get("current_stop", row.get("stop_loss", 0.0))),
                tp1_hit_at=_parse_dt_nullable(row.get("tp1_hit_at")),
                tp2_hit_at=_parse_dt_nullable(row.get("tp2_hit_at")),
                bars_waiting=int(row.get("bars_waiting", 0)),
                bars_in_trade=int(row.get("bars_in_trade", 0)),
                max_wait_bars=int(row.get("max_wait_bars", 0)),
                max_trade_bars=int(row.get("max_trade_bars", 0)),
                gross_pnl=float(row.get("gross_pnl", 0.0)),
                fees_paid=float(row.get("fees_paid", 0.0)),
                net_pnl=float(row.get("net_pnl", 0.0)),
                r_multiple=float(row.get("r_multiple", 0.0)),
                exit_reason=_str_nullable(row.get("exit_reason")),
                metadata=dict(metadata),
            )
        except Exception:
            return None


def _parse_slippage_config(*, params: dict[str, Any], sim_cfg: dict[str, Any]) -> SlippageModelConfig:
    execution_cfg = params.get("execution", {}) if isinstance(params.get("execution", {}), dict) else {}
    slippage_cfg = execution_cfg.get("slippage", {})
    if not isinstance(slippage_cfg, dict):
        slippage_cfg = {}
    legacy_cfg = sim_cfg.get("slippage", {})
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}
    merged = dict(legacy_cfg) | dict(slippage_cfg)

    default_cfg = merged.get("default", {})
    if not isinstance(default_cfg, dict):
        default_cfg = {}
    by_instrument_raw = merged.get("by_instrument", {})
    by_instrument: dict[str, dict[str, float]] = {}
    if isinstance(by_instrument_raw, dict):
        for symbol, row in by_instrument_raw.items():
            if not isinstance(row, dict):
                continue
            normalized = {
                "entry_ticks": max(0.0, _as_float(row.get("entry_ticks"), 0.0)),
                "stop_exit_ticks": max(0.0, _as_float(row.get("stop_exit_ticks"), 0.0)),
                "target_exit_ticks": max(0.0, _as_float(row.get("target_exit_ticks"), 0.0)),
                "forced_exit_ticks": max(0.0, _as_float(row.get("forced_exit_ticks"), 0.0)),
            }
            by_instrument[str(symbol).strip()] = normalized

    return SlippageModelConfig(
        enabled=_as_bool(merged.get("enabled", False), default=False),
        model=str(merged.get("model", "fixed_ticks")).strip().lower() or "fixed_ticks",
        entry_ticks=max(0.0, _as_float(default_cfg.get("entry_ticks", merged.get("entry_ticks", 0.0)), 0.0)),
        stop_exit_ticks=max(
            0.0,
            _as_float(default_cfg.get("stop_exit_ticks", merged.get("stop_exit_ticks", 0.0)), 0.0),
        ),
        target_exit_ticks=max(
            0.0,
            _as_float(default_cfg.get("target_exit_ticks", merged.get("target_exit_ticks", 0.0)), 0.0),
        ),
        forced_exit_ticks=max(
            0.0,
            _as_float(default_cfg.get("forced_exit_ticks", merged.get("forced_exit_ticks", 0.0)), 0.0),
        ),
        by_instrument=by_instrument,
    )


def _parse_fill_model(*, execution_cfg: dict[str, Any], sim_cfg: dict[str, Any]) -> str:
    raw = execution_cfg.get("fill_model")
    if raw is None:
        raw = sim_cfg.get("fill_model", "signal_entry_mode")
    normalized = str(raw).strip().lower()
    if normalized in {"", "legacy", "signal_entry_mode"}:
        return "signal_entry_mode"
    if normalized in {"conservative_next_bar_open", "next_bar_open"}:
        return "conservative_next_bar_open"
    return "signal_entry_mode"


def _parse_intrabar_conflict_policy(*, execution_cfg: dict[str, Any], sim_cfg: dict[str, Any]) -> bool:
    raw = execution_cfg.get("intrabar_conflict_policy")
    if raw is None:
        return to_bool(sim_cfg.get("intrabar_stop_priority", True), default=True)
    normalized = str(raw).strip().lower()
    if normalized in {"pessimistic_stop_priority", "stop_priority", "conservative"}:
        return True
    if normalized in {"optimistic_target_priority", "target_priority"}:
        return False
    return to_bool(sim_cfg.get("intrabar_stop_priority", True), default=True)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, *, default: bool) -> bool:
    return to_bool(value, default=default)


def _extract_position_qty(metadata: dict[str, Any]) -> float:
    for raw in (metadata.get("position_qty"), metadata.get("qty")):
        try:
            qty = float(raw)
        except (TypeError, ValueError):
            continue
        if qty > 0.0:
            return qty
    return 1.0


def _as_float_nullable(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_nullable(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("empty datetime")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_dt_nullable(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _parse_dt(text)
