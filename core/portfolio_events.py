"""Domain-level event contract for portfolio/runtime layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.models import StrategySignal, Trade
from core.trade_simulator import TradeEvent


class DomainEventType(str, Enum):
    """Canonical domain events emitted by portfolio/execution layers."""

    SIGNAL_ACCEPTED = "SignalAccepted"
    SIGNAL_REJECTED = "SignalRejected"
    POSITION_OPENED = "PositionOpened"
    POSITION_UPDATED = "PositionUpdated"
    POSITION_CLOSED = "PositionClosed"
    TRADE_CLOSED = "TradeClosed"
    RISK_REJECTED = "RiskRejected"
    ALLOCATION_REJECTED = "AllocationRejected"


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Normalized domain event used by stats/reporting/notifier/pipeline."""

    kind: DomainEventType
    event_time: datetime
    instrument: str | None
    strategy: str | None
    signal_id: str | None
    trade_id: str | None
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Legacy-compatible snake_case event identifier."""
        return _LEGACY_EVENT_NAME[self.kind]


PortfolioEvent = DomainEvent


def signal_accepted_event(
    *,
    signal: StrategySignal,
    payload: dict[str, Any] | None = None,
) -> DomainEvent:
    return DomainEvent(
        kind=DomainEventType.SIGNAL_ACCEPTED,
        event_time=signal.timestamp,
        instrument=signal.instrument,
        strategy=signal.strategy,
        signal_id=signal.signal_id,
        trade_id=None,
        payload=payload or {},
    )


def signal_rejected_event(
    *,
    signal: StrategySignal,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> DomainEvent:
    return DomainEvent(
        kind=DomainEventType.SIGNAL_REJECTED,
        event_time=signal.timestamp,
        instrument=signal.instrument,
        strategy=signal.strategy,
        signal_id=signal.signal_id,
        trade_id=None,
        reason=reason,
        payload=payload or {},
    )


def risk_rejected_event(
    *,
    signal: StrategySignal,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> DomainEvent:
    return DomainEvent(
        kind=DomainEventType.RISK_REJECTED,
        event_time=signal.timestamp,
        instrument=signal.instrument,
        strategy=signal.strategy,
        signal_id=signal.signal_id,
        trade_id=None,
        reason=reason,
        payload=payload or {},
    )


def allocation_rejected_event(
    *,
    signal: StrategySignal,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> DomainEvent:
    return DomainEvent(
        kind=DomainEventType.ALLOCATION_REJECTED,
        event_time=signal.timestamp,
        instrument=signal.instrument,
        strategy=signal.strategy,
        signal_id=signal.signal_id,
        trade_id=None,
        reason=reason,
        payload=payload or {},
    )


def normalize_trade_event(event: TradeEvent, *, trade: Trade | None = None) -> tuple[DomainEvent, ...]:
    """Map low-level simulator events to canonical domain events."""

    base_payload = dict(event.payload) if isinstance(event.payload, dict) else {}
    base_payload["source_trade_event"] = event.event_type
    if trade is not None:
        base_payload = base_payload | {
            "trade_status": trade.status or "",
            "side": trade.side.value,
            "trade_pnl": float(trade.pnl),
            "qty": float(trade.size),
            "fees_paid": float(trade.fees_paid or 0.0),
            "net_pnl": float(trade.pnl),
            "r_multiple": float(trade.r_multiple or 0.0),
            "entry_price": float(trade.entry_price),
            "entry_fill_price": float(trade.entry_fill_price or trade.entry_price),
            "stop_loss": _safe_float((trade.metadata or {}).get("stop_loss")),
            "take_profit": _safe_float((trade.metadata or {}).get("tp2")),
            "planned_risk_money": _safe_float((trade.metadata or {}).get("planned_risk_money")),
            "planned_risk_pct": _safe_float((trade.metadata or {}).get("planned_risk_pct")),
            "expected_rr": _safe_float((trade.metadata or {}).get("post_fill_rr")),
        }

    if event.event_type == "activated":
        return (
            DomainEvent(
                kind=DomainEventType.POSITION_OPENED,
                event_time=event.event_time,
                instrument=event.instrument,
                strategy=event.strategy,
                signal_id=event.signal_id,
                trade_id=event.trade_id,
                payload=base_payload,
            ),
        )

    if event.event_type in {"new_signal", "tp1_hit"}:
        return (
            DomainEvent(
                kind=DomainEventType.POSITION_UPDATED,
                event_time=event.event_time,
                instrument=event.instrument,
                strategy=event.strategy,
                signal_id=event.signal_id,
                trade_id=event.trade_id,
                payload=base_payload,
            ),
        )

    if event.event_type in {"tp2_hit", "sl_hit", "expired", "cancelled_by_news", "cancelled_by_session_end"}:
        close_reason = str(base_payload.get("reason", "")).strip() or event.event_type
        closed_event = DomainEvent(
            kind=DomainEventType.POSITION_CLOSED,
            event_time=event.event_time,
            instrument=event.instrument,
            strategy=event.strategy,
            signal_id=event.signal_id,
            trade_id=event.trade_id,
            reason=close_reason,
            payload=base_payload,
        )
        trade_closed = DomainEvent(
            kind=DomainEventType.TRADE_CLOSED,
            event_time=event.event_time,
            instrument=event.instrument,
            strategy=event.strategy,
            signal_id=event.signal_id,
            trade_id=event.trade_id,
            reason=close_reason,
            payload=base_payload,
        )
        return (closed_event, trade_closed)

    return (
        DomainEvent(
            kind=DomainEventType.POSITION_UPDATED,
            event_time=event.event_time,
            instrument=event.instrument,
            strategy=event.strategy,
            signal_id=event.signal_id,
            trade_id=event.trade_id,
            payload=base_payload,
        ),
    )


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


_LEGACY_EVENT_NAME = {
    DomainEventType.SIGNAL_ACCEPTED: "signal_accepted",
    DomainEventType.SIGNAL_REJECTED: "signal_rejected",
    DomainEventType.POSITION_OPENED: "position_opened",
    DomainEventType.POSITION_UPDATED: "position_updated",
    DomainEventType.POSITION_CLOSED: "position_closed",
    DomainEventType.TRADE_CLOSED: "trade_closed",
    DomainEventType.RISK_REJECTED: "risk_rejected",
    DomainEventType.ALLOCATION_REJECTED: "allocation_rejected",
}
