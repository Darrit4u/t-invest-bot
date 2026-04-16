"""Aggregated runtime statistics from simulated trades."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.models import Trade
from core.portfolio_events import DomainEventType, PortfolioEvent


@dataclass(slots=True)
class MetricBucket:
    """PnL and outcome metrics for one dimension (instrument/strategy/global)."""

    signals: int = 0
    activated: int = 0
    closed: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    sl_hits: int = 0
    expired: int = 0
    cancelled_news: int = 0
    cancelled_session: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    sum_r: float = 0.0
    gross_wins: float = 0.0
    gross_losses_abs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        expectancy = self.net_pnl / self.closed if self.closed else 0.0
        avg_r = self.sum_r / self.closed if self.closed else 0.0
        profit_factor = self.gross_wins / self.gross_losses_abs if self.gross_losses_abs > 0 else 0.0
        win_rate = self.wins / self.closed if self.closed else 0.0
        return {
            "signals": self.signals,
            "activated": self.activated,
            "closed": self.closed,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "sl_hits": self.sl_hits,
            "expired": self.expired,
            "cancelled_news": self.cancelled_news,
            "cancelled_session": self.cancelled_session,
            "wins": self.wins,
            "losses": self.losses,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "fees": self.fees,
            "expectancy": expectancy,
            "avg_r": avg_r,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
        }


@dataclass(slots=True)
class PortfolioMetricBucket:
    """High-level portfolio routing and lifecycle counters."""

    signal_accepted: int = 0
    signal_rejected: int = 0
    risk_rejected: int = 0
    allocation_rejected: int = 0
    position_opened: int = 0
    position_updated: int = 0
    position_closed: int = 0
    trade_closed: int = 0
    accepted_qty_sum: float = 0.0
    accepted_planned_risk_money_sum: float = 0.0
    accepted_planned_risk_pct_sum: float = 0.0
    closed_r_multiple_sum: float = 0.0
    risk_reject_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_accepted": self.signal_accepted,
            "signal_rejected": self.signal_rejected,
            "risk_rejected": self.risk_rejected,
            "allocation_rejected": self.allocation_rejected,
            "position_opened": self.position_opened,
            "position_updated": self.position_updated,
            "position_closed": self.position_closed,
            "trade_closed": self.trade_closed,
            "accepted_qty_sum": self.accepted_qty_sum,
            "accepted_planned_risk_money_sum": self.accepted_planned_risk_money_sum,
            "accepted_planned_risk_pct_sum": self.accepted_planned_risk_pct_sum,
            "closed_r_multiple_sum": self.closed_r_multiple_sum,
            "avg_planned_risk_money": (
                self.accepted_planned_risk_money_sum / self.signal_accepted if self.signal_accepted else 0.0
            ),
            "avg_planned_risk_pct": (
                self.accepted_planned_risk_pct_sum / self.signal_accepted if self.signal_accepted else 0.0
            ),
            "avg_qty": self.accepted_qty_sum / self.signal_accepted if self.signal_accepted else 0.0,
            "avg_closed_r_multiple": (
                self.closed_r_multiple_sum / self.trade_closed if self.trade_closed else 0.0
            ),
            "risk_reject_reasons": dict(sorted(self.risk_reject_reasons.items(), key=lambda row: row[0])),
        }


class StatsEngine:
    """Tracks global/instrument/strategy metrics over runtime."""

    def __init__(self) -> None:
        self._global = MetricBucket()
        self._by_instrument: dict[str, MetricBucket] = defaultdict(MetricBucket)
        self._by_strategy: dict[str, MetricBucket] = defaultdict(MetricBucket)
        self._portfolio = PortfolioMetricBucket()
        self._peak_equity = 0.0
        self._equity = 0.0
        self._max_drawdown = 0.0

    def record_signal(self, *, instrument: str, strategy: str) -> None:
        self._global.signals += 1
        self._by_instrument[instrument].signals += 1
        self._by_strategy[strategy].signals += 1

    def record_event(self, event: Any) -> None:
        """Legacy adapter for low-level simulator events.

        Preferred contract for upper layers is `record_portfolio_event(...)`.
        """
        if event.event_type == "activated":
            self._global.activated += 1
            self._by_instrument[event.instrument].activated += 1
            self._by_strategy[event.strategy].activated += 1
        elif event.event_type == "tp1_hit":
            self._global.tp1_hits += 1
            self._by_instrument[event.instrument].tp1_hits += 1
            self._by_strategy[event.strategy].tp1_hits += 1

    def record_trade_closed(self, trade: Trade) -> None:
        self._apply_closed_trade(self._global, trade)
        self._apply_closed_trade(self._by_instrument[trade.instrument], trade)
        self._apply_closed_trade(self._by_strategy[trade.strategy_id], trade)

        self._equity += float(trade.pnl)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = self._peak_equity - self._equity
        self._max_drawdown = max(self._max_drawdown, drawdown)

    def summary(self) -> dict[str, Any]:
        return {
            "global": self._global.to_dict() | {"max_drawdown": self._max_drawdown},
            "by_instrument": {key: bucket.to_dict() for key, bucket in sorted(self._by_instrument.items())},
            "by_strategy": {key: bucket.to_dict() for key, bucket in sorted(self._by_strategy.items())},
            "portfolio": self._portfolio.to_dict(),
        }

    def record_portfolio_event(self, event: PortfolioEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.kind == DomainEventType.SIGNAL_ACCEPTED:
            self._portfolio.signal_accepted += 1
            self._portfolio.accepted_qty_sum += _to_float(payload.get("position_qty"))
            self._portfolio.accepted_planned_risk_money_sum += _to_float(payload.get("planned_risk_money"))
            self._portfolio.accepted_planned_risk_pct_sum += _to_float(payload.get("planned_risk_pct"))
        elif event.kind == DomainEventType.SIGNAL_REJECTED:
            self._portfolio.signal_rejected += 1
        elif event.kind == DomainEventType.RISK_REJECTED:
            self._portfolio.risk_rejected += 1
            reason_key = str(event.reason or "unknown").strip() or "unknown"
            self._portfolio.risk_reject_reasons[reason_key] = self._portfolio.risk_reject_reasons.get(reason_key, 0) + 1
        elif event.kind == DomainEventType.ALLOCATION_REJECTED:
            self._portfolio.allocation_rejected += 1
        elif event.kind == DomainEventType.POSITION_OPENED:
            self._portfolio.position_opened += 1
            self._global.activated += 1
            if event.instrument:
                self._by_instrument[event.instrument].activated += 1
            if event.strategy:
                self._by_strategy[event.strategy].activated += 1
        elif event.kind == DomainEventType.POSITION_UPDATED:
            self._portfolio.position_updated += 1
            source = str(payload.get("source_trade_event", "")).strip()
            if source == "tp1_hit":
                self._global.tp1_hits += 1
                if event.instrument:
                    self._by_instrument[event.instrument].tp1_hits += 1
                if event.strategy:
                    self._by_strategy[event.strategy].tp1_hits += 1
        elif event.kind == DomainEventType.POSITION_CLOSED:
            self._portfolio.position_closed += 1
        elif event.kind == DomainEventType.TRADE_CLOSED:
            self._portfolio.trade_closed += 1
            self._portfolio.closed_r_multiple_sum += _to_float(payload.get("r_multiple"))

    @staticmethod
    def _apply_closed_trade(bucket: MetricBucket, trade: Trade) -> None:
        net_pnl = float(trade.pnl)
        gross_pnl = float(trade.gross_pnl) if trade.gross_pnl is not None else net_pnl
        fees_paid = float(trade.fees_paid) if trade.fees_paid is not None else max(0.0, gross_pnl - net_pnl)
        r_multiple = float(trade.r_multiple) if trade.r_multiple is not None else 0.0
        status_value = str(trade.status or "")

        bucket.closed += 1
        bucket.gross_pnl += gross_pnl
        bucket.net_pnl += net_pnl
        bucket.fees += fees_paid
        bucket.sum_r += r_multiple

        if net_pnl >= 0:
            bucket.wins += 1
            bucket.gross_wins += net_pnl
        else:
            bucket.losses += 1
            bucket.gross_losses_abs += abs(net_pnl)

        if status_value == "tp2_hit":
            bucket.tp2_hits += 1
        elif status_value == "sl_hit":
            bucket.sl_hits += 1
        elif status_value == "expired":
            bucket.expired += 1
        elif status_value == "cancelled_by_news":
            bucket.cancelled_news += 1
        elif status_value == "cancelled_by_session_end":
            bucket.cancelled_session += 1


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
