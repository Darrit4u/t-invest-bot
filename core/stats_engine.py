"""Aggregated runtime statistics from simulated trades."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from core.trade_simulator import SimulatedTrade, TradeEvent, TradeStatus


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


class StatsEngine:
    """Tracks global/instrument/strategy metrics over runtime."""

    def __init__(self) -> None:
        self._global = MetricBucket()
        self._by_instrument: dict[str, MetricBucket] = defaultdict(MetricBucket)
        self._by_strategy: dict[str, MetricBucket] = defaultdict(MetricBucket)
        self._peak_equity = 0.0
        self._equity = 0.0
        self._max_drawdown = 0.0

    def record_signal(self, *, instrument: str, strategy: str) -> None:
        self._global.signals += 1
        self._by_instrument[instrument].signals += 1
        self._by_strategy[strategy].signals += 1

    def record_event(self, event: TradeEvent) -> None:
        if event.event_type == "activated":
            self._global.activated += 1
            self._by_instrument[event.instrument].activated += 1
            self._by_strategy[event.strategy].activated += 1
        elif event.event_type == "tp1_hit":
            self._global.tp1_hits += 1
            self._by_instrument[event.instrument].tp1_hits += 1
            self._by_strategy[event.strategy].tp1_hits += 1

    def record_trade_closed(self, trade: SimulatedTrade) -> None:
        self._apply_closed_trade(self._global, trade)
        self._apply_closed_trade(self._by_instrument[trade.instrument], trade)
        self._apply_closed_trade(self._by_strategy[trade.strategy], trade)

        self._equity += trade.net_pnl
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = self._peak_equity - self._equity
        self._max_drawdown = max(self._max_drawdown, drawdown)

    def summary(self) -> dict[str, Any]:
        return {
            "global": self._global.to_dict() | {"max_drawdown": self._max_drawdown},
            "by_instrument": {key: bucket.to_dict() for key, bucket in sorted(self._by_instrument.items())},
            "by_strategy": {key: bucket.to_dict() for key, bucket in sorted(self._by_strategy.items())},
        }

    @staticmethod
    def _apply_closed_trade(bucket: MetricBucket, trade: SimulatedTrade) -> None:
        bucket.closed += 1
        bucket.gross_pnl += trade.gross_pnl
        bucket.net_pnl += trade.net_pnl
        bucket.fees += trade.fees_paid
        bucket.sum_r += trade.r_multiple

        if trade.net_pnl >= 0:
            bucket.wins += 1
            bucket.gross_wins += trade.net_pnl
        else:
            bucket.losses += 1
            bucket.gross_losses_abs += abs(trade.net_pnl)

        if trade.status == TradeStatus.TP2_HIT:
            bucket.tp2_hits += 1
        elif trade.status == TradeStatus.SL_HIT:
            bucket.sl_hits += 1
        elif trade.status == TradeStatus.EXPIRED:
            bucket.expired += 1
        elif trade.status == TradeStatus.CANCELLED_BY_NEWS:
            bucket.cancelled_news += 1
        elif trade.status == TradeStatus.CANCELLED_BY_SESSION_END:
            bucket.cancelled_session += 1
