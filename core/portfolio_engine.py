"""Portfolio orchestration layer above strategy/signal/execution engines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from core.execution_engine import ExecutionEngine
from core.models import Position, StrategySignal
from core.portfolio_events import (
    PortfolioEvent,
    allocation_rejected_event,
    normalize_trade_event,
    risk_rejected_event,
    signal_accepted_event,
    signal_rejected_event,
)
from core.portfolio_risk_manager import PortfolioRiskManager, _index_correlation_groups

_ALLOCATION_REASONS = {
    "max_positions_limit",
    "max_positions_per_instrument",
    "max_positions_per_strategy",
    "instrument_position_exists",
    "low_rr",
    "lost_in_priority",
}

_DEFAULT_STRATEGY_PRIORITY = {
    "trend_pullback_vwap_ema": 3.0,
    "compression_breakout": 2.0,
    "liquidity_sweep_reversal": 1.0,
}


@dataclass(frozen=True, slots=True)
class PortfolioSignalRejection:
    """Signal rejected by portfolio controls."""

    signal: StrategySignal
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PortfolioSelectionResult:
    """Outcome of portfolio-level signal selection/allocation."""

    accepted_signals: tuple[StrategySignal, ...]
    rejected: tuple[PortfolioSignalRejection, ...]
    events: tuple[PortfolioEvent, ...]


@dataclass(frozen=True, slots=True)
class PortfolioExecutionResult:
    """Outcome of sending selected signals to execution."""

    execution_events: tuple[Any, ...]
    opened_positions: tuple[Position, ...]
    portfolio_events: tuple[PortfolioEvent, ...]


class PortfolioEngine:
    """Collects, selects and routes strategy signals on portfolio level."""

    def __init__(self, *, params: dict[str, Any]):
        self._params = params
        self._risk_manager = PortfolioRiskManager.from_params(params)
        portfolio_cfg = params.get("portfolio", {}) if isinstance(params.get("portfolio", {}), dict) else {}
        self._min_selection_rr = max(0.0, float(portfolio_cfg.get("min_selection_rr", 1.5)))
        raw_priority = portfolio_cfg.get("strategy_priority", {})
        self._strategy_priority = dict(_DEFAULT_STRATEGY_PRIORITY)
        if isinstance(raw_priority, dict):
            for name, value in raw_priority.items():
                normalized = str(name).strip()
                if not normalized:
                    continue
                try:
                    self._strategy_priority[normalized] = float(value)
                except (TypeError, ValueError):
                    continue
        self._instrument_to_groups = _index_correlation_groups(self._risk_manager.config.correlation_groups)

    @property
    def enabled(self) -> bool:
        return self._risk_manager.enabled

    @property
    def risk_manager(self) -> PortfolioRiskManager:
        return self._risk_manager

    def collect_signals(self, *signal_batches: Iterable[StrategySignal]) -> tuple[StrategySignal, ...]:
        out: list[StrategySignal] = []
        for batch in signal_batches:
            out.extend(batch)
        return tuple(out)

    def select_signals(
        self,
        *,
        signals: Sequence[StrategySignal],
        open_positions: Sequence[Position],
    ) -> PortfolioSelectionResult:
        if not signals:
            return PortfolioSelectionResult(accepted_signals=tuple(), rejected=tuple(), events=tuple())

        dedupe_seen: set[tuple[str, str, str]] = set()
        pending: list[StrategySignal] = []
        pre_rejected: list[PortfolioSignalRejection] = []
        pre_events: list[PortfolioEvent] = []
        for signal in signals:
            dedupe_key = (signal.instrument, signal.strategy, signal.timestamp.isoformat())
            if dedupe_key in dedupe_seen:
                rejection = PortfolioSignalRejection(
                    signal=signal,
                    reason="portfolio_duplicate_signal",
                    details={"rejection_category": "duplicate"},
                )
                pre_rejected.append(rejection)
                pre_events.append(signal_rejected_event(signal=signal, reason=rejection.reason, payload=dict(rejection.details)))
                continue
            dedupe_seen.add(dedupe_key)
            rr_value = _signal_rr(signal)
            if rr_value < self._min_selection_rr:
                rejection = PortfolioSignalRejection(
                    signal=signal,
                    reason="low_rr",
                    details={
                        "rejection_category": "low_rr",
                        "signal_rr": rr_value,
                        "min_selection_rr": self._min_selection_rr,
                    },
                )
                pre_rejected.append(rejection)
                pre_events.append(self._rejection_event(signal=signal, rejection=rejection))
                continue
            pending.append(signal)

        accepted: list[StrategySignal] = []
        rejected: list[PortfolioSignalRejection] = list(pre_rejected)
        events: list[PortfolioEvent] = list(pre_events)

        while pending:
            ranked = sorted(
                pending,
                key=lambda item: self._selection_priority(
                    signal=item,
                    open_positions=open_positions,
                    pending_signals=accepted,
                ),
                reverse=True,
            )
            signal = ranked[0]
            pending.remove(signal)

            validation = self._risk_manager.validate_signal(
                signal=signal,
                open_positions=open_positions,
                pending_signals=accepted,
            )
            if not validation.accepted:
                rejection_reason, rejection_details = self._normalize_rejection(
                    signal=signal,
                    validation_reason=validation.reason,
                    validation_details=validation.details,
                    open_positions=open_positions,
                    accepted_signals=accepted,
                    all_ranked=ranked,
                )
                rejection = PortfolioSignalRejection(
                    signal=signal,
                    reason=rejection_reason,
                    details=rejection_details,
                )
                rejected.append(rejection)
                events.append(self._rejection_event(signal=signal, rejection=rejection))
                continue

            accepted_signal = self.allocate_capital(
                signal=signal,
                signal_risk_pct=validation.signal_risk_pct,
                details=dict(validation.details)
                | {
                    "selection_rr": _signal_rr(signal),
                    "selection_strategy_priority": self._strategy_priority_score(signal.strategy),
                    "selection_diversification_score": self._diversification_score(
                        signal=signal,
                        open_positions=open_positions,
                        pending_signals=accepted,
                    ),
                    "selection_score": self._selection_priority(
                        signal=signal,
                        open_positions=open_positions,
                        pending_signals=accepted,
                    ),
                },
            )
            accepted.append(accepted_signal)
            events.append(
                signal_accepted_event(
                    signal=accepted_signal,
                    payload={
                        "signal_risk_pct": validation.signal_risk_pct,
                        "signal_risk_money": validation.details.get("signal_risk_money", 0.0),
                        "position_qty": validation.details.get("position_qty", 0.0),
                        "planned_risk_money": validation.details.get("planned_risk_money", 0.0),
                        "planned_risk_pct": validation.details.get("planned_risk_pct", 0.0),
                        "projected_total_risk_pct": validation.details.get("projected_total_risk_pct", 0.0),
                        "projected_total_risk_money": validation.details.get("projected_total_risk_money", 0.0),
                        "selection_rr": _signal_rr(signal),
                        "selection_strategy_priority": self._strategy_priority_score(signal.strategy),
                        "selection_diversification_score": self._diversification_score(
                            signal=signal,
                            open_positions=open_positions,
                            pending_signals=accepted[:-1],
                        ),
                    },
                )
            )

        return PortfolioSelectionResult(
            accepted_signals=tuple(accepted),
            rejected=tuple(rejected),
            events=tuple(events),
        )

    @staticmethod
    def allocate_capital(
        *,
        signal: StrategySignal,
        signal_risk_pct: float,
        details: dict[str, Any],
    ) -> StrategySignal:
        metadata = dict(signal.metadata) | {
            "position_qty": details.get("position_qty"),
            "qty": details.get("position_qty"),
            "planned_risk_money": details.get("planned_risk_money"),
            "planned_risk_pct": details.get("planned_risk_pct"),
            "signal_risk_money": details.get("signal_risk_money"),
            "portfolio_risk_pct": signal_risk_pct,
            "portfolio_projected_total_risk_pct": details.get("projected_total_risk_pct"),
            "portfolio_projected_total_risk_money": details.get("projected_total_risk_money"),
            "portfolio_projected_instrument_risk_pct": details.get("projected_instrument_risk_pct"),
            "portfolio_projected_instrument_risk_money": details.get("projected_instrument_risk_money"),
            "portfolio_projected_strategy_risk_pct": details.get("projected_strategy_risk_pct"),
            "portfolio_projected_strategy_risk_money": details.get("projected_strategy_risk_money"),
            "selection_rr": details.get("selection_rr"),
            "selection_strategy_priority": details.get("selection_strategy_priority"),
            "selection_diversification_score": details.get("selection_diversification_score"),
            "selection_score": details.get("selection_score"),
        }
        return replace(signal, metadata=metadata)

    def submit_for_execution(
        self,
        *,
        signals: Sequence[StrategySignal],
        execution_engine: ExecutionEngine,
        timeframe: str,
    ) -> PortfolioExecutionResult:
        execution_events: list[Any] = []
        opened_positions: list[Position] = []
        portfolio_events: list[PortfolioEvent] = []

        for signal in signals:
            open_result = execution_engine.open_from_signal(signal=signal, timeframe=timeframe)
            execution_events.extend(open_result.events)
            if open_result.position is not None:
                opened_positions.append(open_result.position)

            for event in open_result.events:
                trade = execution_engine.get_trade_record(event.trade_id)
                portfolio_events.extend(normalize_trade_event(event, trade=trade))

        return PortfolioExecutionResult(
            execution_events=tuple(execution_events),
            opened_positions=tuple(opened_positions),
            portfolio_events=tuple(portfolio_events),
        )

    def normalize_execution_events(
        self,
        *,
        execution_events: Sequence[Any],
        execution_engine: ExecutionEngine,
    ) -> tuple[PortfolioEvent, ...]:
        out: list[PortfolioEvent] = []
        for event in execution_events:
            trade = execution_engine.get_trade_record(event.trade_id)
            out.extend(normalize_trade_event(event, trade=trade))
        return tuple(out)

    @staticmethod
    def _rejection_event(
        *,
        signal: StrategySignal,
        rejection: PortfolioSignalRejection,
    ) -> PortfolioEvent:
        payload = dict(rejection.details)
        if rejection.reason in _ALLOCATION_REASONS:
            return allocation_rejected_event(signal=signal, reason=rejection.reason, payload=payload)
        return risk_rejected_event(signal=signal, reason=rejection.reason, payload=payload)

    def _selection_priority(
        self,
        *,
        signal: StrategySignal,
        open_positions: Sequence[Position],
        pending_signals: Sequence[StrategySignal],
    ) -> float:
        strategy_component = self._strategy_priority_score(signal.strategy) * 100.0
        diversification_component = self._diversification_score(
            signal=signal,
            open_positions=open_positions,
            pending_signals=pending_signals,
        ) * 10.0
        rr_component = _signal_rr(signal) * 5.0
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        quality_component = _as_float(metadata.get("signal_quality_score"), default=0.0)
        edge_component = _as_float(metadata.get("expected_edge_after_fees"), default=0.0)
        return strategy_component + diversification_component + rr_component + quality_component + edge_component

    def _strategy_priority_score(self, strategy_name: str) -> float:
        return float(self._strategy_priority.get(strategy_name, 0.0))

    def _diversification_score(
        self,
        *,
        signal: StrategySignal,
        open_positions: Sequence[Position],
        pending_signals: Sequence[StrategySignal],
    ) -> float:
        exposure = self._risk_manager.current_exposure(
            open_positions=open_positions,
            pending_signals=pending_signals,
        )
        score = 0.0
        instrument_positions = exposure.positions_by_instrument.get(signal.instrument, 0)
        if instrument_positions == 0:
            score += 1.0
        else:
            score -= min(1.0, 0.75 * instrument_positions)

        strategy_positions = exposure.positions_by_strategy.get(signal.strategy, 0)
        if strategy_positions == 0:
            score += 0.35
        else:
            score -= min(0.5, 0.25 * strategy_positions)

        groups = self._instrument_to_groups.get(signal.instrument, tuple())
        if not groups:
            score += 0.10
        for group_name in groups:
            group_positions = exposure.positions_by_group.get(group_name, 0)
            if group_positions == 0:
                score += 0.60
            else:
                score -= min(0.75, 0.50 * group_positions)
        return score

    def _normalize_rejection(
        self,
        *,
        signal: StrategySignal,
        validation_reason: str,
        validation_details: dict[str, Any],
        open_positions: Sequence[Position],
        accepted_signals: Sequence[StrategySignal],
        all_ranked: Sequence[StrategySignal],
    ) -> tuple[str, dict[str, Any]]:
        details = dict(validation_details)
        details["signal_rr"] = _signal_rr(signal)
        details["selection_strategy_priority"] = self._strategy_priority_score(signal.strategy)
        details["selection_score"] = self._selection_priority(
            signal=signal,
            open_positions=open_positions,
            pending_signals=accepted_signals,
        )
        if validation_reason == "correlation_group_limit":
            details["rejection_category"] = "correlation"
            return validation_reason, details
        if validation_reason in {
            "sizing_reject",
            "risk_per_trade_limit",
            "max_total_risk_limit",
            "instrument_risk_cap",
            "group_risk_cap",
            "strategy_risk_cap",
            "invalid_signal_risk",
        }:
            details["rejection_category"] = "risk"
            return validation_reason, details
        if validation_reason in _ALLOCATION_REASONS and accepted_signals:
            details["rejection_category"] = "priority"
            details["priority_conflict_reason"] = validation_reason
            details["higher_priority_signals"] = [
                {
                    "signal_id": item.signal_id,
                    "instrument": item.instrument,
                    "strategy": item.strategy,
                    "selection_score": self._selection_priority(
                        signal=item,
                        open_positions=open_positions,
                        pending_signals=tuple(sig for sig in accepted_signals if sig.signal_id != item.signal_id),
                    ),
                }
                for item in accepted_signals[:3]
            ]
            details["ranked_ahead"] = [
                {
                    "signal_id": item.signal_id,
                    "instrument": item.instrument,
                    "strategy": item.strategy,
                }
                for item in all_ranked[:3]
                if item.signal_id != signal.signal_id
            ]
            return "lost_in_priority", details
        if validation_reason in _ALLOCATION_REASONS:
            details["rejection_category"] = "allocation"
            return validation_reason, details
        details["rejection_category"] = "risk"
        return validation_reason, details


def _signal_rr(signal: StrategySignal) -> float:
    entry = _as_float(signal.entry, default=0.0)
    stop = _as_float(signal.stop_loss, default=0.0)
    tp1 = _as_float(signal.tp1, default=0.0)
    metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
    hinted = metadata.get("post_fill_rr")
    if hinted is not None:
        hinted_rr = _as_float(hinted, default=0.0)
        if hinted_rr > 0.0:
            return hinted_rr
    if signal.direction.value == "LONG":
        risk = entry - stop
        reward = tp1 - entry
    else:
        risk = stop - entry
        reward = entry - tp1
    if risk <= 0.0:
        return 0.0
    return reward / risk


def _signal_priority(signal: StrategySignal) -> tuple[float, float]:
    metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
    quality = _as_float(metadata.get("signal_quality_score"), default=0.0)
    edge = _as_float(metadata.get("expected_edge_after_fees"), default=0.0)
    return (quality, edge)


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
