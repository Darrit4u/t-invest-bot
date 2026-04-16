"""Signal processing layer: filtering, enrichment and deduplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Sequence

from core.models import SignalDecision, StrategyContext, StrategySignal
from core.signal_filter import SignalFilterPipeline


@dataclass(frozen=True, slots=True)
class SignalProcessResult:
    """Processed strategy output."""

    accepted_signals: tuple[StrategySignal, ...]
    rejected_reasons: tuple[str, ...]


class SignalProcessor:
    """Transforms raw strategy outputs into accepted executable signals."""

    def __init__(self, *, params: dict[str, Any]):
        self._signal_filter = SignalFilterPipeline(params=params)

        signal_engine_cfg = params.get("signal_engine", {})
        if not isinstance(signal_engine_cfg, dict):
            signal_engine_cfg = {}
        self._dedupe_history_limit = max(
            1000,
            int(signal_engine_cfg.get("dedupe_history_limit", 20_000)),
        )
        self._accepted_keys_set: set[tuple[str, str, str]] = set()
        self._accepted_keys_queue: deque[tuple[str, str, str]] = deque()

    def process_strategy_output(
        self,
        *,
        strategy_name: str,
        signals: Sequence[StrategySignal],
        context: StrategyContext,
    ) -> SignalProcessResult:
        accepted: list[StrategySignal] = []
        rejected_reasons: list[str] = []

        for raw in signals:
            decision: SignalDecision = self._signal_filter.evaluate(raw, context)
            if not decision.accepted:
                rejected_reasons.append(f"{strategy_name}:{decision.reason}")
                continue

            accepted_signal = raw
            if decision.enriched_metadata:
                accepted_signal = replace(
                    raw,
                    metadata=dict(raw.metadata) | dict(decision.enriched_metadata),
                )

            dedupe_key = (
                accepted_signal.instrument,
                accepted_signal.strategy,
                accepted_signal.timestamp.isoformat(),
            )
            if dedupe_key in self._accepted_keys_set:
                rejected_reasons.append(f"{strategy_name}:duplicate")
                continue

            self._accepted_keys_set.add(dedupe_key)
            self._accepted_keys_queue.append(dedupe_key)
            if len(self._accepted_keys_queue) > self._dedupe_history_limit:
                oldest = self._accepted_keys_queue.popleft()
                self._accepted_keys_set.discard(oldest)
            accepted.append(accepted_signal)

        return SignalProcessResult(
            accepted_signals=tuple(accepted),
            rejected_reasons=tuple(rejected_reasons),
        )
