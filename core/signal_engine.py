"""Signal orchestration: indicators -> regime -> strategies -> filter."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.indicator_engine import IndicatorConfig, IndicatorEngine
from core.instrument_registry import InstrumentRegistry
from core.models import MarketRegime, StrategyContext, StrategySignal
from core.news_filter import NewsBlackoutFilter
from core.regime_classifier import MarketRegimeClassifier
from core.session_manager import SessionManager
from core.signal_filter import SignalFilterPipeline
from storage.memory_store import MemoryCandleStore
from strategies.compression_breakout import CompressionBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from strategies.pullback_vwap_ema import TrendPullbackVWAPEMAStrategy


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Result of processing one candle update."""

    regime: MarketRegime | None
    accepted_signals: tuple[StrategySignal, ...]
    rejected_reasons: tuple[str, ...]


class SignalEngine:
    """Runs strategy pipeline for each candle update."""

    def __init__(
        self,
        *,
        registry: InstrumentRegistry,
        store: MemoryCandleStore,
        params: dict[str, Any],
        blackout_filter: NewsBlackoutFilter,
        logger: logging.Logger,
    ):
        self._registry = registry
        self._store = store
        self._params = params
        self._logger = logger
        self._session_manager = SessionManager()
        indicator_cfg = _build_indicator_config(params)
        self._indicator_engine = IndicatorEngine(config=indicator_cfg)
        self._regime_classifier = MarketRegimeClassifier.from_params(params)
        self._signal_filter = SignalFilterPipeline(params=params)
        self._blackout_filter = blackout_filter
        self._strategies = self._build_strategies(params)
        self._max_eval_candles = int(params.get("max_eval_candles", 350))
        signal_engine_cfg = params.get("signal_engine", {})
        if not isinstance(signal_engine_cfg, dict):
            signal_engine_cfg = {}
        self._dedupe_history_limit = max(
            1000,
            int(signal_engine_cfg.get("dedupe_history_limit", 20_000)),
        )
        self._accepted_keys_set: set[tuple[str, str, str]] = set()
        self._accepted_keys_queue: deque[tuple[str, str, str]] = deque()

    def process_candle(self, *, instrument: str, timeframe: str) -> EngineResult:
        if instrument not in self._registry:
            return EngineResult(None, tuple(), ("unknown_instrument",))

        instrument_meta = self._registry.get(instrument)
        candles = self._store.get_recent(instrument, timeframe, limit=self._max_eval_candles)
        if len(candles) < 18:
            return EngineResult(None, tuple(), tuple())

        snapshot = self._indicator_engine.snapshot(
            candles,
            session_timezone=self._session_manager.primary_timezone(instrument_meta),
        )
        if snapshot is None:
            return EngineResult(None, tuple(), tuple())

        regime = self._regime_classifier.classify(snapshot)
        session_state = self._session_manager.get_state(instrument_meta, candles[-1].datetime)
        blackout, blackout_reason = self._blackout_filter.is_blocked(candles[-1].datetime)

        context = StrategyContext(
            instrument=instrument_meta,
            timeframe=timeframe,
            candles=candles,
            indicators=snapshot,
            regime=regime,
            session_active=session_state.is_active,
            blackout_active=blackout,
            blackout_reason=blackout_reason,
            params=self._params,
        )

        accepted: list[StrategySignal] = []
        rejected_reasons: list[str] = []

        for strategy_name in instrument_meta.allowed_strategies:
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue

            raw = strategy.evaluate(context)
            if raw is None:
                continue

            decision = self._signal_filter.evaluate(raw, context)
            if not decision.accepted:
                rejected_reasons.append(f"{strategy_name}:{decision.reason}")
                continue

            dedupe_key = (
                raw.instrument,
                raw.strategy,
                raw.timestamp.isoformat(),
            )
            if dedupe_key in self._accepted_keys_set:
                rejected_reasons.append(f"{strategy_name}:duplicate")
                continue

            self._accepted_keys_set.add(dedupe_key)
            self._accepted_keys_queue.append(dedupe_key)
            if len(self._accepted_keys_queue) > self._dedupe_history_limit:
                oldest = self._accepted_keys_queue.popleft()
                self._accepted_keys_set.discard(oldest)
            accepted.append(raw)

        return EngineResult(
            regime=regime,
            accepted_signals=tuple(accepted),
            rejected_reasons=tuple(rejected_reasons),
        )

    @staticmethod
    def _build_strategies(params: dict[str, Any]) -> dict[str, Any]:
        section = params.get("strategy_params", {})
        if not isinstance(section, dict):
            section = {}

        return {
            "trend_pullback_vwap_ema": TrendPullbackVWAPEMAStrategy(
                params=_strategy_params(section, "trend_pullback_vwap_ema")
            ),
            "compression_breakout": CompressionBreakoutStrategy(
                params=_strategy_params(section, "compression_breakout")
            ),
            "liquidity_sweep_reversal": LiquiditySweepReversalStrategy(
                params=_strategy_params(section, "liquidity_sweep_reversal")
            ),
        }


def _strategy_params(section: dict[str, Any], key: str) -> dict[str, Any]:
    value = section.get(key, {})
    if isinstance(value, dict):
        return value
    return {}


def _build_indicator_config(params: dict[str, Any]) -> IndicatorConfig:
    section = params.get("indicator_engine", {})
    if not isinstance(section, dict):
        section = {}

    return IndicatorConfig(
        ema_fast=int(section.get("ema_fast", 20)),
        ema_slow=int(section.get("ema_slow", 50)),
        atr_period=int(section.get("atr_period", 14)),
        volume_period=int(section.get("volume_period", 20)),
        slope_period=int(section.get("slope_period", 5)),
        crossing_lookback=int(section.get("crossing_lookback", 30)),
        overlap_window=int(section.get("overlap_window", 12)),
        swing_window=int(section.get("swing_window", 5)),
    )
