"""Portfolio-level risk controls and exposure checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core.models import Position, StrategySignal
from core.position_sizer import PositionSizer


@dataclass(frozen=True, slots=True)
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk controls."""

    enabled: bool = False
    account_size: float = 100_000.0
    max_positions: int = 1
    max_positions_per_instrument: int = 1
    max_positions_per_strategy: int = 3
    allow_multiple_positions_per_instrument: bool = False
    max_risk_per_trade_pct: float = 1.0
    max_total_risk_pct: float = 4.0
    max_instrument_risk_pct: float = 2.0
    max_group_risk_pct: float = 2.5
    max_strategy_risk_pct: float = 3.0
    max_positions_per_correlation_group: int = 1
    instrument_weights: dict[str, float] = field(default_factory=dict)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    correlation_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PortfolioExposure:
    """Current portfolio risk/exposure snapshot."""

    total_positions: int
    total_risk_money: float
    total_risk_pct: float
    risk_money_by_instrument: dict[str, float]
    risk_money_by_strategy: dict[str, float]
    risk_money_by_group: dict[str, float]
    risk_by_instrument: dict[str, float]
    risk_by_strategy: dict[str, float]
    positions_by_instrument: dict[str, int]
    positions_by_strategy: dict[str, int]
    positions_by_group: dict[str, int]


@dataclass(frozen=True, slots=True)
class RiskValidationResult:
    """Result of a signal-level risk validation."""

    accepted: bool
    reason: str
    signal_risk_pct: float
    details: dict[str, Any] = field(default_factory=dict)


class PortfolioRiskManager:
    """Enforces deterministic portfolio limits for incoming signals."""

    def __init__(self, config: PortfolioRiskConfig):
        self._config = config
        self._instrument_to_groups = _index_correlation_groups(config.correlation_groups)
        self._position_sizer = PositionSizer()

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "PortfolioRiskManager":
        section = params.get("portfolio", {})
        if not isinstance(section, dict):
            section = {}

        weights_raw = section.get("instrument_weights", {})
        if not isinstance(weights_raw, dict):
            weights_raw = {}
        instrument_weights = _normalized_weights(weights_raw)

        strategy_weights_raw = section.get("strategy_weights", {})
        if not isinstance(strategy_weights_raw, dict):
            strategy_weights_raw = {}
        strategy_weights = _normalized_weights(strategy_weights_raw)

        groups_raw = section.get("correlation_groups", {})
        correlation_groups: dict[str, tuple[str, ...]] = {}
        if isinstance(groups_raw, dict):
            for group_name, items in groups_raw.items():
                if isinstance(items, (list, tuple, set)):
                    symbols = tuple(str(item).strip() for item in items if str(item).strip())
                    if symbols:
                        correlation_groups[str(group_name).strip()] = symbols

        config = PortfolioRiskConfig(
            enabled=_to_bool(section.get("enabled", False), default=False),
            account_size=max(1.0, float(section.get("account_size", 100_000.0))),
            max_positions=max(1, int(section.get("max_positions", 1))),
            max_positions_per_instrument=max(1, int(section.get("max_positions_per_instrument", 1))),
            max_positions_per_strategy=max(1, int(section.get("max_positions_per_strategy", 3))),
            allow_multiple_positions_per_instrument=_to_bool(
                section.get("allow_multiple_positions_per_instrument", False),
                default=False,
            ),
            max_risk_per_trade_pct=max(0.0, float(section.get("max_risk_per_trade_pct", 1.0))),
            max_total_risk_pct=max(0.0, float(section.get("max_total_risk_pct", 4.0))),
            max_instrument_risk_pct=max(0.0, float(section.get("max_instrument_risk_pct", 2.0))),
            max_group_risk_pct=max(0.0, float(section.get("max_group_risk_pct", section.get("max_instrument_risk_pct", 2.0)))),
            max_strategy_risk_pct=max(0.0, float(section.get("max_strategy_risk_pct", 3.0))),
            max_positions_per_correlation_group=max(
                1,
                int(section.get("max_positions_per_correlation_group", 1)),
            ),
            instrument_weights=instrument_weights,
            strategy_weights=strategy_weights,
            correlation_groups=correlation_groups,
        )
        return cls(config=config)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> PortfolioRiskConfig:
        return self._config

    def current_exposure(
        self,
        *,
        open_positions: Sequence[Position],
        pending_signals: Sequence[StrategySignal] = (),
    ) -> PortfolioExposure:
        risk_money_by_instrument: dict[str, float] = {}
        risk_money_by_strategy: dict[str, float] = {}
        risk_money_by_group: dict[str, float] = {}
        positions_by_instrument: dict[str, int] = {}
        positions_by_strategy: dict[str, int] = {}
        positions_by_group: dict[str, int] = {}

        total_risk_money = 0.0
        total_positions = 0

        for position in open_positions:
            risk_money = self.estimate_position_risk_money(position)
            total_risk_money += risk_money
            total_positions += 1
            risk_money_by_instrument[position.instrument] = risk_money_by_instrument.get(position.instrument, 0.0) + risk_money
            risk_money_by_strategy[position.strategy_id] = risk_money_by_strategy.get(position.strategy_id, 0.0) + risk_money
            positions_by_instrument[position.instrument] = positions_by_instrument.get(position.instrument, 0) + 1
            positions_by_strategy[position.strategy_id] = positions_by_strategy.get(position.strategy_id, 0) + 1
            for group in self._groups_for_symbol(position.instrument):
                positions_by_group[group] = positions_by_group.get(group, 0) + 1
                risk_money_by_group[group] = risk_money_by_group.get(group, 0.0) + risk_money

        for signal in pending_signals:
            risk_money = self.estimate_signal_risk_money(signal)
            total_risk_money += risk_money
            total_positions += 1
            risk_money_by_instrument[signal.instrument] = risk_money_by_instrument.get(signal.instrument, 0.0) + risk_money
            risk_money_by_strategy[signal.strategy] = risk_money_by_strategy.get(signal.strategy, 0.0) + risk_money
            positions_by_instrument[signal.instrument] = positions_by_instrument.get(signal.instrument, 0) + 1
            positions_by_strategy[signal.strategy] = positions_by_strategy.get(signal.strategy, 0) + 1
            for group in self._groups_for_symbol(signal.instrument):
                positions_by_group[group] = positions_by_group.get(group, 0) + 1
                risk_money_by_group[group] = risk_money_by_group.get(group, 0.0) + risk_money

        total_risk_pct = self._money_to_pct(total_risk_money)
        risk_by_instrument = {symbol: self._money_to_pct(amount) for symbol, amount in risk_money_by_instrument.items()}
        risk_by_strategy = {name: self._money_to_pct(amount) for name, amount in risk_money_by_strategy.items()}

        return PortfolioExposure(
            total_positions=total_positions,
            total_risk_money=total_risk_money,
            total_risk_pct=total_risk_pct,
            risk_money_by_instrument=risk_money_by_instrument,
            risk_money_by_strategy=risk_money_by_strategy,
            risk_money_by_group=risk_money_by_group,
            risk_by_instrument=risk_by_instrument,
            risk_by_strategy=risk_by_strategy,
            positions_by_instrument=positions_by_instrument,
            positions_by_strategy=positions_by_strategy,
            positions_by_group=positions_by_group,
        )

    def validate_signal(
        self,
        *,
        signal: StrategySignal,
        open_positions: Sequence[Position],
        pending_signals: Sequence[StrategySignal] = (),
    ) -> RiskValidationResult:
        sizing = self._size_signal(signal)
        if sizing is None:
            return RiskValidationResult(
                accepted=False,
                reason="sizing_reject",
                signal_risk_pct=0.0,
                details={"sizing_reject_reason": "invalid_signal"},
            )
        if not sizing.accepted:
            return RiskValidationResult(
                accepted=False,
                reason="sizing_reject",
                signal_risk_pct=0.0,
                details={
                    "sizing_reject_reason": sizing.reject.reason if sizing.reject is not None else "unknown",
                    "sizing_reject_details": sizing.reject.details if sizing.reject is not None else {},
                },
            )
        assert sizing.result is not None
        signal_risk_money = sizing.result.qty * sizing.result.money_per_contract
        signal_risk_pct = self._money_to_pct(signal_risk_money)
        position_qty = float(sizing.result.qty)

        if not self._config.enabled:
            return RiskValidationResult(
                accepted=True,
                reason="portfolio_disabled",
                signal_risk_pct=signal_risk_pct,
                details={
                    "signal_risk_pct": signal_risk_pct,
                    "signal_risk_money": signal_risk_money,
                    "position_qty": position_qty,
                    "planned_risk_money": signal_risk_money,
                    "planned_risk_pct": signal_risk_pct,
                    "ticks": int(sizing.result.ticks),
                    "money_per_contract": float(sizing.result.money_per_contract),
                },
            )

        if signal_risk_money <= 0.0:
            return RiskValidationResult(
                accepted=False,
                reason="invalid_signal_risk",
                signal_risk_pct=signal_risk_pct,
                details={
                    "signal_risk_pct": signal_risk_pct,
                    "signal_risk_money": signal_risk_money,
                },
            )

        if signal_risk_pct > self._config.max_risk_per_trade_pct:
            return RiskValidationResult(
                accepted=False,
                reason="risk_per_trade_limit",
                signal_risk_pct=signal_risk_pct,
                details={
                    "signal_risk_pct": signal_risk_pct,
                    "signal_risk_money": signal_risk_money,
                    "max_risk_per_trade_pct": self._config.max_risk_per_trade_pct,
                    "max_risk_per_trade_money": self._pct_to_money(self._config.max_risk_per_trade_pct),
                },
            )

        exposure = self.current_exposure(open_positions=open_positions, pending_signals=pending_signals)

        if exposure.total_positions >= self._config.max_positions:
            return RiskValidationResult(
                accepted=False,
                reason="max_positions_limit",
                signal_risk_pct=signal_risk_pct,
                details={
                    "total_positions": exposure.total_positions,
                    "max_positions": self._config.max_positions,
                    "signal_risk_money": signal_risk_money,
                },
            )

        instrument_positions = exposure.positions_by_instrument.get(signal.instrument, 0)
        if not self._config.allow_multiple_positions_per_instrument and instrument_positions >= 1:
            return RiskValidationResult(
                accepted=False,
                reason="instrument_position_exists",
                signal_risk_pct=signal_risk_pct,
                details={"instrument": signal.instrument, "open_positions": instrument_positions},
            )

        if instrument_positions >= self._config.max_positions_per_instrument:
            return RiskValidationResult(
                accepted=False,
                reason="max_positions_per_instrument",
                signal_risk_pct=signal_risk_pct,
                details={
                    "instrument": signal.instrument,
                    "open_positions": instrument_positions,
                    "max_positions_per_instrument": self._config.max_positions_per_instrument,
                    "signal_risk_money": signal_risk_money,
                },
            )

        strategy_positions = exposure.positions_by_strategy.get(signal.strategy, 0)
        if strategy_positions >= self._config.max_positions_per_strategy:
            return RiskValidationResult(
                accepted=False,
                reason="max_positions_per_strategy",
                signal_risk_pct=signal_risk_pct,
                details={
                    "strategy": signal.strategy,
                    "open_positions": strategy_positions,
                    "max_positions_per_strategy": self._config.max_positions_per_strategy,
                    "signal_risk_money": signal_risk_money,
                },
            )

        for group_name in self._groups_for_symbol(signal.instrument):
            group_positions = exposure.positions_by_group.get(group_name, 0)
            if group_positions >= self._config.max_positions_per_correlation_group:
                return RiskValidationResult(
                    accepted=False,
                    reason="correlation_group_limit",
                    signal_risk_pct=signal_risk_pct,
                    details={
                        "group": group_name,
                        "group_positions": group_positions,
                        "max_positions_per_correlation_group": self._config.max_positions_per_correlation_group,
                    },
                )
            projected_group_risk_money = exposure.risk_money_by_group.get(group_name, 0.0) + signal_risk_money
            group_cap_money = self._pct_to_money(self._config.max_group_risk_pct)
            if projected_group_risk_money > group_cap_money:
                return RiskValidationResult(
                    accepted=False,
                    reason="group_risk_cap",
                    signal_risk_pct=signal_risk_pct,
                    details={
                        "group": group_name,
                        "projected_group_risk_money": projected_group_risk_money,
                        "group_risk_cap_money": group_cap_money,
                        "projected_group_risk_pct": self._money_to_pct(projected_group_risk_money),
                        "group_risk_cap_pct": self._config.max_group_risk_pct,
                    },
                )

        projected_total_risk_money = exposure.total_risk_money + signal_risk_money
        max_total_risk_money = self._pct_to_money(self._config.max_total_risk_pct)
        if projected_total_risk_money > max_total_risk_money:
            return RiskValidationResult(
                accepted=False,
                reason="max_total_risk_limit",
                signal_risk_pct=signal_risk_pct,
                details={
                    "projected_total_risk_money": projected_total_risk_money,
                    "max_total_risk_money": max_total_risk_money,
                    "projected_total_risk_pct": self._money_to_pct(projected_total_risk_money),
                    "max_total_risk_pct": self._config.max_total_risk_pct,
                },
            )

        projected_instrument_risk_money = exposure.risk_money_by_instrument.get(signal.instrument, 0.0) + signal_risk_money
        instrument_cap_money = self._instrument_risk_cap_money(signal.instrument)
        if projected_instrument_risk_money > instrument_cap_money:
            return RiskValidationResult(
                accepted=False,
                reason="instrument_risk_cap",
                signal_risk_pct=signal_risk_pct,
                details={
                    "instrument": signal.instrument,
                    "projected_instrument_risk_money": projected_instrument_risk_money,
                    "instrument_risk_cap_money": instrument_cap_money,
                    "projected_instrument_risk_pct": self._money_to_pct(projected_instrument_risk_money),
                    "instrument_risk_cap_pct": self._money_to_pct(instrument_cap_money),
                },
            )

        projected_strategy_risk_money = exposure.risk_money_by_strategy.get(signal.strategy, 0.0) + signal_risk_money
        strategy_cap_money = self._strategy_risk_cap_money(signal.strategy)
        if projected_strategy_risk_money > strategy_cap_money:
            return RiskValidationResult(
                accepted=False,
                reason="strategy_risk_cap",
                signal_risk_pct=signal_risk_pct,
                details={
                    "strategy": signal.strategy,
                    "projected_strategy_risk_money": projected_strategy_risk_money,
                    "strategy_risk_cap_money": strategy_cap_money,
                    "projected_strategy_risk_pct": self._money_to_pct(projected_strategy_risk_money),
                    "strategy_risk_cap_pct": self._money_to_pct(strategy_cap_money),
                },
            )

        return RiskValidationResult(
            accepted=True,
            reason="accepted",
            signal_risk_pct=signal_risk_pct,
            details={
                "signal_risk_pct": signal_risk_pct,
                "signal_risk_money": signal_risk_money,
                "position_qty": position_qty,
                "planned_risk_money": signal_risk_money,
                "planned_risk_pct": signal_risk_pct,
                "ticks": int(sizing.result.ticks),
                "money_per_contract": float(sizing.result.money_per_contract),
                "projected_total_risk_money": projected_total_risk_money,
                "projected_total_risk_pct": self._money_to_pct(projected_total_risk_money),
                "projected_instrument_risk_money": projected_instrument_risk_money,
                "projected_instrument_risk_pct": self._money_to_pct(projected_instrument_risk_money),
                "projected_strategy_risk_money": projected_strategy_risk_money,
                "projected_strategy_risk_pct": self._money_to_pct(projected_strategy_risk_money),
            },
        )

    def estimate_signal_risk_pct(self, signal: StrategySignal) -> float:
        return self._money_to_pct(self.estimate_signal_risk_money(signal))

    def estimate_signal_risk_money(self, signal: StrategySignal) -> float:
        meta = signal.metadata if isinstance(signal.metadata, dict) else {}
        raw = meta.get("planned_risk_money")
        try:
            hinted = float(raw)
        except (TypeError, ValueError):
            hinted = None
        if hinted is not None and hinted >= 0.0:
            return hinted

        entry = float(signal.entry)
        stop = float(signal.stop_loss)
        if entry <= 0:
            return 0.0
        base_pct = abs(entry - stop) / max(abs(entry), 1e-9) * 100.0
        return self._pct_to_money(base_pct)

    def estimate_position_risk_pct(self, position: Position) -> float:
        return self._money_to_pct(self.estimate_position_risk_money(position))

    def estimate_position_risk_money(self, position: Position) -> float:
        if position.stop_loss is None:
            return 0.0
        entry = float(position.entry_price)
        stop = float(position.stop_loss)
        if entry <= 0:
            return 0.0

        meta = position.metadata if isinstance(position.metadata, dict) else {}
        raw_money = meta.get("planned_risk_money")
        if raw_money is None:
            raw_money = meta.get("signal_risk_money")
        if raw_money is not None:
            try:
                money_value = float(raw_money)
            except (TypeError, ValueError):
                money_value = None
            if money_value is not None and money_value >= 0.0:
                return money_value

        raw_hint = meta.get("portfolio_risk_pct")
        if raw_hint is not None:
            try:
                hint_value = float(raw_hint)
            except (TypeError, ValueError):
                hint_value = None
            if hint_value is not None and hint_value >= 0.0:
                return self._pct_to_money(hint_value)

        base = abs(entry - stop) / max(abs(entry), 1e-9) * 100.0
        size = max(0.0, float(position.size))
        return self._pct_to_money(base * max(size, 1.0))

    def _size_signal(self, signal: StrategySignal):
        meta = signal.metadata if isinstance(signal.metadata, dict) else {}
        instrument_meta = {
            "tick_size": meta.get("tick_size"),
            "tick_value": meta.get("tick_value"),
            "lot_size": meta.get("lot_size", meta.get("lot")),
            "min_qty": meta.get("min_qty", 1.0),
            "qty_step": meta.get("qty_step", 1.0),
        }
        return self._position_sizer.size(
            entry_price=float(signal.entry),
            stop_loss=float(signal.stop_loss),
            instrument_metadata=instrument_meta,
            account_equity=self._config.account_size,
            risk_per_trade_pct=self._config.max_risk_per_trade_pct / 100.0,
        )

    def _groups_for_symbol(self, symbol: str) -> tuple[str, ...]:
        return self._instrument_to_groups.get(symbol, tuple())

    def _instrument_risk_cap_money(self, symbol: str) -> float:
        weight = self._config.instrument_weights.get(symbol)
        if weight is not None and weight > 0.0:
            return self._pct_to_money(self._config.max_total_risk_pct * weight)
        return self._pct_to_money(self._config.max_instrument_risk_pct)

    def _strategy_risk_cap_money(self, strategy: str) -> float:
        weight = self._config.strategy_weights.get(strategy)
        if weight is not None and weight > 0.0:
            return self._pct_to_money(self._config.max_total_risk_pct * weight)
        return self._pct_to_money(self._config.max_strategy_risk_pct)

    def _money_to_pct(self, amount: float) -> float:
        return (max(amount, 0.0) / max(self._config.account_size, 1e-9)) * 100.0

    def _pct_to_money(self, pct: float) -> float:
        return max(0.0, float(pct)) / 100.0 * max(self._config.account_size, 1e-9)


def _normalized_weights(raw: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            values[name] = numeric

    total = sum(values.values())
    if total <= 0.0:
        return {}
    return {key: val / total for key, val in values.items()}


def _index_correlation_groups(groups: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for group_name, symbols in groups.items():
        normalized_group = str(group_name).strip()
        if not normalized_group:
            continue
        for symbol in symbols:
            normalized_symbol = str(symbol).strip()
            if not normalized_symbol:
                continue
            out.setdefault(normalized_symbol, []).append(normalized_group)

    return {symbol: tuple(sorted(set(items))) for symbol, items in out.items()}


def _to_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default
