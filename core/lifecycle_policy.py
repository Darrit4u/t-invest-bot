"""Lifecycle policies for intraday and swing execution behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.trading_mode import TradingMode, resolve_trading_mode


@dataclass(frozen=True, slots=True)
class LifecycleCloseAction:
    """Instruction to force-close trade in simulator."""

    status: str
    event_type: str
    reason: str


class BaseLifecyclePolicy:
    """Base API for lifecycle behavior by trading mode."""

    def waiting_session_action(self, *, session_active: bool) -> LifecycleCloseAction | None:
        raise NotImplementedError

    def active_session_action(
        self,
        *,
        session_active: bool,
        profitable: bool,
    ) -> LifecycleCloseAction | None:
        raise NotImplementedError

    def holding_expiry_action(
        self,
        *,
        opened_at: datetime,
        now: datetime,
        bars_in_trade: int,
    ) -> LifecycleCloseAction | None:
        raise NotImplementedError

    @property
    def gap_risk_handling(self) -> str:
        return "legacy"


@dataclass(frozen=True, slots=True)
class IntradayLifecyclePolicy(BaseLifecyclePolicy):
    max_trade_bars: int
    close_profitable_on_session_end: bool
    gap_risk_mode: str = "legacy"

    def waiting_session_action(self, *, session_active: bool) -> LifecycleCloseAction | None:
        if session_active:
            return None
        return LifecycleCloseAction(
            status="cancelled_by_session_end",
            event_type="cancelled_by_session_end",
            reason="session_inactive_before_activation",
        )

    def active_session_action(
        self,
        *,
        session_active: bool,
        profitable: bool,
    ) -> LifecycleCloseAction | None:
        if session_active:
            return None
        if self.close_profitable_on_session_end and not profitable:
            return None
        reason = "session_end_take_profit" if self.close_profitable_on_session_end else "session_inactive"
        return LifecycleCloseAction(
            status="cancelled_by_session_end",
            event_type="cancelled_by_session_end",
            reason=reason,
        )

    def holding_expiry_action(
        self,
        *,
        opened_at: datetime,
        now: datetime,
        bars_in_trade: int,
    ) -> LifecycleCloseAction | None:
        if bars_in_trade > self.max_trade_bars:
            return LifecycleCloseAction(
                status="expired",
                event_type="expired",
                reason="max_bars_in_trade",
            )
        return None

    @property
    def gap_risk_handling(self) -> str:
        return self.gap_risk_mode


@dataclass(frozen=True, slots=True)
class SwingLifecyclePolicy(BaseLifecyclePolicy):
    allow_overnight: bool
    use_session_force_close: bool
    max_holding_bars: int
    max_holding_days: int
    fallback_max_trade_bars: int
    close_profitable_on_session_end: bool
    gap_risk_mode: str

    def waiting_session_action(self, *, session_active: bool) -> LifecycleCloseAction | None:
        if session_active or self.allow_overnight:
            return None
        return LifecycleCloseAction(
            status="cancelled_by_session_end",
            event_type="cancelled_by_session_end",
            reason="session_inactive_before_activation",
        )

    def active_session_action(
        self,
        *,
        session_active: bool,
        profitable: bool,
    ) -> LifecycleCloseAction | None:
        if session_active or not self.use_session_force_close:
            return None
        if self.close_profitable_on_session_end and not profitable:
            return None
        reason = "session_end_take_profit" if self.close_profitable_on_session_end else "session_inactive"
        return LifecycleCloseAction(
            status="cancelled_by_session_end",
            event_type="cancelled_by_session_end",
            reason=reason,
        )

    def holding_expiry_action(
        self,
        *,
        opened_at: datetime,
        now: datetime,
        bars_in_trade: int,
    ) -> LifecycleCloseAction | None:
        bars_limit = self.max_holding_bars if self.max_holding_bars > 0 else self.fallback_max_trade_bars
        if bars_limit > 0 and bars_in_trade > bars_limit:
            return LifecycleCloseAction(
                status="expired",
                event_type="expired",
                reason="max_holding_bars",
            )

        if self.max_holding_days > 0:
            if now >= opened_at + timedelta(days=self.max_holding_days):
                return LifecycleCloseAction(
                    status="expired",
                    event_type="expired",
                    reason="max_holding_days",
                )
        return None

    @property
    def gap_risk_handling(self) -> str:
        return self.gap_risk_mode


def build_lifecycle_policy(
    *,
    params: dict[str, Any],
    max_trade_bars: int,
    close_profitable_on_session_end: bool,
) -> BaseLifecyclePolicy:
    mode = resolve_trading_mode(params)
    sim_cfg = params.get("trade_simulator", {}) if isinstance(params.get("trade_simulator", {}), dict) else {}
    swing_cfg = params.get("swing", {}) if isinstance(params.get("swing", {}), dict) else {}
    execution_cfg = params.get("execution", {}) if isinstance(params.get("execution", {}), dict) else {}
    execution_gap_mode = str(execution_cfg.get("gap_risk_handling", "")).strip().lower()

    if mode == TradingMode.SWING:
        return SwingLifecyclePolicy(
            allow_overnight=_to_bool(swing_cfg.get("allow_overnight", True), default=True),
            use_session_force_close=_to_bool(swing_cfg.get("use_session_force_close", False), default=False),
            max_holding_bars=max(0, int(swing_cfg.get("max_holding_bars", 40))),
            max_holding_days=max(0, int(swing_cfg.get("max_holding_days", 5))),
            fallback_max_trade_bars=max_trade_bars,
            close_profitable_on_session_end=close_profitable_on_session_end,
            gap_risk_mode=(
                execution_gap_mode
                or str(swing_cfg.get("gap_risk_handling", "conservative")).strip().lower()
                or "conservative"
            ),
        )

    return IntradayLifecyclePolicy(
        max_trade_bars=max_trade_bars,
        close_profitable_on_session_end=close_profitable_on_session_end,
        gap_risk_mode=execution_gap_mode or str(sim_cfg.get("gap_risk_handling", "legacy")).strip().lower() or "legacy",
    )


def _to_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
