"""Shared post-fill validation helpers for filter and simulator."""

from __future__ import annotations

from dataclasses import dataclass

from core.models import SignalDirection


@dataclass(frozen=True, slots=True)
class PostFillValidationConfig:
    commission_roundtrip: float
    safety_multiplier: float
    min_rr_after_fill: float
    min_expected_edge_after_fees: float


@dataclass(frozen=True, slots=True)
class PostFillMetrics:
    entry_price: float
    risk: float
    reward: float
    post_fill_rr: float
    expected_edge_after_fees: float


@dataclass(frozen=True, slots=True)
class PostFillValidationResult:
    accepted: bool
    reason: str | None
    metrics: PostFillMetrics


def expected_fill_price(
    *,
    direction: SignalDirection,
    entry_mode: str,
    planned_entry: float,
    last_close: float,
    atr: float,
    expected_open_slippage_atr: float,
) -> float:
    normalized_mode = str(entry_mode).strip().upper()
    if normalized_mode != "NEXT_BAR_OPEN":
        return float(planned_entry)

    safe_atr = max(float(atr), 1e-9)
    slippage = safe_atr * max(float(expected_open_slippage_atr), 0.0)
    if direction == SignalDirection.LONG:
        baseline = float(last_close) + slippage
        return max(float(planned_entry), baseline)
    baseline = float(last_close) - slippage
    return min(float(planned_entry), baseline)


def compute_post_fill_metrics(
    *,
    direction: SignalDirection,
    stop_loss: float,
    tp1: float,
    entry_price: float,
    commission_roundtrip: float,
    safety_multiplier: float,
) -> PostFillMetrics:
    safe_entry = float(entry_price)
    safe_stop = float(stop_loss)
    safe_tp1 = float(tp1)
    commission_cost = (
        max(safe_entry, 0.0) * max(float(commission_roundtrip), 0.0) * max(float(safety_multiplier), 0.0)
    )

    if direction == SignalDirection.LONG:
        risk = safe_entry - safe_stop
        reward = safe_tp1 - safe_entry
    else:
        risk = safe_stop - safe_entry
        reward = safe_entry - safe_tp1

    if risk <= 0.0 or reward <= 0.0:
        return PostFillMetrics(
            entry_price=safe_entry,
            risk=max(risk, 0.0),
            reward=max(reward, 0.0),
            post_fill_rr=0.0,
            expected_edge_after_fees=-(abs(reward) + commission_cost),
        )

    rr = reward / risk
    return PostFillMetrics(
        entry_price=safe_entry,
        risk=risk,
        reward=reward,
        post_fill_rr=rr,
        expected_edge_after_fees=reward - commission_cost,
    )


def validate_post_fill(
    *,
    direction: SignalDirection,
    stop_loss: float,
    tp1: float,
    entry_price: float,
    config: PostFillValidationConfig,
) -> PostFillValidationResult:
    metrics = compute_post_fill_metrics(
        direction=direction,
        stop_loss=stop_loss,
        tp1=tp1,
        entry_price=entry_price,
        commission_roundtrip=config.commission_roundtrip,
        safety_multiplier=config.safety_multiplier,
    )
    if metrics.post_fill_rr < float(config.min_rr_after_fill):
        return PostFillValidationResult(
            accepted=False,
            reason="poor_rr_after_fill",
            metrics=metrics,
        )
    if metrics.expected_edge_after_fees < float(config.min_expected_edge_after_fees):
        return PostFillValidationResult(
            accepted=False,
            reason="low_expected_edge",
            metrics=metrics,
        )
    return PostFillValidationResult(
        accepted=True,
        reason=None,
        metrics=metrics,
    )
