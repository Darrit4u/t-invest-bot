"""Deterministic position sizing helpers for risk-based quantity calculation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PositionSizingResult:
    """Successful sizing output."""

    qty: float
    risk_money: float
    risk_pct: float
    ticks: int
    money_per_contract: float


@dataclass(frozen=True, slots=True)
class PositionSizingReject:
    """Rejected sizing output with explicit reason."""

    reason: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PositionSizingDecision:
    """Either a successful sizing result or a reject reason."""

    accepted: bool
    result: PositionSizingResult | None = None
    reject: PositionSizingReject | None = None

    @staticmethod
    def ok(result: PositionSizingResult) -> "PositionSizingDecision":
        return PositionSizingDecision(accepted=True, result=result, reject=None)

    @staticmethod
    def fail(reason: str, details: dict[str, Any] | None = None) -> "PositionSizingDecision":
        return PositionSizingDecision(
            accepted=False,
            result=None,
            reject=PositionSizingReject(reason=reason, details=dict(details or {})),
        )


@dataclass(frozen=True, slots=True)
class InstrumentSizingMeta:
    """Normalized instrument metadata required for quantity sizing."""

    tick_size: float
    tick_value: float
    lot_size: float
    min_qty: float
    qty_step: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "InstrumentSizingMeta | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            tick_size = float(raw.get("tick_size"))
            tick_value = float(raw.get("tick_value"))
            # Accept both lot_size and legacy lot key for compatibility.
            lot_size = float(raw.get("lot_size", raw.get("lot")))
            min_qty = float(raw.get("min_qty", 1.0))
            qty_step = float(raw.get("qty_step", 1.0))
        except (TypeError, ValueError):
            return None
        if tick_size <= 0.0 or tick_value <= 0.0 or lot_size <= 0.0 or min_qty <= 0.0 or qty_step <= 0.0:
            return None
        return cls(
            tick_size=tick_size,
            tick_value=tick_value,
            lot_size=lot_size,
            min_qty=min_qty,
            qty_step=qty_step,
        )


class PositionSizer:
    """Pure position sizing calculator, independent from execution pipeline."""

    def size(
        self,
        *,
        entry_price: float,
        stop_loss: float,
        instrument_metadata: Mapping[str, Any] | None,
        account_equity: float,
        risk_per_trade_pct: float,
    ) -> PositionSizingDecision:
        meta = InstrumentSizingMeta.from_mapping(instrument_metadata)
        if meta is None:
            return PositionSizingDecision.fail("missing_metadata")

        try:
            entry = float(entry_price)
            stop = float(stop_loss)
            equity = float(account_equity)
            risk_pct = float(risk_per_trade_pct)
        except (TypeError, ValueError):
            return PositionSizingDecision.fail("invalid_input_types")

        if entry <= 0.0:
            return PositionSizingDecision.fail("invalid_entry_price", {"entry_price": entry})
        if stop <= 0.0:
            return PositionSizingDecision.fail("invalid_stop_loss", {"stop_loss": stop})
        if equity <= 0.0:
            return PositionSizingDecision.fail("invalid_account_equity", {"account_equity": equity})
        if risk_pct <= 0.0:
            return PositionSizingDecision.fail("invalid_risk_pct", {"risk_per_trade_pct": risk_pct})

        price_distance = abs(entry - stop)
        if price_distance <= 0.0:
            return PositionSizingDecision.fail("zero_stop_distance")

        risk_money = equity * risk_pct
        ticks = int(math.ceil(price_distance / meta.tick_size))
        if ticks <= 0:
            return PositionSizingDecision.fail("invalid_ticks", {"ticks": ticks})

        money_per_contract = ticks * meta.tick_value * meta.lot_size
        if money_per_contract <= 0.0:
            return PositionSizingDecision.fail(
                "invalid_money_per_contract",
                {"money_per_contract": money_per_contract},
            )

        qty_raw = risk_money / money_per_contract
        qty = _floor_to_step(qty_raw, meta.qty_step)
        if qty < meta.min_qty:
            return PositionSizingDecision.fail(
                "qty_below_min_qty",
                {
                    "qty": qty,
                    "min_qty": meta.min_qty,
                    "qty_raw": qty_raw,
                },
            )

        result = PositionSizingResult(
            qty=qty,
            risk_money=risk_money,
            risk_pct=(qty * money_per_contract) / equity,
            ticks=ticks,
            money_per_contract=money_per_contract,
        )
        return PositionSizingDecision.ok(result)


def _floor_to_step(value: float, step: float) -> float:
    ratio = (value + 1e-12) / step
    units = math.floor(ratio)
    rounded = units * step
    if step >= 1.0:
        return float(int(rounded))
    decimals = max(0, int(round(-math.log10(step))))
    return round(rounded, decimals + 2)
