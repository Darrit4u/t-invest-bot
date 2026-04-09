"""Market regime classification based on indicator snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.models import IndicatorSnapshot, MarketRegime


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Thresholds used by market regime classifier."""

    trend_ema_distance_atr: float = 0.20
    trend_vwap_slope_atr: float = 0.03
    trend_crossing_max: int = 3
    compression_range_min_atr: float = 0.7
    compression_range_max_atr: float = 2.2
    compression_ema_distance_atr: float = 0.12
    compression_vwap_slope_abs_atr: float = 0.05
    compression_overlap_min: float = 0.60
    balance_crossing_min: int = 4
    balance_ema_distance_atr: float = 0.12
    balance_vwap_slope_abs_atr: float = 0.05


class MarketRegimeClassifier:
    """Classifies current market regime for strategy routing."""

    def __init__(self, config: RegimeConfig | None = None):
        self._config = config or RegimeConfig()

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "MarketRegimeClassifier":
        section = params.get("regime_classifier", {})
        if not isinstance(section, dict):
            section = {}
        config = RegimeConfig(
            trend_ema_distance_atr=float(section.get("trend_ema_distance_atr", 0.20)),
            trend_vwap_slope_atr=float(section.get("trend_vwap_slope_atr", 0.03)),
            trend_crossing_max=int(section.get("trend_crossing_max", 3)),
            compression_range_min_atr=float(section.get("compression_range_min_atr", 0.7)),
            compression_range_max_atr=float(section.get("compression_range_max_atr", 2.2)),
            compression_ema_distance_atr=float(section.get("compression_ema_distance_atr", 0.12)),
            compression_vwap_slope_abs_atr=float(
                section.get("compression_vwap_slope_abs_atr", 0.05)
            ),
            compression_overlap_min=float(section.get("compression_overlap_min", 0.60)),
            balance_crossing_min=int(section.get("balance_crossing_min", 4)),
            balance_ema_distance_atr=float(section.get("balance_ema_distance_atr", 0.12)),
            balance_vwap_slope_abs_atr=float(section.get("balance_vwap_slope_abs_atr", 0.05)),
        )
        return cls(config=config)

    def classify(self, snapshot: IndicatorSnapshot) -> MarketRegime:
        atr = max(snapshot.atr, 1e-9)
        vwap_slope_atr = snapshot.vwap_slope / atr
        ema_distance_atr = snapshot.ema_distance / atr
        range_atr = snapshot.range_width / atr

        if self._is_compression(snapshot, range_atr, ema_distance_atr, vwap_slope_atr):
            return MarketRegime.COMPRESSION

        if self._is_trend(snapshot, ema_distance_atr, vwap_slope_atr):
            return MarketRegime.TREND

        if self._is_balance(snapshot, ema_distance_atr, vwap_slope_atr):
            return MarketRegime.BALANCE

        return MarketRegime.NEUTRAL

    def _is_trend(
        self,
        snapshot: IndicatorSnapshot,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> bool:
        long_bias = snapshot.close > snapshot.vwap and snapshot.ema_fast > snapshot.ema_slow
        short_bias = snapshot.close < snapshot.vwap and snapshot.ema_fast < snapshot.ema_slow
        has_slope = abs(vwap_slope_atr) >= self._config.trend_vwap_slope_atr

        return (
            (long_bias or short_bias)
            and has_slope
            and ema_distance_atr >= self._config.trend_ema_distance_atr
            and snapshot.crossing_count <= self._config.trend_crossing_max
        )

    def _is_compression(
        self,
        snapshot: IndicatorSnapshot,
        range_atr: float,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> bool:
        return (
            self._config.compression_range_min_atr <= range_atr <= self._config.compression_range_max_atr
            and ema_distance_atr <= self._config.compression_ema_distance_atr
            and abs(vwap_slope_atr) <= self._config.compression_vwap_slope_abs_atr
            and snapshot.overlap_ratio >= self._config.compression_overlap_min
        )

    def _is_balance(
        self,
        snapshot: IndicatorSnapshot,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> bool:
        return (
            snapshot.crossing_count >= self._config.balance_crossing_min
            and ema_distance_atr <= self._config.balance_ema_distance_atr
            and abs(vwap_slope_atr) <= self._config.balance_vwap_slope_abs_atr
        )
