"""Market regime classification based on score model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.models import IndicatorSnapshot, MarketRegime, MarketRegimeState


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
    min_dominant_score: float = 0.52


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
            min_dominant_score=float(section.get("min_dominant_score", 0.52)),
        )
        return cls(config=config)

    def classify(self, snapshot: IndicatorSnapshot) -> MarketRegime:
        return self.classify_state(snapshot).dominant

    def classify_state(self, snapshot: IndicatorSnapshot) -> MarketRegimeState:
        atr = max(snapshot.atr, 1e-9)
        vwap_slope_atr = snapshot.vwap_slope / atr
        ema_distance_atr = snapshot.ema_distance / atr
        range_atr = snapshot.range_width / atr

        trend_score, trend_reasons = self._trend_score(
            snapshot=snapshot,
            ema_distance_atr=ema_distance_atr,
            vwap_slope_atr=vwap_slope_atr,
        )
        compression_score, compression_reasons = self._compression_score(
            snapshot=snapshot,
            range_atr=range_atr,
            ema_distance_atr=ema_distance_atr,
            vwap_slope_atr=vwap_slope_atr,
        )
        balance_score, balance_reasons = self._balance_score(
            snapshot=snapshot,
            ema_distance_atr=ema_distance_atr,
            vwap_slope_atr=vwap_slope_atr,
        )

        dominant = self._dominant_regime(
            trend_score=trend_score,
            compression_score=compression_score,
            balance_score=balance_score,
        )
        reasons = tuple(trend_reasons + compression_reasons + balance_reasons)
        details = {
            "vwap_slope_atr": vwap_slope_atr,
            "ema_distance_atr": ema_distance_atr,
            "range_atr": range_atr,
            "crossing_count": snapshot.crossing_count,
            "overlap_ratio": snapshot.overlap_ratio,
        }
        return MarketRegimeState(
            dominant=dominant,
            trend_score=trend_score,
            compression_score=compression_score,
            balance_score=balance_score,
            reason_codes=reasons,
            details=details,
        )

    def _trend_score(
        self,
        *,
        snapshot: IndicatorSnapshot,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> tuple[float, list[str]]:
        ema_trend = _safe_float(snapshot.extra.get("ema_trend"))
        if ema_trend is not None:
            long_bias = snapshot.close > ema_trend and snapshot.ema_slow > ema_trend and snapshot.ema_fast > snapshot.ema_slow
            short_bias = snapshot.close < ema_trend and snapshot.ema_slow < ema_trend and snapshot.ema_fast < snapshot.ema_slow
        else:
            long_bias = snapshot.close > snapshot.vwap and snapshot.ema_fast > snapshot.ema_slow
            short_bias = snapshot.close < snapshot.vwap and snapshot.ema_fast < snapshot.ema_slow
        directional_alignment = 1.0 if (long_bias or short_bias) else 0.0
        slope_strength = _clamp01(
            abs(vwap_slope_atr) / max(self._config.trend_vwap_slope_atr, 1e-9)
        )
        distance_strength = _clamp01(
            ema_distance_atr / max(self._config.trend_ema_distance_atr, 1e-9)
        )
        crossing_cleanliness = _clamp01(
            1.0 - _safe_div(snapshot.crossing_count, max(self._config.trend_crossing_max, 1))
        )
        score = (
            0.35 * directional_alignment
            + 0.25 * slope_strength
            + 0.25 * distance_strength
            + 0.15 * crossing_cleanliness
        )
        reasons: list[str] = []
        if directional_alignment >= 1.0:
            reasons.append("trend_alignment_ok")
        else:
            reasons.append("trend_alignment_missing")
        if slope_strength >= 0.9:
            reasons.append("trend_slope_strong")
        elif slope_strength <= 0.25:
            reasons.append("trend_slope_weak")
        if crossing_cleanliness <= 0.25:
            reasons.append("trend_crossing_noise")
        return _clamp01(score), reasons

    def _compression_score(
        self,
        *,
        snapshot: IndicatorSnapshot,
        range_atr: float,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> tuple[float, list[str]]:
        min_range = self._config.compression_range_min_atr
        max_range = self._config.compression_range_max_atr
        if min_range <= range_atr <= max_range:
            range_fit = 1.0
        else:
            outlier = min(abs(range_atr - min_range), abs(range_atr - max_range))
            range_fit = _clamp01(1.0 - _safe_div(outlier, max(max_range - min_range, 1e-9)))

        ema_tightness = _clamp01(
            1.0 - _safe_div(ema_distance_atr, max(self._config.compression_ema_distance_atr, 1e-9))
        )
        slope_flatness = _clamp01(
            1.0
            - _safe_div(
                abs(vwap_slope_atr),
                max(self._config.compression_vwap_slope_abs_atr, 1e-9),
            )
        )
        overlap_strength = _clamp01(
            _safe_div(snapshot.overlap_ratio, max(self._config.compression_overlap_min, 1e-9))
        )
        score = (
            0.35 * range_fit
            + 0.25 * ema_tightness
            + 0.20 * slope_flatness
            + 0.20 * overlap_strength
        )
        reasons: list[str] = []
        if range_fit >= 0.9:
            reasons.append("compression_range_fit")
        if overlap_strength >= 0.9:
            reasons.append("compression_overlap_high")
        if slope_flatness <= 0.25:
            reasons.append("compression_slope_too_strong")
        return _clamp01(score), reasons

    def _balance_score(
        self,
        *,
        snapshot: IndicatorSnapshot,
        ema_distance_atr: float,
        vwap_slope_atr: float,
    ) -> tuple[float, list[str]]:
        crossing_strength = _clamp01(
            _safe_div(snapshot.crossing_count, max(self._config.balance_crossing_min, 1))
        )
        if snapshot.crossing_count < self._config.balance_crossing_min:
            crossing_strength *= 0.4
        ema_tightness = _clamp01(
            1.0 - _safe_div(ema_distance_atr, max(self._config.balance_ema_distance_atr, 1e-9))
        )
        slope_flatness = _clamp01(
            1.0 - _safe_div(abs(vwap_slope_atr), max(self._config.balance_vwap_slope_abs_atr, 1e-9))
        )
        score = (
            0.45 * crossing_strength
            + 0.30 * ema_tightness
            + 0.25 * slope_flatness
        )
        reasons: list[str] = []
        if crossing_strength >= 0.9:
            reasons.append("balance_crossing_density_high")
        if slope_flatness >= 0.9:
            reasons.append("balance_vwap_flat")
        if ema_tightness <= 0.2:
            reasons.append("balance_ema_spread_wide")
        return _clamp01(score), reasons

    def _dominant_regime(
        self,
        *,
        trend_score: float,
        compression_score: float,
        balance_score: float,
    ) -> MarketRegime:
        ranked = [
            (MarketRegime.TREND, trend_score),
            (MarketRegime.COMPRESSION, compression_score),
            (MarketRegime.BALANCE, balance_score),
        ]
        ranked.sort(key=lambda row: row[1], reverse=True)
        best_regime, best_score = ranked[0]
        second_score = ranked[1][1]

        if best_score < self._config.min_dominant_score:
            return MarketRegime.NEUTRAL
        if (best_score - second_score) < 0.08 and best_score < (self._config.min_dominant_score + 0.08):
            return MarketRegime.NEUTRAL
        return best_regime


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)
