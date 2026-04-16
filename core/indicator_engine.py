"""Indicator engine for intraday regime/strategy calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import tzinfo
from typing import Any

import pandas as pd

from core.market_data import Candle
from core.models import IndicatorSnapshot


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Global indicator defaults."""

    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    atr_period: int = 14
    volume_period: int = 20
    slope_period: int = 5
    crossing_lookback: int = 30
    overlap_window: int = 12
    swing_window: int = 5


class IndicatorEngine:
    """Builds indicator snapshot from candle history."""

    def __init__(self, config: IndicatorConfig | None = None):
        self._config = config or IndicatorConfig()

    def snapshot(
        self,
        candles: list[Candle],
        *,
        session_timezone: tzinfo | None = None,
    ) -> IndicatorSnapshot | None:
        min_required = max(
            self._config.atr_period + 2,
            self._config.volume_period + 2,
            self._config.slope_period + 2,
            self._config.overlap_window + 2,
            self._config.swing_window + 2,
            10,
        )
        if len(candles) < min_required:
            return None

        frame = _to_frame(candles)
        if frame.empty:
            return None

        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        tpv = typical * frame["volume"]
        utc_dt = pd.to_datetime(frame["datetime"], utc=True)
        if session_timezone is not None:
            local_dt = utc_dt.dt.tz_convert(session_timezone)
            session_key = local_dt.dt.date
        else:
            session_key = utc_dt.dt.date

        cumulative_tpv = tpv.groupby(session_key).cumsum()
        cumulative_vol = frame["volume"].groupby(session_key).cumsum().replace(0.0, pd.NA)
        frame["vwap"] = cumulative_tpv / cumulative_vol.ffill().fillna(1.0)

        frame["ema_fast"] = frame["close"].ewm(span=self._config.ema_fast, adjust=False).mean()
        frame["ema_slow"] = frame["close"].ewm(span=self._config.ema_slow, adjust=False).mean()
        frame["ema_trend"] = frame["close"].ewm(span=self._config.ema_trend, adjust=False).mean()

        prev_close = frame["close"].shift(1)
        tr_a = frame["high"] - frame["low"]
        tr_b = (frame["high"] - prev_close).abs()
        tr_c = (frame["low"] - prev_close).abs()
        true_range = pd.concat([tr_a, tr_b, tr_c], axis=1).max(axis=1)
        frame["atr"] = true_range.ewm(alpha=1.0 / self._config.atr_period, adjust=False).mean()

        frame["volume_avg"] = frame["volume"].rolling(self._config.volume_period).mean()

        slope_period = max(1, self._config.slope_period)
        frame["vwap_slope"] = frame["vwap"] - frame["vwap"].shift(slope_period)
        frame["ema_fast_slope"] = frame["ema_fast"] - frame["ema_fast"].shift(slope_period)
        frame["ema_slow_slope"] = frame["ema_slow"] - frame["ema_slow"].shift(slope_period)

        overlap_window = max(3, self._config.overlap_window)
        overlap_ratio = _calculate_overlap_ratio(frame.tail(overlap_window))

        crossing_count = _count_crossings(
            close=frame["close"].tail(self._config.crossing_lookback),
            level=frame["vwap"].tail(self._config.crossing_lookback),
        )

        swing_window = max(3, self._config.swing_window)
        swing_slice = frame.tail(swing_window)

        last = frame.iloc[-1]
        atr = float(last["atr"])
        range_width = float(frame["high"].tail(overlap_window).max() - frame["low"].tail(overlap_window).min())

        if not pd.notna(last["volume_avg"]):
            volume_avg = float(frame["volume"].tail(self._config.volume_period).mean())
        else:
            volume_avg = float(last["volume_avg"])

        return IndicatorSnapshot(
            timestamp=candles[-1].datetime,
            close=float(last["close"]),
            vwap=float(last["vwap"]),
            ema_fast=float(last["ema_fast"]),
            ema_slow=float(last["ema_slow"]),
            atr=max(atr, 1e-9),
            rolling_volume_avg=max(volume_avg, 1e-9),
            vwap_slope=float(last["vwap_slope"]) if pd.notna(last["vwap_slope"]) else 0.0,
            ema_fast_slope=float(last["ema_fast_slope"]) if pd.notna(last["ema_fast_slope"]) else 0.0,
            ema_slow_slope=float(last["ema_slow_slope"]) if pd.notna(last["ema_slow_slope"]) else 0.0,
            ema_distance=abs(float(last["ema_fast"] - last["ema_slow"])),
            crossing_count=crossing_count,
            range_width=range_width,
            overlap_ratio=overlap_ratio,
            swing_high=float(swing_slice["high"].max()),
            swing_low=float(swing_slice["low"].min()),
            extra={
                "ema_trend": float(last["ema_trend"]),
                "recent_high": float(frame["high"].tail(20).max()),
                "recent_low": float(frame["low"].tail(20).min()),
                "recent_close": float(last["close"]),
            },
        )


def _to_frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [item.datetime for item in candles],
            "open": [item.open for item in candles],
            "high": [item.high for item in candles],
            "low": [item.low for item in candles],
            "close": [item.close for item in candles],
            "volume": [item.volume for item in candles],
        }
    )


def _count_crossings(close: pd.Series, level: pd.Series) -> int:
    if len(close) < 2 or len(level) < 2:
        return 0

    delta = close - level
    signs = delta.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0)).tolist()
    count = 0
    last_non_zero = 0
    for sign in signs:
        if sign == 0:
            continue
        if last_non_zero != 0 and sign != last_non_zero:
            count += 1
        last_non_zero = sign
    return count


def _calculate_overlap_ratio(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 0.0

    overlaps = 0
    comparisons = 0
    highs = frame["high"].tolist()
    lows = frame["low"].tolist()

    for idx in range(1, len(frame)):
        prev_high = highs[idx - 1]
        prev_low = lows[idx - 1]
        cur_high = highs[idx]
        cur_low = lows[idx]

        overlap = max(0.0, min(cur_high, prev_high) - max(cur_low, prev_low))
        prev_range = max(1e-9, prev_high - prev_low)
        comparisons += 1
        if overlap / prev_range >= 0.5:
            overlaps += 1

    return overlaps / comparisons if comparisons else 0.0
