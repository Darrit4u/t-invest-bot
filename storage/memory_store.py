"""In-memory candle storage for runtime computations."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque

from core.market_data import Candle


@dataclass(frozen=True, slots=True)
class StoreStats:
    """Lightweight storage metrics for monitoring and tests."""

    instruments: int
    streams: int
    candles: int


class MemoryCandleStore:
    """Keeps last N candles for each (instrument, timeframe)."""

    def __init__(self, history_depth: int):
        if history_depth < 10:
            raise ValueError("history_depth must be >= 10")
        self._history_depth = history_depth
        self._data: dict[tuple[str, str], Deque[Candle]] = defaultdict(
            lambda: deque(maxlen=self._history_depth)
        )

    def upsert(self, candle: Candle) -> str:
        """
        Insert a candle or update the latest one.

        Returns: one of `inserted`, `updated`, `ignored`.
        """

        key = (candle.instrument, candle.timeframe)
        bucket = self._data[key]

        if not bucket:
            bucket.append(candle)
            return "inserted"

        last = bucket[-1]
        if candle.datetime == last.datetime:
            bucket[-1] = candle
            return "updated"

        if candle.datetime > last.datetime:
            bucket.append(candle)
            return "inserted"

        # Out-of-order update of an existing timestamp.
        for idx, existing in enumerate(bucket):
            if existing.datetime == candle.datetime:
                bucket[idx] = candle
                return "updated"

        # Too old and missing from retained history.
        return "ignored"

    def get_recent(
        self,
        instrument: str,
        timeframe: str,
        limit: int | None = None,
    ) -> list[Candle]:
        key = (instrument, timeframe)
        bucket = self._data.get(key)
        if not bucket:
            return []

        rows = list(bucket)
        if limit is None or limit >= len(rows):
            return rows
        return rows[-limit:]

    def latest(self, instrument: str, timeframe: str) -> Candle | None:
        key = (instrument, timeframe)
        bucket = self._data.get(key)
        if not bucket:
            return None
        return bucket[-1]

    def stats(self) -> StoreStats:
        streams = len(self._data)
        instruments = len({instrument for instrument, _ in self._data})
        candles = sum(len(bucket) for bucket in self._data.values())
        return StoreStats(instruments=instruments, streams=streams, candles=candles)

    @staticmethod
    def timestamps(rows: list[Candle]) -> list[datetime]:
        return [item.datetime for item in rows]
