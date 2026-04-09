"""News blackout window checks."""

from __future__ import annotations

from datetime import datetime

from core.config_loader import BlackoutWindow


class NewsBlackoutFilter:
    """Answers whether new signals are blocked at a timestamp."""

    def __init__(self, windows: tuple[BlackoutWindow, ...]):
        self._windows = tuple(sorted(windows, key=lambda item: item.start))

    def is_blocked(self, timestamp: datetime) -> tuple[bool, str | None]:
        for window in self._windows:
            if window.start <= timestamp <= window.end:
                return True, window.description or "blackout"
        return False, None

    def windows(self) -> tuple[BlackoutWindow, ...]:
        return self._windows
