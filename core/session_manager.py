"""Session-aware trading availability checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from core.instrument_registry import InstrumentMeta


@dataclass(frozen=True, slots=True)
class SessionState:
    """Current session state for an instrument."""

    is_active: bool
    session_name: str | None


class SessionManager:
    """Determines whether an instrument is tradable at a given time."""

    def get_state(self, instrument: InstrumentMeta, timestamp_utc: datetime) -> SessionState:
        for session in instrument.sessions:
            tz = ZoneInfo(session.timezone)
            local_dt = timestamp_utc.astimezone(tz)
            current = local_dt.timetz().replace(tzinfo=None)

            start = _parse_time(session.start)
            end = _parse_time(session.end)

            if _is_time_inside(current=current, start=start, end=end):
                return SessionState(is_active=True, session_name=session.name)

        return SessionState(is_active=False, session_name=None)

    def primary_timezone(self, instrument: InstrumentMeta) -> ZoneInfo:
        if instrument.sessions:
            return ZoneInfo(instrument.sessions[0].timezone)
        return ZoneInfo("UTC")


def _parse_time(text: str) -> time:
    value = text.strip()
    if len(value) == 5:
        return datetime.strptime(value, "%H:%M").time()
    return datetime.strptime(value, "%H:%M:%S").time()


def _is_time_inside(*, current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    # Overnight sessions like 23:00-02:00
    return current >= start or current <= end
