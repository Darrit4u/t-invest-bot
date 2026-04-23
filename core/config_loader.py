"""Configuration loading and validation for the signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from core.timeframes import supported_timeframes


class ConfigError(ValueError):
    """Raised when configuration files are missing or invalid."""


@dataclass(frozen=True, slots=True)
class SessionRule:
    """Trading session definition for an instrument."""

    name: str
    start: str
    end: str
    timezone: str


@dataclass(frozen=True, slots=True)
class InstrumentConfig:
    """Static metadata for a tradable instrument."""

    symbol: str
    enabled: bool
    uid: str | None
    figi: str | None
    ticker: str | None
    class_code: str | None
    tick_size: float
    tick_value: float
    lot: int
    sessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlackoutWindow:
    """Datetime interval where new signals are blocked."""

    start: datetime
    end: datetime
    description: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Aggregated application configuration."""

    config_dir: Path
    history_depth: int
    default_timeframe: str
    session_rules: dict[str, SessionRule]
    instruments: dict[str, InstrumentConfig]
    strategies_by_instrument: dict[str, tuple[str, ...]]
    params: dict[str, Any]
    blackout_windows: tuple[BlackoutWindow, ...] = field(default_factory=tuple)


class ConfigLoader:
    """Loads and validates YAML configuration files from a folder."""

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir

    def load(self) -> AppConfig:
        if not self._config_dir.exists():
            raise ConfigError(f"Config directory does not exist: {self._config_dir}")

        instruments_raw = self._read_yaml("instruments.yaml")
        strategies_raw = self._read_yaml("strategies.yaml")
        params_raw = self._read_yaml("params.yaml", default={})
        blackout_file = str(params_raw.get("news_blackout_file", "news_blackout.yaml")).strip()
        blackout_raw = self._read_yaml(blackout_file or "news_blackout.yaml", default=[])

        history_depth = int(instruments_raw.get("history_depth", 500))
        if history_depth < 10:
            raise ConfigError("history_depth must be >= 10")

        default_timeframe = str(instruments_raw.get("default_timeframe", "1min"))
        if not default_timeframe:
            raise ConfigError("default_timeframe must not be empty")
        if default_timeframe.lower() not in supported_timeframes():
            raise ConfigError(
                "default_timeframe must be one of: "
                + ", ".join(sorted(supported_timeframes()))
            )

        session_rules = self._parse_sessions(instruments_raw.get("session_rules", {}))
        instruments = self._parse_instruments(instruments_raw.get("instruments", {}), session_rules)
        strategies_by_instrument = self._parse_strategy_map(strategies_raw, instruments)
        blackout_windows = self._parse_blackout_windows(
            blackout_raw,
            default_timezone=str(params_raw.get("timezone", "Europe/Moscow")),
        )

        return AppConfig(
            config_dir=self._config_dir,
            history_depth=history_depth,
            default_timeframe=default_timeframe,
            session_rules=session_rules,
            instruments=instruments,
            strategies_by_instrument=strategies_by_instrument,
            params=params_raw,
            blackout_windows=blackout_windows,
        )

    def _read_yaml(self, name: str, default: Any | None = None) -> Any:
        path = self._config_dir / name
        if not path.exists():
            if default is not None:
                return default
            raise ConfigError(f"Required config file is missing: {path}")

        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse failed for {path}: {exc}") from exc

        if payload is None and default is not None:
            return default
        return payload

    def _parse_sessions(self, payload: Any) -> dict[str, SessionRule]:
        if not isinstance(payload, dict):
            raise ConfigError("session_rules must be a mapping")

        sessions: dict[str, SessionRule] = {}
        for name, rule in payload.items():
            if not isinstance(rule, dict):
                raise ConfigError(f"session_rules.{name} must be an object")

            start = str(rule.get("start", "")).strip()
            end = str(rule.get("end", "")).strip()
            timezone = str(rule.get("timezone", "Europe/Moscow")).strip()
            if not start or not end:
                raise ConfigError(f"session_rules.{name} must include start and end")
            self._validate_session_time(start, field=f"session_rules.{name}.start")
            self._validate_session_time(end, field=f"session_rules.{name}.end")

            try:
                ZoneInfo(timezone)
            except Exception as exc:  # pragma: no cover - platform-dependent
                raise ConfigError(f"Invalid timezone in session_rules.{name}: {timezone}") from exc

            sessions[name] = SessionRule(name=name, start=start, end=end, timezone=timezone)
        return sessions

    def _parse_instruments(
        self,
        payload: Any,
        session_rules: dict[str, SessionRule],
    ) -> dict[str, InstrumentConfig]:
        if not isinstance(payload, dict):
            raise ConfigError("instruments must be a mapping")

        instruments: dict[str, InstrumentConfig] = {}
        for symbol, row in payload.items():
            if not isinstance(row, dict):
                raise ConfigError(f"instruments.{symbol} must be an object")

            sessions = tuple(str(item) for item in row.get("sessions", []))
            if not sessions:
                raise ConfigError(f"instruments.{symbol}.sessions must not be empty")

            for session_name in sessions:
                if session_name not in session_rules:
                    raise ConfigError(
                        f"instruments.{symbol} references unknown session: {session_name}"
                    )

            try:
                tick_size = float(row["tick_size"])
                tick_value = float(row["tick_value"])
                lot = int(row.get("lot", 1))
            except KeyError as exc:
                raise ConfigError(f"instruments.{symbol} missing required key: {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"instruments.{symbol} tick_size/tick_value/lot contain invalid types"
                ) from exc

            if tick_size <= 0 or tick_value <= 0 or lot <= 0:
                raise ConfigError(f"instruments.{symbol} tick_size, tick_value and lot must be > 0")

            instruments[symbol] = InstrumentConfig(
                symbol=symbol,
                enabled=bool(row.get("enabled", True)),
                uid=self._optional_string(row.get("uid")),
                figi=self._optional_string(row.get("figi")),
                ticker=self._optional_string(row.get("ticker", symbol)),
                class_code=self._optional_string(row.get("class_code")),
                tick_size=tick_size,
                tick_value=tick_value,
                lot=lot,
                sessions=sessions,
            )

        if not instruments:
            raise ConfigError("At least one instrument must be configured")

        enabled_count = sum(1 for item in instruments.values() if item.enabled)
        if enabled_count == 0:
            raise ConfigError("At least one enabled instrument is required")

        return instruments

    def _parse_strategy_map(
        self,
        payload: Any,
        instruments: dict[str, InstrumentConfig],
    ) -> dict[str, tuple[str, ...]]:
        if not isinstance(payload, dict):
            raise ConfigError("strategies.yaml root must be a mapping")

        strategies_payload = payload.get("strategies")
        if not isinstance(strategies_payload, dict):
            raise ConfigError("strategies.yaml must contain 'strategies' mapping")

        result: dict[str, tuple[str, ...]] = {}
        for symbol, instrument in instruments.items():
            row = strategies_payload.get(symbol, [])
            if not isinstance(row, list):
                raise ConfigError(f"strategies.{symbol} must be a list")
            values = tuple(str(item).strip() for item in row if str(item).strip())
            if instrument.enabled and not values:
                raise ConfigError(f"Enabled instrument {symbol} must have at least one strategy")
            result[symbol] = values
        return result

    def _parse_blackout_windows(
        self,
        payload: Any,
        default_timezone: str,
    ) -> tuple[BlackoutWindow, ...]:
        if payload in ({}, None, ""):
            return tuple()

        if not isinstance(payload, list):
            raise ConfigError("news_blackout.yaml must be a list")

        zone = ZoneInfo(default_timezone)
        windows: list[BlackoutWindow] = []
        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ConfigError(f"news_blackout item #{idx} must be an object")

            start_text = str(row.get("start", "")).strip()
            end_text = str(row.get("end", "")).strip()
            description = str(row.get("description", "")).strip()
            if not start_text or not end_text:
                raise ConfigError(f"news_blackout item #{idx} must contain start and end")

            start = self._parse_datetime(start_text, zone)
            end = self._parse_datetime(end_text, zone)
            if end <= start:
                raise ConfigError(f"news_blackout item #{idx} has end <= start")

            windows.append(BlackoutWindow(start=start, end=end, description=description))

        return tuple(windows)

    @staticmethod
    def _parse_datetime(value: str, zone: ZoneInfo) -> datetime:
        formats = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S")
        for pattern in formats:
            try:
                naive = datetime.strptime(value, pattern)
                return naive.replace(tzinfo=zone)
            except ValueError:
                continue
        raise ConfigError(f"Unsupported datetime format: {value}")

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _validate_session_time(value: str, *, field: str) -> None:
        patterns = ("%H:%M", "%H:%M:%S")
        for pattern in patterns:
            try:
                datetime.strptime(value, pattern)
                return
            except ValueError:
                continue
        raise ConfigError(f"Invalid time in {field}: {value!r}. Use HH:MM or HH:MM:SS")

