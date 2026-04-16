"""No-trade windows analysis for backtest reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from core.config_loader import AppConfig
from core.instrument_registry import InstrumentMeta, InstrumentRegistry


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Thresholds and constraints for no-trade candidate selection."""

    min_trades_per_bucket: int = 10
    min_winrate_gap_pp: float = 12.0
    min_negative_avg_r: float = 0.0
    validation_ratio: float = 0.3
    min_expected_trades_per_day: float = 1.0
    include_combined_buckets: bool = False


@dataclass(frozen=True, slots=True)
class TradeSample:
    """One closed trade normalized for analysis."""

    profile: str
    strategy: str
    instrument: str
    trade_id: str
    timestamp_utc: datetime
    timestamp_local: datetime
    local_hour: int
    local_weekday: int
    local_date: str
    net_pnl: float
    r_multiple: float


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    """Aggregate metrics for one collection of trades."""

    trades_count: int
    wins: int
    losses: int
    winrate: float
    avg_r: float
    net_pnl: float
    profit_factor: float
    max_loss_streak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades_count": self.trades_count,
            "wins": self.wins,
            "losses": self.losses,
            "winrate": self.winrate,
            "winrate_pct": self.winrate * 100.0,
            "avg_r": self.avg_r,
            "net_pnl": self.net_pnl,
            "profit_factor": self.profit_factor if self.profit_factor < float("inf") else None,
            "max_loss_streak": self.max_loss_streak,
        }


@dataclass(frozen=True, slots=True)
class BucketCandidate:
    """Confirmed poor-performance bucket in both train and validation."""

    bucket_type: str
    bucket_value: int | str
    severity_score: float
    train_metrics: TradeMetrics
    validation_metrics: TradeMetrics
    train_winrate_gap_pp: float
    validation_winrate_gap_pp: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_type": self.bucket_type,
            "bucket_value": self.bucket_value,
            "severity_score": self.severity_score,
            "train_winrate_gap_pp": self.train_winrate_gap_pp,
            "validation_winrate_gap_pp": self.validation_winrate_gap_pp,
            "reasons": list(self.reasons),
            "train_metrics": self.train_metrics.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairRecommendation:
    """Actionable filter recommendation for one instrument+strategy pair."""

    blocked_entry_hours_local: tuple[int, ...]
    blocked_entry_weekdays_local: tuple[int, ...]
    blocked_time_windows_local: tuple[str, ...]
    high_risk_filter: bool
    expected_trades_per_day_before: float
    expected_trades_per_day_after: float
    trades_before: int
    trades_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_entry_hours_local": list(self.blocked_entry_hours_local),
            "blocked_entry_weekdays_local": list(self.blocked_entry_weekdays_local),
            "blocked_time_windows_local": list(self.blocked_time_windows_local),
            "high_risk_filter": self.high_risk_filter,
            "expected_trades_per_day_before": self.expected_trades_per_day_before,
            "expected_trades_per_day_after": self.expected_trades_per_day_after,
            "trades_before": self.trades_before,
            "trades_after": self.trades_after,
        }


@dataclass(frozen=True, slots=True)
class PairAnalysisResult:
    """Full analysis output for one instrument+strategy pair."""

    instrument: str
    strategy: str
    timezone: str
    trades_total: int
    train_metrics: TradeMetrics
    validation_metrics: TradeMetrics
    candidates_hour_local: tuple[BucketCandidate, ...]
    candidates_weekday_local: tuple[BucketCandidate, ...]
    candidates_hour_weekday_local: tuple[BucketCandidate, ...]
    recommendation: PairRecommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "strategy": self.strategy,
            "timezone": self.timezone,
            "trades_total": self.trades_total,
            "train_metrics": self.train_metrics.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
            "candidates": {
                "hour_local": [item.to_dict() for item in self.candidates_hour_local],
                "weekday_local": [item.to_dict() for item in self.candidates_weekday_local],
                "hour_weekday_local": [item.to_dict() for item in self.candidates_hour_weekday_local],
            },
            "recommendation": self.recommendation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LoadedTrades:
    """Raw loaded trades and ingestion counters."""

    trades: tuple[TradeSample, ...]
    skipped_not_closed: int
    skipped_missing_time_source: int
    skipped_parse_errors: int


def resolve_report_dir(*, reports_dir: Path, report: str) -> Path:
    """Resolve report directory by name/path or return latest."""
    report_value = report.strip()
    if report_value.lower() == "latest":
        candidates = [
            path
            for path in reports_dir.iterdir()
            if path.is_dir() and (path / "trades.csv").exists()
        ]
        if not candidates:
            raise FileNotFoundError(f"No reports found in {reports_dir}")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    explicit = Path(report_value)
    if explicit.exists():
        target = explicit
    else:
        target = reports_dir / report_value

    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Report directory not found: {target}")
    if not (target / "trades.csv").exists():
        raise FileNotFoundError(f"trades.csv not found in report directory: {target}")
    return target


def load_closed_trades(
    *,
    trades_csv_path: Path,
    app_config: AppConfig,
    registry: InstrumentRegistry,
    time_source: str,
    instrument_filter: str,
    strategy_filter: str,
    profile_filter: str,
) -> LoadedTrades:
    """Load and normalize closed trades for no-trade analysis."""
    safe_time_source = time_source.strip().lower()
    if safe_time_source not in {"created_at", "activated_at"}:
        raise ValueError("time_source must be 'created_at' or 'activated_at'")

    safe_instrument = instrument_filter.strip().upper()
    safe_strategy = strategy_filter.strip().lower()
    safe_profile = profile_filter.strip().lower()

    skipped_not_closed = 0
    skipped_missing_time_source = 0
    skipped_parse_errors = 0
    out: list[TradeSample] = []

    with trades_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            closed_at = str(row.get("closed_at", "")).strip()
            if not closed_at:
                skipped_not_closed += 1
                continue

            instrument = str(row.get("instrument", "")).strip().upper()
            strategy = str(row.get("strategy", "")).strip()
            profile = str(row.get("profile", "")).strip()
            if safe_instrument and instrument != safe_instrument:
                continue
            if safe_strategy and strategy.lower() != safe_strategy:
                continue
            if safe_profile and profile.lower() != safe_profile:
                continue

            ts_text = str(row.get(safe_time_source, "")).strip()
            if not ts_text:
                skipped_missing_time_source += 1
                continue

            try:
                timestamp_utc = _parse_datetime(ts_text)
                timezone_name = _instrument_timezone_name(
                    app_config=app_config,
                    registry=registry,
                    instrument=instrument,
                )
                timestamp_local = timestamp_utc.astimezone(ZoneInfo(timezone_name))
                net_pnl = _parse_float(row.get("net_pnl"))
                r_multiple = _parse_float(row.get("r_multiple"))
            except Exception:
                skipped_parse_errors += 1
                continue

            out.append(
                TradeSample(
                    profile=profile,
                    strategy=strategy,
                    instrument=instrument,
                    trade_id=str(row.get("trade_id", "")).strip(),
                    timestamp_utc=timestamp_utc,
                    timestamp_local=timestamp_local,
                    local_hour=timestamp_local.hour,
                    local_weekday=timestamp_local.weekday(),
                    local_date=timestamp_local.strftime("%Y-%m-%d"),
                    net_pnl=net_pnl,
                    r_multiple=r_multiple,
                )
            )

    ordered = tuple(sorted(out, key=lambda item: item.timestamp_utc))
    return LoadedTrades(
        trades=ordered,
        skipped_not_closed=skipped_not_closed,
        skipped_missing_time_source=skipped_missing_time_source,
        skipped_parse_errors=skipped_parse_errors,
    )


def analyze_no_trade_pairs(
    *,
    trades: tuple[TradeSample, ...],
    config: AnalysisConfig,
) -> tuple[PairAnalysisResult, ...]:
    """Run no-trade analysis for all instrument+strategy pairs."""
    grouped: dict[tuple[str, str], list[TradeSample]] = {}
    for trade in trades:
        grouped.setdefault((trade.instrument, trade.strategy), []).append(trade)

    results: list[PairAnalysisResult] = []
    for (instrument, strategy), pair_trades in sorted(grouped.items(), key=lambda item: item[0]):
        ordered_pair = sorted(pair_trades, key=lambda item: item.timestamp_utc)
        if len(ordered_pair) < 2:
            continue

        train, validation = split_train_validation(
            trades=ordered_pair,
            validation_ratio=config.validation_ratio,
        )
        if not train or not validation:
            continue

        train_metrics = compute_trade_metrics(train)
        validation_metrics = compute_trade_metrics(validation)

        hour_candidates = _find_confirmed_candidates(
            train_trades=train,
            validation_trades=validation,
            train_baseline=train_metrics,
            validation_baseline=validation_metrics,
            bucket_type="hour_local",
            config=config,
        )
        weekday_candidates = _find_confirmed_candidates(
            train_trades=train,
            validation_trades=validation,
            train_baseline=train_metrics,
            validation_baseline=validation_metrics,
            bucket_type="weekday_local",
            config=config,
        )
        combined_candidates: tuple[BucketCandidate, ...] = tuple()
        if config.include_combined_buckets:
            combined_candidates = _find_confirmed_candidates(
                train_trades=train,
                validation_trades=validation,
                train_baseline=train_metrics,
                validation_baseline=validation_metrics,
                bucket_type="hour_weekday_local",
                config=config,
            )

        recommendation = build_pair_recommendation(
            pair_trades=ordered_pair,
            hour_candidates=hour_candidates,
            weekday_candidates=weekday_candidates,
            min_expected_trades_per_day=config.min_expected_trades_per_day,
        )

        timezone_name = str(ordered_pair[0].timestamp_local.tzinfo or "Europe/Moscow")
        results.append(
            PairAnalysisResult(
                instrument=instrument,
                strategy=strategy,
                timezone=timezone_name,
                trades_total=len(ordered_pair),
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                candidates_hour_local=hour_candidates,
                candidates_weekday_local=weekday_candidates,
                candidates_hour_weekday_local=combined_candidates,
                recommendation=recommendation,
            )
        )

    return tuple(results)


def split_train_validation(
    *,
    trades: list[TradeSample],
    validation_ratio: float,
) -> tuple[list[TradeSample], list[TradeSample]]:
    """Chronological train/validation split."""
    if not trades:
        return [], []
    if len(trades) == 1:
        return trades[:], []

    safe_ratio = min(0.9, max(0.1, float(validation_ratio)))
    val_size = max(1, int(round(len(trades) * safe_ratio)))
    if val_size >= len(trades):
        val_size = max(1, len(trades) // 2)

    split_index = len(trades) - val_size
    if split_index <= 0:
        split_index = 1

    train = trades[:split_index]
    validation = trades[split_index:]
    return train, validation


def compute_trade_metrics(trades: list[TradeSample]) -> TradeMetrics:
    """Calculate key metrics for a list of trades."""
    if not trades:
        return TradeMetrics(
            trades_count=0,
            wins=0,
            losses=0,
            winrate=0.0,
            avg_r=0.0,
            net_pnl=0.0,
            profit_factor=0.0,
            max_loss_streak=0,
        )

    net_values = [item.net_pnl for item in trades]
    wins = sum(1 for value in net_values if value > 0)
    losses = sum(1 for value in net_values if value < 0)
    gross_wins = sum(value for value in net_values if value > 0)
    gross_losses_abs = sum(abs(value) for value in net_values if value < 0)
    if gross_losses_abs > 0:
        profit_factor = gross_wins / gross_losses_abs
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    streak = 0
    max_streak = 0
    for trade in trades:
        if trade.net_pnl < 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    return TradeMetrics(
        trades_count=len(trades),
        wins=wins,
        losses=losses,
        winrate=(wins / len(trades)),
        avg_r=mean(item.r_multiple for item in trades),
        net_pnl=sum(net_values),
        profit_factor=profit_factor,
        max_loss_streak=max_streak,
    )


def build_pair_recommendation(
    *,
    pair_trades: list[TradeSample],
    hour_candidates: tuple[BucketCandidate, ...],
    weekday_candidates: tuple[BucketCandidate, ...],
    min_expected_trades_per_day: float,
) -> PairRecommendation:
    """Build final recommended filters while respecting frequency floor."""
    blocked_hours: list[int] = []
    blocked_weekdays: list[int] = []
    high_risk_filter = False

    before_tpd = _trades_per_day(pair_trades)

    for candidate in sorted(hour_candidates, key=lambda item: item.severity_score, reverse=True):
        hour = int(candidate.bucket_value)
        if hour in blocked_hours:
            continue
        tentative_hours = sorted(blocked_hours + [hour])
        after_tpd = _trades_per_day(
            [
                trade
                for trade in pair_trades
                if trade.local_hour not in tentative_hours and trade.local_weekday not in blocked_weekdays
            ]
        )
        if after_tpd >= min_expected_trades_per_day:
            blocked_hours = tentative_hours
        else:
            high_risk_filter = True

    for candidate in sorted(weekday_candidates, key=lambda item: item.severity_score, reverse=True):
        weekday = int(candidate.bucket_value)
        if weekday in blocked_weekdays:
            continue
        tentative_weekdays = sorted(blocked_weekdays + [weekday])
        after_tpd = _trades_per_day(
            [
                trade
                for trade in pair_trades
                if trade.local_hour not in blocked_hours and trade.local_weekday not in tentative_weekdays
            ]
        )
        if after_tpd >= min_expected_trades_per_day:
            blocked_weekdays = tentative_weekdays
        else:
            high_risk_filter = True

    filtered_trades = [
        trade
        for trade in pair_trades
        if trade.local_hour not in blocked_hours and trade.local_weekday not in blocked_weekdays
    ]
    after_tpd = _trades_per_day(filtered_trades)

    return PairRecommendation(
        blocked_entry_hours_local=tuple(blocked_hours),
        blocked_entry_weekdays_local=tuple(blocked_weekdays),
        blocked_time_windows_local=tuple(),
        high_risk_filter=high_risk_filter,
        expected_trades_per_day_before=before_tpd,
        expected_trades_per_day_after=after_tpd,
        trades_before=len(pair_trades),
        trades_after=len(filtered_trades),
    )


def build_no_trade_patch_payload(results: tuple[PairAnalysisResult, ...]) -> dict[str, Any]:
    """Build config patch payload for strategy_params.by_instrument."""
    by_instrument: dict[str, dict[str, Any]] = {}
    for result in results:
        recommendation = result.recommendation
        if (
            not recommendation.blocked_entry_hours_local
            and not recommendation.blocked_entry_weekdays_local
            and not recommendation.blocked_time_windows_local
            and not recommendation.high_risk_filter
        ):
            continue

        strategy_payload: dict[str, Any] = {
            "blocked_entry_hours_local": list(recommendation.blocked_entry_hours_local),
            "blocked_entry_weekdays_local": list(recommendation.blocked_entry_weekdays_local),
        }
        if recommendation.blocked_time_windows_local:
            strategy_payload["blocked_time_windows_local"] = list(
                recommendation.blocked_time_windows_local
            )
        if recommendation.high_risk_filter:
            strategy_payload["high_risk_filter"] = True

        by_instrument.setdefault(result.instrument, {})[result.strategy] = strategy_payload

    return {
        "strategy_params": {
            "by_instrument": by_instrument,
        }
    }


def weekday_name(weekday: int) -> str:
    names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    if 0 <= weekday < len(names):
        return names[weekday]
    return str(weekday)


def render_bucket_label(bucket_type: str, bucket_value: int | str) -> str:
    if bucket_type == "hour_local":
        hour = int(bucket_value)
        return f"{hour:02d}:00-{hour:02d}:59"
    if bucket_type == "weekday_local":
        weekday = int(bucket_value)
        return weekday_name(weekday)
    if bucket_type == "hour_weekday_local":
        text = str(bucket_value)
        if "-" in text:
            weekday_text, hour_text = text.split("-", 1)
            try:
                weekday = int(weekday_text)
                hour = int(hour_text)
                return f"{weekday_name(weekday)} {hour:02d}:00-{hour:02d}:59"
            except ValueError:
                return text
    return str(bucket_value)


def _find_confirmed_candidates(
    *,
    train_trades: list[TradeSample],
    validation_trades: list[TradeSample],
    train_baseline: TradeMetrics,
    validation_baseline: TradeMetrics,
    bucket_type: str,
    config: AnalysisConfig,
) -> tuple[BucketCandidate, ...]:
    train_buckets = _bucketize(trades=train_trades, bucket_type=bucket_type)
    validation_buckets = _bucketize(trades=validation_trades, bucket_type=bucket_type)

    out: list[BucketCandidate] = []
    for bucket_value, train_bucket_trades in train_buckets.items():
        validation_bucket_trades = validation_buckets.get(bucket_value, [])
        train_metrics = compute_trade_metrics(train_bucket_trades)
        validation_metrics = compute_trade_metrics(validation_bucket_trades)
        train_bad, train_reasons, train_gap = _is_bad_bucket(
            bucket_metrics=train_metrics,
            baseline=train_baseline,
            config=config,
        )
        validation_bad, validation_reasons, validation_gap = _is_bad_bucket(
            bucket_metrics=validation_metrics,
            baseline=validation_baseline,
            config=config,
        )
        if not train_bad or not validation_bad:
            continue

        reasons = tuple(sorted(set(train_reasons + validation_reasons)))
        severity = (
            max(0.0, train_gap) + max(0.0, validation_gap)
            + max(0.0, -train_metrics.avg_r) * 100.0
            + max(0.0, -validation_metrics.avg_r) * 100.0
        ) / 2.0

        out.append(
            BucketCandidate(
                bucket_type=bucket_type,
                bucket_value=bucket_value,
                severity_score=severity,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                train_winrate_gap_pp=train_gap,
                validation_winrate_gap_pp=validation_gap,
                reasons=reasons,
            )
        )

    return tuple(sorted(out, key=lambda item: item.severity_score, reverse=True))


def _is_bad_bucket(
    *,
    bucket_metrics: TradeMetrics,
    baseline: TradeMetrics,
    config: AnalysisConfig,
) -> tuple[bool, tuple[str, ...], float]:
    if bucket_metrics.trades_count < config.min_trades_per_bucket:
        return False, tuple(), 0.0

    winrate_gap_pp = (baseline.winrate - bucket_metrics.winrate) * 100.0
    worse_winrate = winrate_gap_pp >= config.min_winrate_gap_pp
    negative_avg_r = bucket_metrics.avg_r <= config.min_negative_avg_r
    negative_net_pnl = bucket_metrics.net_pnl < 0

    bad = (worse_winrate and (negative_avg_r or negative_net_pnl)) or (
        negative_avg_r and negative_net_pnl
    )
    reasons: list[str] = []
    if worse_winrate:
        reasons.append("winrate_below_baseline")
    if negative_avg_r:
        reasons.append("negative_avg_r")
    if negative_net_pnl:
        reasons.append("negative_net_pnl")
    return bad, tuple(reasons), winrate_gap_pp


def _bucketize(
    *,
    trades: list[TradeSample],
    bucket_type: str,
) -> dict[int | str, list[TradeSample]]:
    out: dict[int | str, list[TradeSample]] = {}
    for trade in trades:
        if bucket_type == "hour_local":
            key: int | str = trade.local_hour
        elif bucket_type == "weekday_local":
            key = trade.local_weekday
        elif bucket_type == "hour_weekday_local":
            key = f"{trade.local_weekday}-{trade.local_hour}"
        else:
            raise ValueError(f"Unsupported bucket_type: {bucket_type}")
        out.setdefault(key, []).append(trade)
    return out


def _trades_per_day(trades: list[TradeSample]) -> float:
    if not trades:
        return 0.0
    days = {trade.local_date for trade in trades}
    if not days:
        return 0.0
    return len(trades) / len(days)


def _instrument_timezone_name(
    *,
    app_config: AppConfig,
    registry: InstrumentRegistry,
    instrument: str,
) -> str:
    if instrument not in registry:
        return str(app_config.params.get("timezone", "Europe/Moscow"))
    meta: InstrumentMeta = registry.get(instrument)
    if not meta.sessions:
        return str(app_config.params.get("timezone", "Europe/Moscow"))
    return meta.sessions[0].timezone


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("datetime is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def _parse_float(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    return float(text)
