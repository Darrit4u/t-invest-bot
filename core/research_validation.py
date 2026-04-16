"""Strategy research and validation helpers (Stage C).

This module extends the existing backtest pipeline with:
- train/validation/test split orchestration
- walk-forward validation
- parameter sensitivity runs
- portfolio-level analytics for robustness and overfitting checks
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any, Callable, Iterable, Sequence

from core.backtest_matrix import PortfolioRunResult, run_portfolio_backtest
from core.config_loader import AppConfig
from core.market_data import Candle
from core.models import Trade

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Chronological split definition for train/validation/test."""

    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Rolling walk-forward window definition in bars."""

    train_bars: int = 240
    test_bars: int = 60
    step_bars: int = 60
    min_folds: int = 1


@dataclass(frozen=True, slots=True)
class SensitivityConfig:
    """Sensitivity grid settings."""

    param_grid: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    target_metric: str = "net_pnl"
    use_test_window: bool = True


ResearchRunner = Callable[..., PortfolioRunResult]


@dataclass(slots=True)
class ResearchConfig:
    """Input bundle for Stage C research functions."""

    profile: str
    candles_by_instrument: dict[str, list[Candle]]
    params: dict[str, Any]
    timeframe: str
    app_config: AppConfig | None = None
    report_start_utc: datetime | None = None
    report_end_utc: datetime | None = None
    selected_instruments: tuple[str, ...] = ()
    selected_strategies: tuple[str, ...] = ()
    initial_capital: float = 100_000.0
    split: SplitConfig = field(default_factory=SplitConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    runner: ResearchRunner | None = None


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Closed datetime interval [start_utc, end_utc]."""

    start_utc: datetime
    end_utc: datetime


def run_backtest(config: ResearchConfig) -> dict[str, Any]:
    """Run train/validation/test backtests and derive overfitting indicators."""

    timestamps = _selected_timestamps(config)
    split_windows = _build_split_windows(timestamps=timestamps, split=config.split)

    segments: dict[str, dict[str, Any]] = {}
    all_closed_rows: list[dict[str, Any]] = []
    for name in ("train", "validation", "test"):
        window = split_windows[name]
        result = _execute_window(config=config, window=window, params=config.params)
        analysis = analyze_portfolio(result, initial_capital=config.initial_capital)
        all_closed_rows.extend(_extract_closed_trade_rows(result))
        segments[name] = {
            "window": _window_payload(window),
            "result": result,
            "analysis": analysis,
        }

    train_metrics = segments["train"]["analysis"]["portfolio_metrics"]
    validation_metrics = segments["validation"]["analysis"]["portfolio_metrics"]
    test_metrics = segments["test"]["analysis"]["portfolio_metrics"]
    train_net = float(train_metrics.get("net_pnl", 0.0))
    validation_net = float(validation_metrics.get("net_pnl", 0.0))
    test_net = float(test_metrics.get("net_pnl", 0.0))
    train_test_ratio = (
        (train_net / max(abs(test_net), _EPS))
        if abs(test_net) > _EPS
        else (math.inf if abs(train_net) > _EPS else 1.0)
    )
    stability_score = _stability_score((train_net, validation_net, test_net))
    generalization_gap = train_net - test_net

    overfitted = bool(
        (train_test_ratio > 2.5 and generalization_gap > 0.0 and test_net <= 0.0)
        or stability_score < 0.35
    )
    if overfitted:
        strategy_bucket = "overfitted"
    elif test_net <= 0.0 or float(test_metrics.get("profit_factor", 0.0)) < 1.0 or stability_score < 0.55:
        strategy_bucket = "unstable"
    else:
        strategy_bucket = "working"

    return {
        "split_windows": {key: _window_payload(value) for key, value in split_windows.items()},
        "segments": segments,
        "portfolio_analysis_all": analyze_portfolio(
            all_closed_rows,
            initial_capital=config.initial_capital,
        ),
        "overfitting": {
            "train_test_ratio": train_test_ratio,
            "generalization_gap": generalization_gap,
            "stability_score": stability_score,
            "overfitted": overfitted,
        },
        "strategy_bucket": strategy_bucket,
    }


def run_walk_forward(config: ResearchConfig) -> dict[str, Any]:
    """Run rolling walk-forward train->test validation."""

    timestamps = _selected_timestamps(config)
    folds = _build_walk_forward_folds(timestamps=timestamps, walk_forward=config.walk_forward)

    rows: list[dict[str, Any]] = []
    test_net_values: list[float] = []
    overfitting_fold_count = 0

    for index, fold in enumerate(folds, start=1):
        train_result = _execute_window(config=config, window=fold["train"], params=config.params)
        test_result = _execute_window(config=config, window=fold["test"], params=config.params)
        train_analysis = analyze_portfolio(train_result, initial_capital=config.initial_capital)
        test_analysis = analyze_portfolio(test_result, initial_capital=config.initial_capital)
        train_net = float(train_analysis["portfolio_metrics"].get("net_pnl", 0.0))
        test_net = float(test_analysis["portfolio_metrics"].get("net_pnl", 0.0))
        test_net_values.append(test_net)
        train_test_ratio = (
            (train_net / max(abs(test_net), _EPS))
            if abs(test_net) > _EPS
            else (math.inf if abs(train_net) > _EPS else 1.0)
        )
        fold_overfit = bool(train_test_ratio > 2.5 and train_net > 0.0 and test_net <= 0.0)
        if fold_overfit:
            overfitting_fold_count += 1

        rows.append(
            {
                "fold": index,
                "train_window": _window_payload(fold["train"]),
                "test_window": _window_payload(fold["test"]),
                "train_metrics": train_analysis["portfolio_metrics"],
                "test_metrics": test_analysis["portfolio_metrics"],
                "train_test_ratio": train_test_ratio,
                "overfit_flag": fold_overfit,
            }
        )

    distribution = _distribution_summary(test_net_values)
    return {
        "folds": rows,
        "summary": {
            "folds": len(rows),
            "min_required_folds": config.walk_forward.min_folds,
            "test_distribution": distribution,
            "stability_score": _stability_score(test_net_values),
            "positive_fold_ratio": _positive_ratio(test_net_values),
            "overfitting_fold_count": overfitting_fold_count,
        },
    }


def run_sensitivity(config: ResearchConfig) -> dict[str, Any]:
    """Run grid sensitivity for selected parameters."""

    grid = {key: tuple(values) for key, values in config.sensitivity.param_grid.items() if tuple(values)}
    if not grid:
        return {
            "rows": [],
            "summary": {
                "target_metric": config.sensitivity.target_metric,
                "runs": 0,
                "parameter_robustness_score": 0.0,
            },
        }

    timestamps = _selected_timestamps(config)
    if config.sensitivity.use_test_window:
        window = _build_split_windows(timestamps=timestamps, split=config.split)["test"]
    else:
        window = TimeWindow(start_utc=timestamps[0], end_utc=timestamps[-1])

    keys = sorted(grid.keys())
    rows: list[dict[str, Any]] = []
    metric_values: list[float] = []

    for values in product(*(grid[key] for key in keys)):
        overrides = {key: value for key, value in zip(keys, values)}
        params_variant = _params_with_overrides(config.params, overrides)
        result = _execute_window(config=config, window=window, params=params_variant)
        analysis = analyze_portfolio(result, initial_capital=config.initial_capital)
        metric_value = float(analysis["portfolio_metrics"].get(config.sensitivity.target_metric, 0.0))
        metric_values.append(metric_value)
        rows.append(
            {
                "params": overrides,
                "window": _window_payload(window),
                "metrics": analysis["portfolio_metrics"],
                "distribution": analysis["distribution"],
                "contribution": analysis["contribution"],
                "target_metric_value": metric_value,
            }
        )

    rows.sort(key=lambda item: float(item["target_metric_value"]), reverse=True)
    return {
        "rows": rows,
        "summary": {
            "target_metric": config.sensitivity.target_metric,
            "runs": len(rows),
            "best_params": rows[0]["params"],
            "best_metric_value": float(rows[0]["target_metric_value"]),
            "median_metric_value": statistics.median(metric_values),
            "positive_ratio": _positive_ratio(metric_values),
            "parameter_robustness_score": _parameter_robustness_score(metric_values),
        },
    }


def analyze_portfolio(
    results: PortfolioRunResult | dict[str, Any] | Sequence[Trade | dict[str, Any]],
    *,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Compute portfolio-level metrics and robustness diagnostics."""

    closed_rows = _extract_closed_trade_rows(results)
    normalized = [_normalize_closed_trade_row(item) for item in closed_rows]
    normalized = [item for item in normalized if item is not None]
    trades = sorted(normalized, key=lambda item: item["closed_at"])

    pnl_values = [float(item["net_pnl"]) for item in trades]
    trade_count = len(trades)
    wins = sum(1 for pnl in pnl_values if pnl >= 0.0)
    losses = trade_count - wins
    net_pnl = sum(pnl_values)
    gross_wins = sum(value for value in pnl_values if value >= 0.0)
    gross_losses_abs = abs(sum(value for value in pnl_values if value < 0.0))
    win_rate = (wins / trade_count) if trade_count else 0.0
    if gross_losses_abs > _EPS:
        profit_factor = gross_wins / gross_losses_abs
    elif gross_wins > _EPS:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    expectancy = (net_pnl / trade_count) if trade_count else 0.0
    average_win = (gross_wins / wins) if wins else 0.0
    average_loss = ((-gross_losses_abs) / losses) if losses else 0.0
    max_loss_streak = _max_loss_streak(pnl_values)
    average_holding_hours = _average_holding_hours(trades)
    trading_days = _trading_days(trades)
    turnover = (trade_count / trading_days) if trading_days > 0 else 0.0

    drawdown = _equity_and_drawdown(pnl_values)
    sharpe, sortino = _risk_adjusted_ratios(
        pnl_values=pnl_values,
        initial_capital=initial_capital,
        turnover=turnover,
    )
    cagr = _cagr(
        net_pnl=net_pnl,
        initial_capital=initial_capital,
        trading_days=trading_days,
    )

    contribution = _contribution(trades=trades, total_net=net_pnl)
    correlation = _correlation_stability(trades)
    exposure = _exposure(trades)

    return {
        "portfolio_metrics": {
            "trades": trade_count,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "gross_wins": gross_wins,
            "gross_losses_abs": gross_losses_abs,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "average_win": average_win,
            "average_loss": average_loss,
            "max_losing_streak": max_loss_streak,
            "max_drawdown": float(drawdown["max_drawdown"]),
            "max_drawdown_pct": float(drawdown["max_drawdown_pct"]),
            "max_recovery_trades": int(drawdown["max_recovery_trades"]),
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "turnover": turnover,
            "trading_days": trading_days,
            "average_holding_hours": average_holding_hours,
            "exposure_ratio": float(exposure["total_exposure_ratio"]),
        },
        "distribution": {
            **_distribution_summary(pnl_values),
            "skew": _skew(pnl_values),
            "kurtosis": _kurtosis_excess(pnl_values),
            "tail_p05": _percentile(pnl_values, 0.05),
            "tail_p95": _percentile(pnl_values, 0.95),
            "max_losing_streak": max_loss_streak,
        },
        "equity": {
            "curve": drawdown["equity_curve"],
            "drawdown_curve": drawdown["drawdown_curve"],
        },
        "contribution": contribution,
        "correlation_stability": correlation,
        "exposure": exposure,
    }


def _selected_timestamps(config: ResearchConfig) -> list[datetime]:
    instrument_scope = (
        set(config.selected_instruments) if config.selected_instruments else set(config.candles_by_instrument)
    )
    timestamps: set[datetime] = set()
    for instrument, candles in config.candles_by_instrument.items():
        if instrument_scope and instrument not in instrument_scope:
            continue
        for candle in candles:
            if config.report_start_utc is not None and candle.datetime < config.report_start_utc:
                continue
            if config.report_end_utc is not None and candle.datetime > config.report_end_utc:
                continue
            timestamps.add(_aware(candle.datetime))
    ordered = sorted(timestamps)
    if len(ordered) < 6:
        raise ValueError("Not enough candles in selected range for research splits")
    return ordered


def _build_split_windows(*, timestamps: list[datetime], split: SplitConfig) -> dict[str, TimeWindow]:
    counts = _split_counts(
        total=len(timestamps),
        weights=(float(split.train_ratio), float(split.validation_ratio), float(split.test_ratio)),
    )
    train_n, validation_n, test_n = counts

    train_end = train_n - 1
    validation_end = train_n + validation_n - 1
    test_end = train_n + validation_n + test_n - 1

    return {
        "train": TimeWindow(start_utc=timestamps[0], end_utc=timestamps[train_end]),
        "validation": TimeWindow(
            start_utc=timestamps[train_end + 1],
            end_utc=timestamps[validation_end],
        ),
        "test": TimeWindow(
            start_utc=timestamps[validation_end + 1],
            end_utc=timestamps[test_end],
        ),
    }


def _build_walk_forward_folds(
    *,
    timestamps: list[datetime],
    walk_forward: WalkForwardConfig,
) -> list[dict[str, TimeWindow]]:
    train_bars = max(3, int(walk_forward.train_bars))
    test_bars = max(1, int(walk_forward.test_bars))
    step_bars = max(1, int(walk_forward.step_bars))
    total = len(timestamps)
    folds: list[dict[str, TimeWindow]] = []

    start = 0
    while (start + train_bars + test_bars) <= total:
        train_start = start
        train_end = start + train_bars - 1
        test_start = train_end + 1
        test_end = test_start + test_bars - 1
        folds.append(
            {
                "train": TimeWindow(start_utc=timestamps[train_start], end_utc=timestamps[train_end]),
                "test": TimeWindow(start_utc=timestamps[test_start], end_utc=timestamps[test_end]),
            }
        )
        start += step_bars

    if len(folds) < max(1, int(walk_forward.min_folds)):
        raise ValueError(
            f"Not enough candles for walk-forward: got {len(folds)} fold(s), "
            f"required >= {walk_forward.min_folds}"
        )
    return folds


def _execute_window(*, config: ResearchConfig, window: TimeWindow, params: dict[str, Any]) -> PortfolioRunResult:
    runner = config.runner or run_portfolio_backtest
    if config.runner is None and config.app_config is None:
        raise ValueError("ResearchConfig.app_config is required when custom runner is not provided")
    return runner(
        profile=config.profile,
        candles_by_instrument=config.candles_by_instrument,
        app_config=config.app_config,
        params=params,
        timeframe=config.timeframe,
        report_start_utc=window.start_utc,
        report_end_utc=window.end_utc,
        selected_instruments=list(config.selected_instruments) if config.selected_instruments else None,
        selected_strategies=list(config.selected_strategies) if config.selected_strategies else None,
    )


def _window_payload(window: TimeWindow) -> dict[str, str]:
    return {
        "start_utc": window.start_utc.isoformat(),
        "end_utc": window.end_utc.isoformat(),
    }


def _extract_closed_trade_rows(
    results: PortfolioRunResult | dict[str, Any] | Sequence[Trade | dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(results, PortfolioRunResult):
        source = list(results.trades)
    elif isinstance(results, dict):
        source = list(results.get("trades", []))
    else:
        source = list(results)

    out: list[dict[str, Any]] = []
    for item in source:
        if isinstance(item, Trade):
            if item.closed_at is None:
                continue
            out.append(
                {
                    "trade_id": item.trade_id or "",
                    "instrument": item.instrument,
                    "strategy": item.strategy_id,
                    "opened_at": (item.activated_at or item.opened_at).isoformat(),
                    "closed_at": item.closed_at.isoformat(),
                    "net_pnl": float(item.pnl),
                    "r_multiple": float(item.r_multiple) if item.r_multiple is not None else 0.0,
                }
            )
            continue
        if isinstance(item, dict):
            closed_raw = item.get("closed_at", "")
            if not str(closed_raw).strip():
                continue
            out.append(dict(item))
    return out


def _normalize_closed_trade_row(row: dict[str, Any]) -> dict[str, Any] | None:
    closed_raw = row.get("closed_at")
    if not closed_raw:
        return None
    opened_raw = row.get("activated_at") or row.get("opened_at") or row.get("created_at")
    try:
        closed_at = _parse_dt(closed_raw)
        opened_at = _parse_dt(opened_raw) if opened_raw else closed_at
    except ValueError:
        return None

    instrument = str(row.get("instrument", "")).strip()
    strategy = str(row.get("strategy", "") or row.get("strategy_id", "")).strip()
    if not strategy:
        strategy = "unknown"
    if not instrument:
        instrument = "unknown"

    return {
        "trade_id": str(row.get("trade_id", "")).strip(),
        "instrument": instrument,
        "strategy": strategy,
        "opened_at": opened_at,
        "closed_at": closed_at,
        "net_pnl": float(row.get("net_pnl", row.get("pnl", 0.0)) or 0.0),
        "r_multiple": float(row.get("r_multiple", 0.0) or 0.0),
    }


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty datetime")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported datetime value: {value!r}") from exc
    return _aware(parsed)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _split_counts(*, total: int, weights: tuple[float, float, float]) -> tuple[int, int, int]:
    if total < 6:
        raise ValueError("Need at least 6 timestamps for train/validation/test split")
    raw = [max(0.0, float(item)) for item in weights]
    weight_sum = sum(raw)
    if weight_sum <= _EPS:
        raw = [0.6, 0.2, 0.2]
        weight_sum = 1.0
    normalized = [item / weight_sum for item in raw]

    counts = [max(1, int(total * ratio)) for ratio in normalized]
    while sum(counts) > total:
        idx = counts.index(max(counts))
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break
    while sum(counts) < total:
        idx = counts.index(max(counts))
        counts[idx] += 1

    if any(item <= 0 for item in counts):
        raise ValueError("Split produced empty segment")
    return counts[0], counts[1], counts[2]


def _params_with_overrides(params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(params)
    for path, value in overrides.items():
        _set_nested(output, path=path, value=value)
    return output


def _set_nested(payload: dict[str, Any], *, path: str, value: Any) -> None:
    keys = [item.strip() for item in str(path).split(".") if item.strip()]
    if not keys:
        return
    current: Any = payload
    for key in keys[:-1]:
        if not isinstance(current, dict):
            return
        next_item = current.get(key)
        if not isinstance(next_item, dict):
            next_item = {}
            current[key] = next_item
        current = next_item
    if isinstance(current, dict):
        current[keys[-1]] = value


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_ratio": 0.0,
        }
    ordered = sorted(float(item) for item in values)
    return {
        "count": float(len(ordered)),
        "mean": float(statistics.mean(ordered)),
        "median": float(statistics.median(ordered)),
        "stdev": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "positive_ratio": _positive_ratio(ordered),
    }


def _positive_ratio(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    positives = sum(1 for item in values if float(item) > 0.0)
    return positives / len(values)


def _stability_score(values: Iterable[float]) -> float:
    sample = [float(item) for item in values]
    if not sample:
        return 0.0
    if len(sample) == 1:
        return 1.0
    mean_abs = statistics.mean(abs(item) for item in sample)
    dispersion = statistics.pstdev(sample)
    if mean_abs <= _EPS and dispersion <= _EPS:
        return 1.0
    normalized = dispersion / max(mean_abs, _EPS)
    score = 1.0 / (1.0 + normalized)
    return max(0.0, min(1.0, score))


def _parameter_robustness_score(values: Sequence[float]) -> float:
    sample = [float(item) for item in values]
    if not sample:
        return 0.0
    positive_ratio = _positive_ratio(sample)
    spread = max(sample) - min(sample)
    median_abs = abs(statistics.median(sample))
    peak_penalty = spread / max(median_abs, 1.0)
    raw = positive_ratio * (1.0 / (1.0 + peak_penalty))
    return max(0.0, min(1.0, raw))


def _max_loss_streak(pnl_values: Sequence[float]) -> int:
    streak = 0
    max_streak = 0
    for value in pnl_values:
        if float(value) < 0.0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _average_holding_hours(trades: Sequence[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    durations = []
    for item in trades:
        opened_at = item["opened_at"]
        closed_at = item["closed_at"]
        durations.append(max(0.0, (closed_at - opened_at).total_seconds()) / 3600.0)
    return statistics.mean(durations) if durations else 0.0


def _trading_days(trades: Sequence[dict[str, Any]]) -> int:
    if not trades:
        return 0
    start = min(item["opened_at"] for item in trades)
    end = max(item["closed_at"] for item in trades)
    return max(1, (end.date() - start.date()).days + 1)


def _equity_and_drawdown(pnl_values: Sequence[float]) -> dict[str, Any]:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    drawdown_curve: list[float] = []
    equity_curve: list[float] = []
    drawdown_start: int | None = None
    max_recovery_trades = 0

    for index, pnl in enumerate(pnl_values):
        equity += float(pnl)
        equity_curve.append(equity)
        if equity >= peak:
            if drawdown_start is not None:
                max_recovery_trades = max(max_recovery_trades, index - drawdown_start)
                drawdown_start = None
            peak = equity
            drawdown_curve.append(0.0)
            continue

        if drawdown_start is None:
            drawdown_start = index
        drawdown = peak - equity
        drawdown_curve.append(drawdown)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if peak > _EPS:
            max_drawdown_pct = max(max_drawdown_pct, drawdown / peak)

    if drawdown_start is not None:
        max_recovery_trades = max(max_recovery_trades, len(pnl_values) - drawdown_start)

    return {
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "max_recovery_trades": max_recovery_trades,
    }


def _risk_adjusted_ratios(*, pnl_values: Sequence[float], initial_capital: float, turnover: float) -> tuple[float, float]:
    if not pnl_values or initial_capital <= _EPS:
        return 0.0, 0.0
    returns = [float(item) / initial_capital for item in pnl_values]
    if len(returns) < 2:
        return 0.0, 0.0
    mean_ret = statistics.mean(returns)
    stdev = statistics.pstdev(returns)
    downside = [item for item in returns if item < 0.0]
    downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    annual_factor = math.sqrt(max(1.0, turnover * 252.0))
    sharpe = (mean_ret / stdev * annual_factor) if stdev > _EPS else 0.0
    sortino = (mean_ret / downside_dev * annual_factor) if downside_dev > _EPS else 0.0
    return sharpe, sortino


def _cagr(*, net_pnl: float, initial_capital: float, trading_days: int) -> float:
    if initial_capital <= _EPS or trading_days <= 0:
        return 0.0
    final_equity = initial_capital + float(net_pnl)
    if final_equity <= _EPS:
        return -1.0
    years = trading_days / 365.0
    if years <= _EPS:
        return 0.0
    return (final_equity / initial_capital) ** (1.0 / years) - 1.0


def _aggregate_bucket(trades: Sequence[dict[str, Any]], *, key: str, total_net: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in trades:
        bucket = str(item.get(key, "unknown"))
        grouped.setdefault(bucket, []).append(item)

    output: dict[str, dict[str, Any]] = {}
    for bucket, rows in sorted(grouped.items()):
        pnl_values = [float(item["net_pnl"]) for item in rows]
        net = sum(pnl_values)
        wins = sum(1 for value in pnl_values if value >= 0.0)
        losses = len(rows) - wins
        output[bucket] = {
            "trades": len(rows),
            "net_pnl": net,
            "win_rate": (wins / len(rows)) if rows else 0.0,
            "expectancy": (net / len(rows)) if rows else 0.0,
            "drawdown_proxy": abs(sum(value for value in pnl_values if value < 0.0)),
            "share_of_total_net": (net / total_net) if abs(total_net) > _EPS else 0.0,
            "wins": wins,
            "losses": losses,
        }
    return output


def _remove_one_table(by_bucket: dict[str, dict[str, Any]], *, total_net: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, metrics in by_bucket.items():
        bucket_net = float(metrics.get("net_pnl", 0.0))
        rows.append(
            {
                "bucket": bucket,
                "bucket_net_pnl": bucket_net,
                "portfolio_net_without_bucket": total_net - bucket_net,
            }
        )
    rows.sort(key=lambda item: float(item["bucket_net_pnl"]))
    return rows


def _contribution(*, trades: Sequence[dict[str, Any]], total_net: float) -> dict[str, Any]:
    by_strategy = _aggregate_bucket(trades, key="strategy", total_net=total_net)
    by_instrument = _aggregate_bucket(trades, key="instrument", total_net=total_net)
    return {
        "by_strategy": by_strategy,
        "by_instrument": by_instrument,
        "remove_one_strategy": _remove_one_table(by_strategy, total_net=total_net),
        "remove_one_instrument": _remove_one_table(by_instrument, total_net=total_net),
    }


def _exposure(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "total_exposure_ratio": 0.0,
            "by_instrument": {},
            "by_strategy": {},
        }
    period_start = min(item["opened_at"] for item in trades)
    period_end = max(item["closed_at"] for item in trades)
    period_seconds = max(_EPS, (period_end - period_start).total_seconds())

    hold_total = 0.0
    hold_by_instrument: dict[str, float] = {}
    hold_by_strategy: dict[str, float] = {}
    for item in trades:
        duration = max(0.0, (item["closed_at"] - item["opened_at"]).total_seconds())
        hold_total += duration
        hold_by_instrument[item["instrument"]] = hold_by_instrument.get(item["instrument"], 0.0) + duration
        hold_by_strategy[item["strategy"]] = hold_by_strategy.get(item["strategy"], 0.0) + duration

    by_instrument = {
        key: {
            "holding_seconds": value,
            "share_of_holding_time": value / max(hold_total, _EPS),
        }
        for key, value in sorted(hold_by_instrument.items())
    }
    by_strategy = {
        key: {
            "holding_seconds": value,
            "share_of_holding_time": value / max(hold_total, _EPS),
        }
        for key, value in sorted(hold_by_strategy.items())
    }
    return {
        "total_exposure_ratio": hold_total / period_seconds,
        "by_instrument": by_instrument,
        "by_strategy": by_strategy,
    }


def _correlation_stability(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "pairwise_corr": {},
            "half_split_stability": {},
            "mean_abs_corr_drift": 0.0,
        }
    dates = sorted({item["closed_at"].date() for item in trades})
    instruments = sorted({item["instrument"] for item in trades})
    daily_by_instrument: dict[str, dict[Any, float]] = {item: {} for item in instruments}
    for trade in trades:
        key = trade["instrument"]
        day = trade["closed_at"].date()
        daily = daily_by_instrument[key]
        daily[day] = daily.get(day, 0.0) + float(trade["net_pnl"])

    def _series_for(instrument: str, day_list: list[Any]) -> list[float]:
        rows = daily_by_instrument.get(instrument, {})
        return [float(rows.get(day, 0.0)) for day in day_list]

    pairwise_corr: dict[str, float] = {}
    pairwise_drift: dict[str, float] = {}
    midpoint = max(1, len(dates) // 2)
    left_days = dates[:midpoint]
    right_days = dates[midpoint:] or dates[midpoint - 1 :]
    drift_values: list[float] = []

    for idx, left in enumerate(instruments):
        for right in instruments[idx + 1 :]:
            key = f"{left}__{right}"
            full_corr = _pearson(_series_for(left, dates), _series_for(right, dates))
            left_corr = _pearson(_series_for(left, left_days), _series_for(right, left_days))
            right_corr = _pearson(_series_for(left, right_days), _series_for(right, right_days))
            drift = abs(left_corr - right_corr)
            drift_values.append(drift)
            pairwise_corr[key] = full_corr
            pairwise_drift[key] = drift

    return {
        "pairwise_corr": pairwise_corr,
        "half_split_stability": pairwise_drift,
        "mean_abs_corr_drift": statistics.mean(drift_values) if drift_values else 0.0,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom <= _EPS:
        return 0.0
    return cov / denom


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    clamped = max(0.0, min(1.0, float(q)))
    index = clamped * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _skew(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = statistics.mean(values)
    centered = [value - mean for value in values]
    m2 = statistics.mean(value * value for value in centered)
    if m2 <= _EPS:
        return 0.0
    m3 = statistics.mean(value**3 for value in centered)
    return m3 / (m2 ** 1.5)


def _kurtosis_excess(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    mean = statistics.mean(values)
    centered = [value - mean for value in values]
    m2 = statistics.mean(value * value for value in centered)
    if m2 <= _EPS:
        return 0.0
    m4 = statistics.mean(value**4 for value in centered)
    return (m4 / (m2**2)) - 3.0
