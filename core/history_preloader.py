"""Historical candle preloading for fast warmup after restart."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from core.instrument_registry import InstrumentRegistry
from core.market_data import Candle, CandleValidationError, _load_tinvest_sdk, _quotation_to_float
from core.strategy_params import iter_strategy_param_variants
from core.timeframes import map_tinvest_history_interval, timeframe_minutes
from storage.memory_store import MemoryCandleStore


@dataclass(frozen=True, slots=True)
class PreloadReport:
    """Summary of preload operation."""

    enabled: bool
    requested_bars: int
    processed_candles: int
    inserted: int
    updated: int
    ignored: int
    instruments_attempted: int
    instruments_with_data: int


async def preload_history(
    *,
    token: str,
    registry: InstrumentRegistry,
    store: MemoryCandleStore,
    params: dict[str, Any],
    timeframe: str,
    logger: Any,
) -> PreloadReport:
    cfg = params.get("history_preload", {})
    if not isinstance(cfg, dict):
        cfg = {}
    enabled = _as_bool(cfg.get("enabled", True))
    if not enabled:
        return PreloadReport(
            enabled=False,
            requested_bars=0,
            processed_candles=0,
            inserted=0,
            updated=0,
            ignored=0,
            instruments_attempted=0,
            instruments_with_data=0,
        )

    if not token.strip():
        logger.warning("History preload skipped: INVEST_TOKEN is empty")
        return PreloadReport(
            enabled=True,
            requested_bars=0,
            processed_candles=0,
            inserted=0,
            updated=0,
            ignored=0,
            instruments_attempted=0,
            instruments_with_data=0,
        )

    sdk = _load_tinvest_sdk()
    if sdk is None:
        logger.warning("History preload skipped: T-Invest SDK is unavailable")
        return PreloadReport(
            enabled=True,
            requested_bars=0,
            processed_candles=0,
            inserted=0,
            updated=0,
            ignored=0,
            instruments_attempted=0,
            instruments_with_data=0,
        )

    bars_override = int(cfg.get("bars", 0))
    required = bars_override if bars_override > 0 else estimate_required_bars(params=params, timeframe=timeframe)
    extra = max(0, int(cfg.get("extra_bars", 30)))
    requested_bars = max(1, required + (0 if bars_override > 0 else extra))
    limit_multiplier = max(1, int(cfg.get("request_limit_multiplier", 3)))
    lookback_start_multiplier = max(1.0, float(cfg.get("lookback_start_multiplier", 1.5)))
    lookback_growth_factor = max(1.1, float(cfg.get("lookback_growth_factor", 2.0)))
    max_lookback_multiplier = max(
        lookback_start_multiplier,
        float(cfg.get("max_lookback_multiplier", 64.0)),
    )
    max_fetch_rounds = max(1, int(cfg.get("max_fetch_rounds", 8)))

    interval = _map_candle_interval(timeframe=timeframe, candle_interval_cls=sdk["CandleInterval"])
    minutes = timeframe_minutes(timeframe)
    to_dt = datetime.now(tz=timezone.utc)

    async_client = sdk["AsyncClient"]
    processed = 0
    inserted = 0
    updated = 0
    ignored = 0
    attempted = 0
    with_data = 0

    async with async_client(token) as client:
        for instrument in registry.enabled():
            instrument_id = (instrument.uid or "").strip() or (instrument.figi or "").strip()
            if not instrument_id:
                logger.warning(
                    "History preload skipped instrument=%s: no UID/FIGI",
                    instrument.symbol,
                )
                continue

            attempted += 1
            try:
                rows = await _fetch_max_available_rows(
                    client=client,
                    instrument_id=instrument_id,
                    interval=interval,
                    to_dt=to_dt,
                    minutes=minutes,
                    requested_bars=requested_bars,
                    limit_multiplier=limit_multiplier,
                    lookback_start_multiplier=lookback_start_multiplier,
                    lookback_growth_factor=lookback_growth_factor,
                    max_lookback_multiplier=max_lookback_multiplier,
                    max_fetch_rounds=max_fetch_rounds,
                )
            except Exception as exc:
                logger.warning(
                    "History preload failed instrument=%s error=%s",
                    instrument.symbol,
                    exc,
                )
                continue

            if not rows:
                continue
            if len(rows) < requested_bars:
                logger.warning(
                    "History preload partial instrument=%s bars=%d requested=%d (using max available)",
                    instrument.symbol,
                    len(rows),
                    requested_bars,
                )
            with_data += 1
            for row in rows:
                try:
                    candle = Candle.validated(
                        dt=getattr(row, "time"),
                        open_=_quotation_to_float(getattr(row, "open")),
                        high=_quotation_to_float(getattr(row, "high")),
                        low=_quotation_to_float(getattr(row, "low")),
                        close=_quotation_to_float(getattr(row, "close")),
                        volume=float(getattr(row, "volume", 0.0)),
                        instrument=instrument.symbol,
                        timeframe=timeframe,
                    )
                except (CandleValidationError, TypeError, ValueError) as exc:
                    logger.debug(
                        "History preload invalid candle instrument=%s error=%s",
                        instrument.symbol,
                        exc,
                    )
                    continue

                processed += 1
                state = store.upsert(candle)
                if state == "inserted":
                    inserted += 1
                elif state == "updated":
                    updated += 1
                else:
                    ignored += 1

    return PreloadReport(
        enabled=True,
        requested_bars=requested_bars,
        processed_candles=processed,
        inserted=inserted,
        updated=updated,
        ignored=ignored,
        instruments_attempted=attempted,
        instruments_with_data=with_data,
    )


def estimate_required_bars(*, params: dict[str, Any], timeframe: str) -> int:
    indicator_cfg = params.get("indicator_engine", {})
    if not isinstance(indicator_cfg, dict):
        indicator_cfg = {}
    ema_trend = int(indicator_cfg.get("ema_trend", 200))
    atr_period = int(indicator_cfg.get("atr_period", 14))
    volume_period = int(indicator_cfg.get("volume_period", 20))
    slope_period = int(indicator_cfg.get("slope_period", 5))
    overlap_window = int(indicator_cfg.get("overlap_window", 12))
    swing_window = int(indicator_cfg.get("swing_window", 5))
    crossing_lookback = int(indicator_cfg.get("crossing_lookback", 30))
    indicator_need = max(
        ema_trend + 2,
        atr_period + 2,
        volume_period + 2,
        slope_period + 2,
        overlap_window + 2,
        swing_window + 2,
        crossing_lookback + 2,
        30,
    )

    strat_cfg = params.get("strategy_params", {})
    if not isinstance(strat_cfg, dict):
        strat_cfg = {}

    trend_variants = iter_strategy_param_variants(
        section=strat_cfg,
        strategy_name="trend_pullback_vwap_ema",
    )
    comp_variants = iter_strategy_param_variants(
        section=strat_cfg,
        strategy_name="compression_breakout",
    )
    sweep_variants = iter_strategy_param_variants(
        section=strat_cfg,
        strategy_name="liquidity_sweep_reversal",
    )

    trend_need = max(int(row.get("impulse_bars", 3)) + 2 for row in trend_variants)
    comp_need = max(
        int(row.get("compression_window_bars", 12)) + int(row.get("max_retest_bars", 2)) + 1
        for row in comp_variants
    )
    sweep_need = max(int(row.get("reference_lookback_bars", 20)) + 2 for row in sweep_variants)

    bars = max(indicator_need, trend_need, comp_need, sweep_need)
    bars = max(
        bars,
        _max_mtf_source_bars(variants=trend_variants, timeframe=timeframe),
        _max_mtf_source_bars(variants=comp_variants, timeframe=timeframe),
        _max_mtf_source_bars(variants=sweep_variants, timeframe=timeframe),
    )
    return bars


def _estimate_mtf_source_bars(*, strategy_cfg: dict[str, Any], timeframe: str) -> int:
    if not _as_bool(strategy_cfg.get("use_mtf_filter", False)):
        return 0
    source = _timeframe_minutes_or_none(timeframe)
    if source is None or source <= 0:
        return 0
    slow = int(strategy_cfg.get("mtf_slow_ema", 6))
    slope = int(strategy_cfg.get("mtf_slope_bars", 2))
    required_agg = slow + slope + 1

    trend_tf = str(strategy_cfg.get("trend_timeframe", "1hour"))
    setup_tf = str(strategy_cfg.get("setup_timeframe", "15min"))
    trend_mult = _ratio(source=source, target_tf=trend_tf)
    setup_mult = _ratio(source=source, target_tf=setup_tf)
    max_mult = max(trend_mult, setup_mult, 1)
    return required_agg * max_mult


def _max_mtf_source_bars(*, variants: tuple[dict[str, Any], ...], timeframe: str) -> int:
    return max(
        (_estimate_mtf_source_bars(strategy_cfg=row, timeframe=timeframe) for row in variants),
        default=0,
    )


def _ratio(*, source: int, target_tf: str) -> int:
    target = _timeframe_minutes_or_none(target_tf)
    if target is None or target < source or target % source != 0:
        return 0
    return target // source


def _map_candle_interval(*, timeframe: str, candle_interval_cls: Any) -> Any:
    return map_tinvest_history_interval(
        timeframe=timeframe,
        candle_interval_cls=candle_interval_cls,
    )


async def _fetch_max_available_rows(
    *,
    client: Any,
    instrument_id: str,
    interval: Any,
    to_dt: datetime,
    minutes: int,
    requested_bars: int,
    limit_multiplier: int,
    lookback_start_multiplier: float,
    lookback_growth_factor: float,
    max_lookback_multiplier: float,
    max_fetch_rounds: int,
) -> list[Any]:
    unique_by_ts: dict[str, Any] = {}
    lookback_multiplier = lookback_start_multiplier
    max_limit = max(50, int(requested_bars * limit_multiplier))

    for _ in range(max_fetch_rounds):
        from_dt = to_dt - timedelta(minutes=int(requested_bars * minutes * lookback_multiplier))
        response = await client.market_data.get_candles(
            instrument_id=instrument_id,
            interval=interval,
            from_=from_dt,
            to=to_dt,
            limit=max_limit,
        )
        rows = list(getattr(response, "candles", []) or [])
        before = len(unique_by_ts)
        for row in rows:
            dt = getattr(row, "time", None)
            if dt is None:
                continue
            key = _dt_key(dt)
            unique_by_ts[key] = row

        if len(unique_by_ts) >= requested_bars:
            break
        if lookback_multiplier >= max_lookback_multiplier and len(unique_by_ts) == before:
            break
        if lookback_multiplier >= max_lookback_multiplier:
            break
        lookback_multiplier = min(max_lookback_multiplier, lookback_multiplier * lookback_growth_factor)

    rows_out = sorted(
        unique_by_ts.values(),
        key=lambda item: _dt_key(getattr(item, "time", None)),
    )
    if len(rows_out) > requested_bars:
        return rows_out[-requested_bars:]
    return rows_out


def _dt_key(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _timeframe_minutes_or_none(timeframe: str) -> int | None:
    try:
        return timeframe_minutes(timeframe)
    except ValueError:
        return None
