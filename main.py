import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from t_tech.invest import AsyncClient, CandleInterval, InstrumentIdType

from stats import StatsTracker

# pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# ==============================
# CONFIG
# ==============================
# Environment variables required:
#   INVEST_TOKEN=...
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=...
#
# Optional:
#   DB_PATH=signals.db
#   CHECK_EVERY_SECONDS=60
#
# Notes:
# 1) Strategy is trend-first:
#    - H1 defines market bias
#    - M15 builds setup
#    - M5 confirms entry on candle close
# 2) Countertrend logic is disabled by default and separated with a flag.
# 3) Cocoa is intentionally excluded from the default basket because its intraday behavior
#    is materially less uniform than Brent / Silver / S&P 500 futures.
#
# Instrument IDs:
# T-Bank requires either instrument UID / FIGI, or ticker + class_code.
# The code below resolves instruments by ticker + class_code via GetInstrumentBy.
# Update TICKER and CLASS_CODE with the exact symbols from your broker terminal / API.
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.environ["INVEST_TOKEN"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
DB_PATH = os.getenv("DB_PATH", "signals.db")
CHECK_EVERY_SECONDS = int(os.getenv("CHECK_EVERY_SECONDS", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("futures-signal-bot")


@dataclass
class InstrumentConfig:
    alias: str
    ticker: str
    class_code: str
    allow_countertrend: bool = False
    atr_stop_mult: float = 1.2
    adx_threshold_h1: float = 18.0
    adx_threshold_m15: float = 16.0
    min_score_trend: int = 3  # MODE: для более строго алгоса = 4
    min_score_setup: int = 3  # MODE: для более строго алгоса = 4
    m15_setup_expiry_bars: int = 4
    rr_tp1: float = 1.0
    rr_tp2: float = 2.0


DEFAULT_INSTRUMENTS: List[InstrumentConfig] = [
    # InstrumentConfig(alias="BR", ticker="BMK6", class_code="SPBFUT", allow_countertrend=False),
    InstrumentConfig(alias="SI", ticker="S1M6", class_code="SPBFUT", allow_countertrend=True),
    InstrumentConfig(alias="SPX", ticker="SFM6", class_code="SPBFUT", allow_countertrend=True),
    InstrumentConfig(alias="CC", ticker="CCJ6", class_code="SPBFUT", allow_countertrend=True),
    # InstrumentConfig(alias="NIKK", ticker="N2M6", class_code="SPBFUT", allow_countertrend=False),
]


# ==============================
# DB / STATE
# ==============================
class StateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    instrument_alias TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load(self, instrument_alias: str) -> Dict[str, Any]:
        import json

        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM state WHERE instrument_alias = ?",
                (instrument_alias,),
            ).fetchone()
        if not row:
            return {}
        return json.loads(row[0])

    def save(self, instrument_alias: str, payload: Dict[str, Any]) -> None:
        import json

        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO state (instrument_alias, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(instrument_alias) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (instrument_alias, json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()


# ==============================
# TELEGRAM
# ==============================
async def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=20) as resp:
            body = await resp.text()
            if resp.status >= 300:
                raise RuntimeError(f"Telegram error {resp.status}: {body}")


# ==============================
# T-BANK HELPERS
# ==============================
def quotation_to_float(q: Any) -> float:
    return q.units + q.nano / 1_000_000_000


async def resolve_instrument(client: AsyncClient, cfg: InstrumentConfig) -> Tuple[str, str]:
    response = await client.instruments.get_instrument_by(
        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
        class_code=cfg.class_code,
        id=cfg.ticker,
    )
    instrument = response.instrument
    instrument_id = instrument.uid or instrument.figi
    if not instrument_id:
        raise RuntimeError(f"Could not resolve instrument id for {cfg.alias}")
    return instrument_id, instrument.name


INTERVAL_TO_DAYS: Dict[CandleInterval, int] = {
    CandleInterval.CANDLE_INTERVAL_HOUR: 35,
    CandleInterval.CANDLE_INTERVAL_15_MIN: 12,
    CandleInterval.CANDLE_INTERVAL_5_MIN: 5,
}


async def fetch_candles_df(
    client: AsyncClient,
    instrument_id: str,
    interval: CandleInterval,
) -> pd.DataFrame:
    from t_tech.invest.utils import now

    rows: List[Dict[str, Any]] = []
    async for candle in client.get_all_candles(
        instrument_id=instrument_id,
        from_=now() - timedelta(days=INTERVAL_TO_DAYS[interval]),
        interval=interval,
    ):
        rows.append(
            {
                "time": candle.time,
                "open": quotation_to_float(candle.open),
                "high": quotation_to_float(candle.high),
                "low": quotation_to_float(candle.low),
                "close": quotation_to_float(candle.close),
                "volume": float(candle.volume),
                "is_complete": bool(getattr(candle, "is_complete", True)),
            }
        )

    if not rows:
        raise RuntimeError(f"No candles returned for {instrument_id}, interval={interval}")

    df = pd.DataFrame(rows).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ==============================
# INDICATORS
# ==============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()



def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)



def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist



def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()



def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_smoothed = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smoothed.replace(0, pd.NA)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smoothed.replace(0, pd.NA)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    return dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean().fillna(0.0)



def session_vwap(df: pd.DataFrame) -> pd.Series:
    # Approximates intraday VWAP by UTC calendar day.
    # If you later need strict exchange-session alignment, group by exchange local session date.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    session_key = df["time"].dt.strftime("%Y-%m-%d")
    pv = typical_price * df["volume"]
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum().replace(0, pd.NA)
    return (cum_pv / cum_vol).fillna(df["close"])



def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)
    macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["atr14"] = atr(df, 14)
    df["adx14"] = adx(df, 14)
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vwap"] = session_vwap(df)
    return df



def align_lower_to_higher(lower_df: pd.DataFrame, higher_df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in higher_df.columns]
    if missing:
        raise KeyError(f"Missing columns in higher_df: {missing}. Available: {list(higher_df.columns)}")

    left = lower_df.sort_values("time").copy()
    rename_map = {c: f"{prefix}_{c}" for c in cols}
    right = (
        higher_df[["time", *cols]]
        .copy()
        .rename(columns=rename_map)
        .sort_values("time")
    )

    merged = pd.merge_asof(left, right, on="time", direction="backward")
    return merged

# ==============================
# STRATEGY
# ==============================
def score_trend_long(row: pd.Series, adx_threshold: float) -> int:
    return sum(
        [
            row["close"] > row["ema9"],
            row["ema9"] > row["ema21"],
            row["close"] > row["vwap"],
            row["rsi14"] > 52,
            row["macd"] > row["macd_signal"],
            row["adx14"] > adx_threshold,
        ]
    )



def score_trend_short(row: pd.Series, adx_threshold: float) -> int:
    return sum(
        [
            row["close"] < row["ema9"],
            row["ema9"] < row["ema21"],
            row["close"] < row["vwap"],
            row["rsi14"] < 48,
            row["macd"] < row["macd_signal"],
            row["adx14"] > adx_threshold,
        ]
    )



def swing_stop_long(df: pd.DataFrame, lookback: int = 5) -> float:
    return float(df["low"].tail(lookback).min())



def swing_stop_short(df: pd.DataFrame, lookback: int = 5) -> float:
    return float(df["high"].tail(lookback).max())



def determine_h1_bias(h1: pd.DataFrame, cfg: InstrumentConfig) -> Dict[str, Any]:
    row = h1.iloc[-1]
    long_score = score_trend_long(row, cfg.adx_threshold_h1)
    short_score = score_trend_short(row, cfg.adx_threshold_h1)

    if long_score >= cfg.min_score_trend and long_score > short_score:
        side = "long"
    elif short_score >= cfg.min_score_trend and short_score > long_score:
        side = "short"
    else:
        side = "neutral"

    return {
        "side": side,
        "long_score": int(long_score),
        "short_score": int(short_score),
        "adx": round(float(row["adx14"]), 2),
        "rsi": round(float(row["rsi14"]), 2),
        "close": round(float(row["close"]), 4),
        "ema9": round(float(row["ema9"]), 4),
        "ema21": round(float(row["ema21"]), 4),
        "vwap": round(float(row["vwap"]), 4),
        "time": row["time"].isoformat(),
    }



def determine_m15_setup(m15: pd.DataFrame, h1_bias: Dict[str, Any], cfg: InstrumentConfig) -> Dict[str, Any]:
    row = m15.iloc[-1]
    prev = m15.iloc[-2]
    side = h1_bias["side"]

    if side == "long":
        setup_score = sum(
            [
                row["close"] > row["ema21"],
                row["ema9"] > row["ema21"],
                row["rsi14"] > 48,
                row["macd"] > row["macd_signal"],
                row["adx14"] > cfg.adx_threshold_m15,
                row["low"] <= row["ema9"] * 1.001 or row["low"] <= row["vwap"] * 1.001,
            ]
        )
        valid = setup_score >= cfg.min_score_setup and row["close"] >= prev["close"]
    elif side == "short":
        setup_score = sum(
            [
                row["close"] < row["ema21"],
                row["ema9"] < row["ema21"],
                row["rsi14"] < 52,
                row["macd"] < row["macd_signal"],
                row["adx14"] > cfg.adx_threshold_m15,
                row["high"] >= row["ema9"] * 0.999 or row["high"] >= row["vwap"] * 0.999,
            ]
        )
        valid = setup_score >= cfg.min_score_setup and row["close"] <= prev["close"]
    else:
        setup_score = 0
        valid = False

    return {
        "active": bool(valid),
        "side": side,
        "score": int(setup_score),
        "time": row["time"].isoformat(),
        "close": round(float(row["close"]), 4),
        "ema9": round(float(row["ema9"]), 4),
        "ema21": round(float(row["ema21"]), 4),
        "vwap": round(float(row["vwap"]), 4),
    }



def determine_m5_entry(m5: pd.DataFrame, h1_bias: Dict[str, Any], m15_setup: Dict[str, Any], cfg: InstrumentConfig) -> Optional[Dict[str, Any]]:
    if not m15_setup["active"]:
        return None

    row = m5.iloc[-1]
    prev = m5.iloc[-2]

    # Softer entry logic for learning / lower-volatility sessions.
    # We keep candle-close confirmation, but reduce the number of mandatory filters.
    if h1_bias["side"] == "long":
        support_touch = (
            row["low"] <= row["ema9"] * 1.002
            or row["low"] <= row["ema21"] * 1.001
            or row["low"] <= row["vwap"] * 1.001
        )
        bullish_reclaim = (
            row["close"] > row["ema9"]
            or (prev["close"] <= prev["ema9"] and row["close"] > prev["high"])
        )
        momentum_ok = sum([
            row["rsi14"] > 48,
            row["macd"] >= row["macd_signal"],
            row["close"] >= row["open"],
            row["volume"] >= row["vol_sma20"] * 0.8 if pd.notna(row["vol_sma20"]) else True,
        ]) >= 2

        if not (support_touch and bullish_reclaim and momentum_ok):
            return None

        entry = float(row["close"])
        atr_stop = entry - float(row["atr14"]) * cfg.atr_stop_mult
        struct_stop = swing_stop_long(m5, lookback=5)
        stop = min(atr_stop, struct_stop)
        risk = entry - stop
        if risk <= 0:
            return None

        score_for = int(score_trend_long(row, cfg.adx_threshold_m15))
        score_against = int(score_trend_short(row, cfg.adx_threshold_m15))
        return {
            "side": "long",
            "countertrend": False,
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "tp1": round(entry + risk * cfg.rr_tp1, 4),
            "tp2": round(entry + risk * cfg.rr_tp2, 4),
            "risk_r": round(risk, 4),
            "entry_time": row["time"].isoformat(),
            "reason": "M5 подтвердил long после отката к EMA/VWAP в сторону тренда H1 и setup M15",
            "score_for": score_for,
            "score_against": score_against,
            "confidence": confidence_from_scores(score_for, score_against),
        }

    if h1_bias["side"] == "short":
        resistance_touch = (
            row["high"] >= row["ema9"] * 0.998
            or row["high"] >= row["ema21"] * 0.999
            or row["high"] >= row["vwap"] * 0.999
        )
        bearish_reclaim = (
            row["close"] < row["ema9"]
            or (prev["close"] >= prev["ema9"] and row["close"] < prev["low"])
        )
        momentum_ok = sum([
            row["rsi14"] < 52,
            row["macd"] <= row["macd_signal"],
            row["close"] <= row["open"],
            row["volume"] >= row["vol_sma20"] * 0.8 if pd.notna(row["vol_sma20"]) else True,
        ]) >= 2

        if not (resistance_touch and bearish_reclaim and momentum_ok):
            return None

        entry = float(row["close"])
        atr_stop = entry + float(row["atr14"]) * cfg.atr_stop_mult
        struct_stop = swing_stop_short(m5, lookback=5)
        stop = max(atr_stop, struct_stop)
        risk = stop - entry
        if risk <= 0:
            return None

        score_for = int(score_trend_short(row, cfg.adx_threshold_m15))
        score_against = int(score_trend_long(row, cfg.adx_threshold_m15))
        return {
            "side": "short",
            "countertrend": False,
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "tp1": round(entry - risk * cfg.rr_tp1, 4),
            "tp2": round(entry - risk * cfg.rr_tp2, 4),
            "risk_r": round(risk, 4),
            "entry_time": row["time"].isoformat(),
            "reason": "M5 подтвердил short после отката к EMA/VWAP в сторону тренда H1 и setup M15",
            "score_for": score_for,
            "score_against": score_against,
            "confidence": confidence_from_scores(score_for, score_against),
        }

    return None


def maybe_countertrend_entry(m5: pd.DataFrame, h1_bias: Dict[str, Any], cfg: InstrumentConfig) -> Optional[Dict[str, Any]]:
    if not cfg.allow_countertrend:
        return None

    row = m5.iloc[-1]
    prev = m5.iloc[-2]

    # Softer countertrend logic, but still stricter than trend entries.
    if h1_bias["side"] == "long":
        stretched_down = row["close"] < row["ema21"] and row["rsi14"] < 45
        reversal_hint = (
            row["close"] < row["ema9"]
            and (row["close"] < row["open"] or row["close"] < prev["low"])
            and row["macd"] <= row["macd_signal"]
        )
        if stretched_down and reversal_hint:
            entry = float(row["close"])
            atr_stop = entry + float(row["atr14"]) * 1.0
            struct_stop = swing_stop_short(m5, lookback=4)
            stop = max(atr_stop, struct_stop)
            risk = stop - entry
            if risk > 0:
                score_for = int(score_trend_short(row, cfg.adx_threshold_m15))
                score_against = int(score_trend_long(row, cfg.adx_threshold_m15))
                return {
                    "side": "short",
                    "countertrend": True,
                    "entry": round(entry, 4),
                    "stop": round(stop, 4),
                    "tp1": round(entry - risk, 4),
                    "tp2": round(entry - 1.5 * risk, 4),
                    "risk_r": round(risk, 4),
                    "entry_time": row["time"].isoformat(),
                    "reason": "Контртрендовый short: M5 показал разворот вниз после локального растяжения против H1 long",
                    "score_for": score_for,
                    "score_against": score_against,
                    "confidence": confidence_from_scores(score_for, score_against),
                }

    if h1_bias["side"] == "short":
        stretched_up = row["close"] > row["ema21"] and row["rsi14"] > 55
        reversal_hint = (
            row["close"] > row["ema9"]
            and (row["close"] > row["open"] or row["close"] > prev["high"])
            and row["macd"] >= row["macd_signal"]
        )
        if stretched_up and reversal_hint:
            entry = float(row["close"])
            atr_stop = entry - float(row["atr14"]) * 1.0
            struct_stop = swing_stop_long(m5, lookback=4)
            stop = min(atr_stop, struct_stop)
            risk = entry - stop
            if risk > 0:
                score_for = int(score_trend_long(row, cfg.adx_threshold_m15))
                score_against = int(score_trend_short(row, cfg.adx_threshold_m15))
                return {
                    "side": "long",
                    "countertrend": True,
                    "entry": round(entry, 4),
                    "stop": round(stop, 4),
                    "tp1": round(entry + risk, 4),
                    "tp2": round(entry + 1.5 * risk, 4),
                    "risk_r": round(risk, 4),
                    "entry_time": row["time"].isoformat(),
                    "reason": "Контртрендовый long: M5 показал разворот вверх после локального растяжения против H1 short",
                    "score_for": score_for,
                    "score_against": score_against,
                    "confidence": confidence_from_scores(score_for, score_against),
                }

    return None



def trade_update(trade: Dict[str, Any], latest_price_row: pd.Series) -> Dict[str, Any]:
    updated = dict(trade)
    side = trade["side"]
    high = float(latest_price_row["high"])
    low = float(latest_price_row["low"])
    stop = float(trade["stop"])
    entry = float(trade["entry"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])

    updated.setdefault("status", "active")
    updated.setdefault("tp1_hit", False)
    updated.setdefault("tp2_hit", False)
    updated.setdefault("breakeven_armed", False)

    if side == "long":
        if not updated["tp1_hit"] and high >= tp1:
            updated["tp1_hit"] = True
            updated["breakeven_armed"] = True
            updated["stop"] = round(entry, 4)
        if high >= tp2:
            updated["tp2_hit"] = True
            updated["status"] = "closed_tp2"
        if low <= float(updated["stop"]):
            updated["status"] = "closed_breakeven" if updated["tp1_hit"] else "closed_stop"
    else:
        if not updated["tp1_hit"] and low <= tp1:
            updated["tp1_hit"] = True
            updated["breakeven_armed"] = True
            updated["stop"] = round(entry, 4)
        if low <= tp2:
            updated["tp2_hit"] = True
            updated["status"] = "closed_tp2"
        if high >= float(updated["stop"]):
            updated["status"] = "closed_breakeven" if updated["tp1_hit"] else "closed_stop"

    return updated



def setup_invalidated(m15: pd.DataFrame, setup: Dict[str, Any], h1_bias: Dict[str, Any], cfg: InstrumentConfig) -> bool:
    if not setup.get("active"):
        return False

    setup_time = pd.Timestamp(setup["time"])
    elapsed_bars = int((m15["time"] > setup_time).sum())
    if elapsed_bars > cfg.m15_setup_expiry_bars:
        return True

    row = m15.iloc[-1]
    side = setup["side"]
    if h1_bias["side"] != side:
        return True

    if side == "long":
        return bool(row["close"] < row["ema21"] and row["close"] < row["vwap"])
    if side == "short":
        return bool(row["close"] > row["ema21"] and row["close"] > row["vwap"])
    return True


# ==============================
# MESSAGE BUILDERS
# ==============================
def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)



def direction_ru(side: str) -> str:
    return {"long": "LONG", "short": "SHORT", "neutral": "НЕЙТРАЛЬНО"}.get(side, side.upper())


def confidence_from_scores(long_score: int, short_score: int, max_score: int = 6) -> int:
    lead = max(long_score, short_score)
    edge = abs(long_score - short_score)
    raw = (lead / max_score) * 70 + (edge / max_score) * 30
    return int(round(min(100, max(0, raw))))


def bias_message(alias: str, instrument_name: str, bias: Dict[str, Any]) -> str:
    confidence = confidence_from_scores(bias["long_score"], bias["short_score"])
    return (
        f'''<b>{alias}</b> | {instrument_name}\n\n'''
        f'''<b>ОБЩИЙ ТРЕНД</b>\n'''
        f'''<b>Тренд H1:</b> {direction_ru(bias['side'])}\n'''
        f'''Счет: \n  - за LONG: {bias['long_score']}\n  - за SHORT: {bias['short_score']}\n'''
        f'''\nОценка силы сигнала: {confidence}%\n'''
    )



def setup_message(alias: str, instrument_name: str, setup: Dict[str, Any]) -> str:
    setup_confidence = int(round(min(100, max(0, (setup['score'] / 6) * 100))))
    return (
        f'''<b>{alias}</b> | {instrument_name}\n\n'''
        f'''<b>Уведомление - M15 активен:</b> {direction_ru(setup['side'])}\n'''
        f'''Счет подтверждений: {setup['score']} из 6\n'''
        f'''Вероятность / сила setup: {setup_confidence}%\n'''
        f'''Цена: {fmt(setup['close'])} | EMA9: {fmt(setup['ema9'])} | EMA21: {fmt(setup['ema21'])}\n'''
        f'''VWAP: {fmt(setup['vwap'])}\n'''
    )



def invalidation_message(alias: str, instrument_name: str, setup: Dict[str, Any]) -> str:
    return (
        f'''<b>{alias}</b> | {instrument_name}\n\n'''
        f'''<b>Уведомление - отмена</b>\n'''
        f'''Направление: {direction_ru(setup.get('side', 'n/a'))}\n'''
        f'''Время предыдущего setup: {setup.get('time', 'n/a')}\n'''
    )



def entry_message(alias: str, instrument_name: str, entry: Dict[str, Any]) -> str:
    is_ct = entry.get("countertrend")
    label = "🔁 Контртренд" if is_ct else "➡️ По тренду"
    confidence = entry.get("confidence")
    score_for = entry.get("score_for")
    score_against = entry.get("score_against")

    extra = ""
    if score_for is not None and score_against is not None:
        extra += f"Счет за вход: {score_for} | против: {score_against}\n"
    if confidence is not None:
        extra += f"Оценка вероятности: {confidence}%\n"

    return (
        f'''<b>{alias}</b> | {instrument_name}\n\n'''
        f'''{label}\n'''
        f'''Тикер: {alias}\n'''
        f'''Направление: {'🟢 LONG' if entry['side']=='long' else '🔴 SHORT'}\n'''
        f'''Причина: {entry['reason']}\n'''
        f'''Вход: {fmt(entry['entry'])} | SL: {fmt(entry['stop'])}\n'''
        f'''TP1: {fmt(entry['tp1'])} | TP2: {fmt(entry['tp2'])}\n'''
        f'''{extra}\n'''
        f'''Риск: {fmt(entry['risk_r'])}\n'''
    )



def trade_event_message(alias: str, instrument_name: str, event: str, trade: Dict[str, Any]) -> str:
    title_map = {
        "tp1": "TP1 достигнут, стоп перенесен в безубыток",
        "tp2": "TP2 достигнут, сделка закрыта",
        "closed_stop": "Сработал стоп-лосс",
        "closed_breakeven": "Сделка закрыта по безубытку",
    }
    title = title_map.get(event, event)
    return (
        f'''<b>{alias}</b> | {instrument_name}\n\n'''
        f'''<b>{title}</b>\n'''
        f'''Тикер: {alias}\n'''
        f'''Направление: {direction_ru(trade['side'])}\n'''
        f'''Вход: {fmt(trade['entry'])}\n'''
        f'''SL: {fmt(trade['stop'])}\n'''
        f'''TP1: {fmt(trade['tp1'])}\n'''
        f'''TP2: {fmt(trade['tp2'])}\n'''
    )


# ==============================
# ENGINE
# ==============================
async def process_instrument(
    client: AsyncClient,
    store: StateStore,
    stats: StatsTracker,
    cfg: InstrumentConfig,
) -> None:
    instrument_id, instrument_name = await resolve_instrument(client, cfg)

    h1 = enrich(await fetch_candles_df(client, instrument_id, CandleInterval.CANDLE_INTERVAL_HOUR))
    m15 = enrich(await fetch_candles_df(client, instrument_id, CandleInterval.CANDLE_INTERVAL_15_MIN))
    m5 = enrich(await fetch_candles_df(client, instrument_id, CandleInterval.CANDLE_INTERVAL_5_MIN))

    # Enrich lower TFs with H1 context when needed later.
    m15 = align_lower_to_higher(m15, h1, ["ema9", "ema21", "vwap", "adx14", "rsi14"], "h1")
    m5 = align_lower_to_higher(m5, m15, ["ema9", "ema21", "vwap", "adx14", "rsi14"], "m15")

    current_bias = determine_h1_bias(h1, cfg)
    current_setup = determine_m15_setup(m15, current_bias, cfg)
    new_entry = determine_m5_entry(m5, current_bias, current_setup, cfg)
    countertrend_entry = maybe_countertrend_entry(m5, current_bias, cfg)

    state = store.load(cfg.alias)
    previous_bias = state.get("bias", {})
    previous_setup = state.get("setup", {})
    active_trade = state.get("trade")

    # 1) H1 bias change
    # if previous_bias.get("side") != current_bias["side"]:
    #     await send_telegram_message(bias_message(cfg.alias, instrument_name, current_bias))

    # 2) M15 setup activation
    # if current_setup["active"] and not previous_setup.get("active", False):
    #     await send_telegram_message(setup_message(cfg.alias, instrument_name, current_setup))

    # 3) M15 invalidation
    if previous_setup.get("active", False) and setup_invalidated(m15, previous_setup, current_bias, cfg):
        # await send_telegram_message(invalidation_message(cfg.alias, instrument_name, previous_setup))
        current_setup = {"active": False, "side": current_bias["side"]}

    # 4) Entries
    chosen_entry = None
    if not active_trade and new_entry:
        chosen_entry = new_entry
    elif not active_trade and countertrend_entry:
        chosen_entry = countertrend_entry

    if chosen_entry:
        active_trade = {
            **chosen_entry,
            "status": "active",
            "tp1_hit": False,
            "tp2_hit": False,
            "breakeven_armed": False,
        }
        stats.record_entry(cfg.alias)
        await send_telegram_message(entry_message(cfg.alias, instrument_name, active_trade))

    # 5) Existing trade lifecycle
    if active_trade:
        prev_trade = dict(active_trade)
        updated_trade = trade_update(active_trade, m5.iloc[-1])

        if not prev_trade.get("tp1_hit") and updated_trade.get("tp1_hit"):
            stats.record_tp1(cfg.alias)
            await send_telegram_message(trade_event_message(cfg.alias, instrument_name, "tp1", updated_trade))

        if not prev_trade.get("tp2_hit") and updated_trade.get("tp2_hit"):
            stats.record_tp2(cfg.alias)
            await send_telegram_message(trade_event_message(cfg.alias, instrument_name, "tp2", updated_trade))

        if prev_trade.get("status") == "active" and updated_trade.get("status") == "closed_stop":
            stats.record_sl(cfg.alias)
            await send_telegram_message(trade_event_message(cfg.alias, instrument_name, "closed_stop", updated_trade))

        if prev_trade.get("status") == "active" and updated_trade.get("status") == "closed_breakeven":
            stats.record_breakeven(cfg.alias)
            await send_telegram_message(
                trade_event_message(cfg.alias, instrument_name, "closed_breakeven", updated_trade))

        if updated_trade.get("status", "active").startswith("closed"):
            active_trade = None
        else:
            active_trade = updated_trade

    new_state = {
        "instrument_id": instrument_id,
        "instrument_name": instrument_name,
        "bias": current_bias,
        "setup": current_setup,
        "trade": active_trade,
        "last_processed_at": datetime.now(timezone.utc).isoformat(),
    }
    store.save(cfg.alias, new_state)


async def run_once(instruments: List[InstrumentConfig], stats: StatsTracker) -> None:
    store = StateStore(DB_PATH)
    async with AsyncClient(TOKEN) as client:
        for cfg in instruments:
            try:
                await process_instrument(client, store, stats, cfg)
            except Exception as exc:
                logger.exception("Error processing %s: %s", cfg.alias, exc)
                try:
                    await send_telegram_message(
                        f"<b>{cfg.alias}</b>\n<b>Error:</b> {type(exc).__name__}: {exc}"
                    )
                except Exception:
                    logger.exception("Failed to send Telegram error message for %s", cfg.alias)


async def main_loop() -> None:
    logger.info("Starting futures signal bot")
    logger.info("Scan interval: %s sec", CHECK_EVERY_SECONDS)

    stats = StatsTracker()

    try:
        while True:
            started = datetime.now(timezone.utc)
            await run_once(DEFAULT_INSTRUMENTS, stats)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            sleep_for = max(1, CHECK_EVERY_SECONDS - int(elapsed))
            await asyncio.sleep(sleep_for)
    finally:
        print(stats.build_report())


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем.")
