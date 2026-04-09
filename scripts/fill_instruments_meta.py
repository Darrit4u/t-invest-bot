"""Rewrite figi/uid/tick_size/tick_value in config/instruments.yaml via t_tech SDK."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill figi/uid/tick_size/tick_value for futures in instruments.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "instruments.yaml",
        help="Path to instruments.yaml",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
        help="Path to .env file with INVEST_TOKEN",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Explicit INVEST_TOKEN value (overrides .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write file, only print planned changes",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE variables from .env if present."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and not value.startswith(("'", '"')) and "#" in value:
            value = value.split("#", 1)[0].strip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_token(args: argparse.Namespace) -> str:
    load_env_file(args.env_file)
    if args.token.strip():
        return args.token.strip()
    return os.getenv("INVEST_TOKEN", "").strip()


def normalize_text(value: Any) -> str:
    return str(value or "").strip().upper()


def price_to_float(value: Any) -> float | None:
    """Convert Quotation/MoneyValue-like object to float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    units = getattr(value, "units", None)
    nano = getattr(value, "nano", None)
    if units is not None or nano is not None:
        return float(units or 0) + float(nano or 0) / 1_000_000_000

    if isinstance(value, dict):
        units = value.get("units")
        nano = value.get("nano")
        if units is not None or nano is not None:
            return float(units or 0) + float(nano or 0) / 1_000_000_000

    return None


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from t_tech.invest import (  # type: ignore
            Client,
            InstrumentIdType,
            InstrumentStatus,
            InstrumentType,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Package t-tech-investments is not installed. "
            "Install it with:\n"
            "pip install t-tech-investments --index-url "
            "https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
        ) from exc
    return Client, InstrumentStatus, InstrumentType, InstrumentIdType


def choose_future(
    *,
    ticker: str,
    class_code: str,
    by_key: dict[tuple[str, str], Any],
    by_ticker: dict[str, list[Any]],
) -> tuple[Any | None, str | None]:
    key = (ticker, class_code)
    if class_code and key in by_key:
        return by_key[key], None

    candidates = by_ticker.get(ticker, [])
    if not candidates:
        return None, f"ticker={ticker} not found in futures list"

    if class_code:
        return None, f"ticker={ticker} found, but class_code={class_code} not matched"

    if len(candidates) == 1:
        return candidates[0], None

    return None, f"ticker={ticker} has {len(candidates)} matches, class_code is required"


def choose_candidate_from_find(*, candidates: list[Any], ticker: str, class_code: str) -> Any | None:
    ticker_u = normalize_text(ticker)
    class_code_u = normalize_text(class_code)

    def matches(item: Any) -> bool:
        if class_code_u and normalize_text(getattr(item, "class_code", "")) != class_code_u:
            return False
        return True

    filtered = [item for item in candidates if matches(item)]
    if not filtered:
        return None

    for item in filtered:
        if normalize_text(getattr(item, "ticker", "")) == ticker_u:
            return item

    if len(filtered) == 1:
        return filtered[0]
    return None


ALIAS_HINTS: dict[str, tuple[str, ...]] = {
    "ES": ("SPYF", "S&P 500", "SPDR S&P 500"),
    "SILVER": ("SILV", "СЕРЕБРО", "SILVER"),
    "BRENT": ("BRENT", "BR-", "НЕФТЬ BRENT"),
    "NG": ("NG-", "ПРИРОДНЫЙ ГАЗ", "NATURAL GAS"),
}


def _future_expiration_sort_key(future: Any) -> tuple[int, datetime]:
    now = datetime.now(tz=timezone.utc)
    expiration = getattr(future, "expiration_date", None) or getattr(future, "last_trade_date", None)
    if not isinstance(expiration, datetime):
        return (2, datetime.max.replace(tzinfo=timezone.utc))

    exp_utc = expiration.astimezone(timezone.utc)
    if exp_utc >= now:
        return (0, exp_utc)
    return (1, exp_utc)


def choose_future_by_hints(
    *,
    symbol: str,
    ticker_raw: str,
    class_code_raw: str,
    futures: list[Any],
) -> Any | None:
    class_code = normalize_text(class_code_raw)
    symbol_u = normalize_text(symbol)
    hints = [normalize_text(ticker_raw), symbol_u]
    hints.extend(normalize_text(item) for item in ALIAS_HINTS.get(symbol_u, ()))

    scored: list[tuple[int, Any]] = []
    for future in futures:
        future_class_code = normalize_text(getattr(future, "class_code", ""))
        if class_code and future_class_code != class_code:
            continue

        ticker = normalize_text(getattr(future, "ticker", ""))
        name = normalize_text(getattr(future, "name", ""))
        basic_asset = normalize_text(getattr(future, "basic_asset", ""))
        text = f"{ticker} {name} {basic_asset}"

        score = 0
        for hint in hints:
            if not hint:
                continue
            if ticker == hint:
                score += 300
            if ticker.startswith(hint):
                score += 100
            if hint in text:
                score += 40

        if score > 0:
            scored.append((score, future))

    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            -item[0],
            _future_expiration_sort_key(item[1]),
        )
    )
    return scored[0][1]


def resolve_future_with_fallback(
    *,
    client: Any,
    symbol: str,
    ticker_raw: str,
    class_code_raw: str,
    futures: list[Any],
    by_key: dict[tuple[str, str], Any],
    by_ticker: dict[str, list[Any]],
    instrument_type_futures: Any,
    instrument_id_type_uid: Any,
) -> tuple[Any | None, str | None]:
    ticker = normalize_text(ticker_raw)
    class_code = normalize_text(class_code_raw)

    future, reason = choose_future(
        ticker=ticker,
        class_code=class_code,
        by_key=by_key,
        by_ticker=by_ticker,
    )
    if future is not None:
        return future, None

    hinted = choose_future_by_hints(
        symbol=symbol,
        ticker_raw=ticker_raw,
        class_code_raw=class_code_raw,
        futures=futures,
    )
    if hinted is not None:
        return hinted, None

    try:
        found = client.instruments.find_instrument(
            query=ticker_raw,
            instrument_kind=instrument_type_futures,
        )
    except Exception as exc:
        return None, f"{reason}; find_instrument failed: {exc}"

    items = list(getattr(found, "instruments", []))
    candidate = choose_candidate_from_find(
        candidates=items,
        ticker=ticker_raw,
        class_code=class_code_raw,
    )
    if candidate is None:
        return None, reason

    uid = str(getattr(candidate, "uid", "") or "").strip()
    if not uid:
        return None, f"{reason}; candidate uid is empty"

    try:
        response = client.instruments.future_by(
            id_type=instrument_id_type_uid,
            id=uid,
        )
    except Exception as exc:
        return None, f"{reason}; future_by(uid) failed: {exc}"

    resolved = getattr(response, "instrument", None)
    if resolved is None:
        return None, f"{reason}; future_by(uid) returned empty instrument"
    return resolved, None


def main() -> int:
    args = parse_args()
    token = normalize_token(args)

    if not token:
        print("INVEST_TOKEN is empty. Set it in .env or pass --token.", file=sys.stderr)
        return 2

    if not args.config.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 2

    config_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        print("Invalid instruments.yaml root: expected mapping", file=sys.stderr)
        return 2

    instruments = config_raw.get("instruments")
    if not isinstance(instruments, dict):
        print("Invalid instruments.yaml: expected 'instruments' mapping", file=sys.stderr)
        return 2

    try:
        Client, InstrumentStatus, InstrumentType, InstrumentIdType = load_sdk()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    with Client(token) as client:
        try:
            response = client.instruments.futures(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            )
        except Exception as exc:
            print(f"Failed to load futures via SDK: {exc}", file=sys.stderr)
            return 1

        futures = list(getattr(response, "instruments", []))

        by_key: dict[tuple[str, str], Any] = {}
        by_ticker: dict[str, list[Any]] = defaultdict(list)
        for future in futures:
            ticker = normalize_text(getattr(future, "ticker", ""))
            class_code = normalize_text(getattr(future, "class_code", ""))
            if not ticker:
                continue
            by_ticker[ticker].append(future)
            if class_code:
                by_key[(ticker, class_code)] = future

        updated_count = 0
        skipped_count = 0
        failed_count = 0

        for symbol, row in instruments.items():
            if not isinstance(row, dict):
                print(f"[SKIP] {symbol}: row is not an object")
                skipped_count += 1
                continue

            ticker_raw = str(row.get("ticker", "")).strip()
            class_code_raw = str(row.get("class_code", "")).strip()
            ticker = normalize_text(ticker_raw)
            class_code = normalize_text(class_code_raw)

            if not ticker:
                print(f"[SKIP] {symbol}: ticker is empty")
                skipped_count += 1
                continue

            future, reason = resolve_future_with_fallback(
                client=client,
                symbol=symbol,
                ticker_raw=ticker_raw,
                class_code_raw=class_code_raw,
                futures=futures,
                by_key=by_key,
                by_ticker=by_ticker,
                instrument_type_futures=InstrumentType.INSTRUMENT_TYPE_FUTURES,
                instrument_id_type_uid=InstrumentIdType.INSTRUMENT_ID_TYPE_UID,
            )
            if future is None:
                print(f"[FAIL] {symbol}: {reason}")
                failed_count += 1
                continue

            figi = str(getattr(future, "figi", "") or "").strip()
            uid = str(getattr(future, "uid", "") or "").strip()
            tick_size = price_to_float(getattr(future, "min_price_increment", None))
            tick_value = price_to_float(getattr(future, "min_price_increment_amount", None))

            # Fallback: ask margin endpoint if amount is not present in futures list.
            if tick_value is None:
                instrument_id = uid or figi
                if instrument_id and hasattr(client.instruments, "get_futures_margin"):
                    try:
                        margin = client.instruments.get_futures_margin(instrument_id=instrument_id)
                        tick_value = price_to_float(
                            getattr(margin, "min_price_increment_amount", None)
                        )
                    except Exception:
                        tick_value = None

            if tick_size is None:
                print(f"[FAIL] {symbol}: min_price_increment is empty")
                failed_count += 1
                continue
            if tick_value is None:
                print(f"[FAIL] {symbol}: min_price_increment_amount is empty")
                failed_count += 1
                continue

            changed = (
                row.get("figi") != figi
                or row.get("uid") != uid
                or maybe_float(row.get("tick_size")) != tick_size
                or maybe_float(row.get("tick_value")) != tick_value
            )

            row["figi"] = figi
            row["uid"] = uid
            row["tick_size"] = tick_size
            row["tick_value"] = tick_value

            if changed:
                updated_count += 1
                print(
                    f"[OK] {symbol}: ticker={ticker_raw} class_code={class_code_raw} "
                    f"figi={figi} uid={uid} tick_size={tick_size} tick_value={tick_value}"
                )
            else:
                print(f"[OK] {symbol}: no changes")

    if args.dry_run:
        print(
            f"[DRY-RUN] summary: updated={updated_count}, skipped={skipped_count}, failed={failed_count}"
        )
        return 1 if failed_count else 0

    args.config.write_text(
        yaml.safe_dump(config_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Saved: {args.config}")
    print(f"Summary: updated={updated_count}, skipped={skipped_count}, failed={failed_count}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
