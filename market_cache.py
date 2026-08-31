"""Local, refreshable market-data cache used by backtests.

The cache stores one normalized CSV per instrument with ``date``, ``close``,
and ``pct_change`` columns. Backtests only read this local cache. RQData is
loaded lazily by the explicit ``refresh`` command and is never a backtest
runtime dependency.

Example
-------
    # cache/manifest.json declares symbols and their first_date values.
    uv run python market_cache.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


MARKET_CACHE_FIELDS: tuple[str, str, str] = ("date", "close", "pct_change")
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MarketDataCacheError(ValueError):
    """Raised when a local market cache cannot safely supply a benchmark."""


@dataclass(frozen=True, slots=True)
class CachedInstrument:
    """A configured symbol, its local cache file, and provider identifier."""

    symbol: str
    rq_symbol: str
    path: Path


def _validate_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not _SYMBOL_PATTERN.fullmatch(symbol):
        raise MarketDataCacheError(
            "Benchmark symbol must contain only letters, digits, '.', '_', or '-'."
        )
    return symbol


def _read_manifest(cache_dir: Path) -> Mapping[str, object] | None:
    path = cache_dir.parent / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketDataCacheError(f"Cannot read market-cache manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise MarketDataCacheError(f"Market-cache manifest must be an object: {path}")
    return manifest


def _direct_cache_path(cache_dir: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix.lower() != ".csv":
        raise MarketDataCacheError("Market-cache files must be direct .csv children of the cache directory.")
    return cache_dir / candidate


def resolve_cached_instrument(cache_dir: Path, symbol: str) -> CachedInstrument:
    """Resolve *symbol* through an optional manifest or a direct cache filename.

    A manifest may map either its ``instrument_id`` or its ``rq_symbol`` to a
    cache file. Without a manifest, ``<symbol>.csv`` is used directly. This
    lets a single cache directory contain any number of independent benchmarks.
    """
    symbol = _validate_symbol(symbol)
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)
    instruments = manifest.get("instruments", []) if manifest is not None else []
    if not isinstance(instruments, list):
        raise MarketDataCacheError("Market-cache manifest has no instruments list.")

    matches = [
        item
        for item in instruments
        if isinstance(item, dict)
        and symbol in {item.get("instrument_id"), item.get("rq_symbol")}
    ]
    if len(matches) > 1:
        raise MarketDataCacheError(f"Market-cache manifest maps benchmark symbol {symbol!r} more than once.")
    if matches:
        item = matches[0]
        instrument_id = item.get("instrument_id")
        rq_symbol = item.get("rq_symbol")
        filename = item.get("cache_file")
        if not isinstance(instrument_id, str) or not isinstance(rq_symbol, str):
            raise MarketDataCacheError(f"Invalid market-cache manifest entry for {symbol!r}.")
        if not isinstance(filename, str):
            filename = f"{instrument_id}.csv"
        return CachedInstrument(symbol, rq_symbol, _direct_cache_path(cache_dir, filename))

    return CachedInstrument(symbol, symbol, _direct_cache_path(cache_dir, f"{symbol}.csv"))


def _read_cache_rows(path: Path) -> dict[date, tuple[float, float]]:
    if not path.is_file():
        raise MarketDataCacheError(f"Benchmark cache is missing: {path}")
    rows: dict[date, tuple[float, float]] = {}
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise MarketDataCacheError(f"Cannot read benchmark cache: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(MARKET_CACHE_FIELDS) - set(reader.fieldnames):
            raise MarketDataCacheError(f"{path} must contain columns {MARKET_CACHE_FIELDS}.")
        for line_number, row in enumerate(reader, start=2):
            try:
                observation_date = date.fromisoformat(row["date"])
                close = float(row["close"])
                pct_change = float(row["pct_change"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataCacheError(f"{path.name} row {line_number} is invalid.") from exc
            if (
                observation_date in rows
                or close <= 0.0
                or not math.isfinite(close)
                or not math.isfinite(pct_change)
            ):
                raise MarketDataCacheError(f"{path.name} row {line_number} is invalid or duplicated.")
            rows[observation_date] = (close, pct_change)
    if not rows:
        raise MarketDataCacheError(f"Benchmark cache is empty: {path}")
    return rows


def _parse_bound(value: str | None, *, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise MarketDataCacheError(f"{name} must be an ISO date, got {value!r}.") from exc


def load_benchmark_returns(
    cache_dir: Path,
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Return cached close-to-close daily benchmark returns for *symbol*."""
    start_date = _parse_bound(start, name="start")
    end_date = _parse_bound(end, name="end")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise MarketDataCacheError("start must not be after end.")

    instrument = resolve_cached_instrument(cache_dir, symbol)
    rows = _read_cache_rows(instrument.path)
    selected = [
        (observation_date, pct_change)
        for observation_date, (_, pct_change) in sorted(rows.items())
        if (start_date is None or observation_date >= start_date)
        and (end_date is None or observation_date <= end_date)
    ]
    if not selected:
        raise MarketDataCacheError(
            f"Benchmark {symbol!r} has no cached returns in the requested date range."
        )
    index = pd.DatetimeIndex([observation_date for observation_date, _ in selected], name="date")
    return pd.Series([pct_change for _, pct_change in selected], index=index, name="ret", dtype=float)


def _rqdatac_client() -> object:
    try:
        import rqdatac as rq
    except ImportError as exc:
        raise MarketDataCacheError(
            "Refreshing a market cache requires rqdatac. Install and configure it first."
        ) from exc
    try:
        username = os.environ.get("RQDATAC_USERNAME")
        password = os.environ.get("RQDATAC_PASSWORD")
        rq.init(username, password) if username and password else rq.init()
    except Exception as exc:  # rqdatac exposes provider-specific exception types.
        raise MarketDataCacheError("RQData authentication failed.") from exc
    return rq


def _daily_closes(price_frame: object, rq_symbol: str) -> dict[date, float]:
    if price_frame is None or getattr(price_frame, "empty", False):
        return {}
    try:
        frame = price_frame.reset_index()
    except AttributeError as exc:
        raise MarketDataCacheError(f"RQData returned an unsupported result for {rq_symbol}.") from exc
    if "close" not in frame.columns:
        raise MarketDataCacheError(f"RQData returned no close column for {rq_symbol}.")

    time_column = next(
        (
            column
            for column in frame.columns
            if column != "close" and any(hasattr(value, "date") for value in frame[column])
        ),
        None,
    )
    if time_column is None:
        raise MarketDataCacheError(f"RQData returned no date index for {rq_symbol}.")

    closes: dict[date, float] = {}
    for timestamp, close in zip(frame[time_column], frame["close"]):
        try:
            observation_date = timestamp.date()
            close_value = float(close)
        except (AttributeError, TypeError, ValueError) as exc:
            raise MarketDataCacheError(f"RQData returned invalid close data for {rq_symbol}.") from exc
        if close_value > 0.0 and math.isfinite(close_value):
            closes[observation_date] = close_value
    return closes


def _write_cache_rows(path: Path, closes: Mapping[date, float]) -> None:
    if not closes:
        raise MarketDataCacheError(f"Cannot write an empty benchmark cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    previous_close: float | None = None
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARKET_CACHE_FIELDS)
        writer.writeheader()
        for observation_date, close in sorted(closes.items()):
            pct_change = 0.0 if previous_close is None else close / previous_close - 1.0
            writer.writerow(
                {
                    "date": observation_date.isoformat(),
                    "close": format(close, ".17g"),
                    "pct_change": format(pct_change, ".17g"),
                }
            )
            previous_close = close
    temporary_path.replace(path)


def _refresh_instruments(cache_dir: Path) -> list[tuple[CachedInstrument, date]]:
    """Return manifest-declared cache instruments and their seed dates."""
    manifest = _read_manifest(cache_dir)
    if manifest is None:
        raise MarketDataCacheError(f"Market-cache manifest is missing: {cache_dir.parent / 'manifest.json'}")
    entries = manifest.get("instruments")
    if not isinstance(entries, list) or not entries:
        raise MarketDataCacheError("Market-cache manifest must declare at least one instrument.")

    refreshed: list[tuple[CachedInstrument, date]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("instrument_id"), str):
            raise MarketDataCacheError("Each market-cache manifest entry requires an instrument_id.")
        instrument_id = entry["instrument_id"]
        if instrument_id in seen_ids:
            raise MarketDataCacheError(f"Market-cache manifest repeats instrument_id {instrument_id!r}.")
        seen_ids.add(instrument_id)
        first_date = _parse_bound(entry.get("first_date"), name=f"first_date for {instrument_id}")
        if first_date is None:
            raise MarketDataCacheError(f"Market-cache manifest entry {instrument_id!r} is missing first_date.")
        refreshed.append((resolve_cached_instrument(cache_dir, instrument_id), first_date))
    return refreshed


def refresh_market_cache(
    cache_dir: Path,
    *,
    end: str | None = None,
) -> dict[str, str]:
    """Refresh every instrument declared by the parent cache-root manifest."""
    cache_dir = Path(cache_dir)
    end_date = _parse_bound(end, name="end") or date.today()
    rq = _rqdatac_client()
    result: dict[str, str] = {}
    for instrument, first_date in _refresh_instruments(cache_dir):
        if first_date > end_date:
            raise MarketDataCacheError(
                f"Manifest first_date for {instrument.symbol!r} is after refresh end date."
            )
        existing = _read_cache_rows(instrument.path) if instrument.path.exists() else {}
        closes = {observation_date: close for observation_date, (close, _) in existing.items()}
        query_start = max(first_date, max(closes)) if closes else first_date
        try:
            frame = rq.get_price(
                instrument.rq_symbol,
                start_date=query_start.isoformat(),
                end_date=end_date.isoformat(),
                adjust_type="post_volume",
            )
        except Exception as exc:  # rqdatac exposes provider-specific exception types.
            raise MarketDataCacheError(f"RQData daily download failed for {instrument.rq_symbol}.") from exc
        closes.update(_daily_closes(frame, instrument.rq_symbol))
        if not closes:
            raise MarketDataCacheError(f"RQData returned no usable closes for {instrument.rq_symbol}.")
        _write_cache_rows(instrument.path, closes)
        result[instrument.symbol] = max(closes).isoformat()
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh normalized RQData market caches.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("refresh",),
        default="refresh",
        help="Optional compatibility command; refresh is the default.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/market_data"),
        help="Market-cache directory (default: %(default)s).",
    )
    parser.add_argument("--end", help="Inclusive ISO end date; defaults to today.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "refresh":
        refreshed = refresh_market_cache(args.cache_dir, end=args.end)
        for symbol, last_date in refreshed.items():
            print(f"{symbol}: cache through {last_date}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MarketDataCacheError as exc:
        print(f"Market-cache refresh failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
