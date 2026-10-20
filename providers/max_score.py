"""Explicit, time-filtered long-form exposure sources for max_score.

Sources are never searched or silently substituted. Dates identify observation
dates; an optional availability column independently qualifies publication time.
Without publication evidence the snapshot remains explicitly unverified.
"""

from __future__ import annotations

from pathlib import Path
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import polars as pl


def canonical_symbol(value: object) -> str:
    text = str(value)
    return text.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")


class SnapshotDataError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def validate_source(source: object, *, industry: bool = False) -> dict:
    if not isinstance(source, dict):
        raise ValueError("exposure source must be an object")
    allowed = {"path", "paths", "date_column", "symbol_column", "value_column",
               "available_at_column", "metadata"}
    if set(source).difference(allowed):
        raise ValueError(f"unknown exposure source fields: {sorted(set(source).difference(allowed))}")
    if ("path" in source) == ("paths" in source):
        raise ValueError("exposure source requires exactly one of path or paths")
    paths = source.get("paths", [source.get("path")])
    if not isinstance(paths, list) or not paths or any(not isinstance(p, (str, Path)) for p in paths):
        raise ValueError("exposure paths must be a non-empty explicit list")
    for key in ("date_column", "symbol_column", "value_column"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"exposure source requires {key}")
    if source.get("available_at_column") is not None and not isinstance(source["available_at_column"], str):
        raise ValueError("available_at_column must be a column name or null")
    metadata = source.get("metadata")
    required = {"exposure_id", "source", "version", "units", "transformation", "semantic_status"}
    if industry:
        required |= {"classification", "level"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError(f"exposure metadata requires {sorted(required)}")
    if any(not isinstance(metadata[k], str) or not metadata[k] for k in required):
        raise ValueError("exposure metadata values must be non-empty strings")
    if metadata["transformation"] != "identity":
        raise ValueError("provide explicitly transformed data; provider only supports identity")
    return source


class LongExposureProvider:
    """Load exactly selected Parquets with date/symbol predicate pushdown."""

    def __init__(self, source: dict, *, start: pd.Timestamp, end: pd.Timestamp, industry: bool = False):
        self.source = validate_source(source, industry=industry)
        self.industry = industry
        date_col, symbol_col, value_col = (source[k] for k in ("date_column", "symbol_column", "value_column"))
        availability = source.get("available_at_column")
        columns = list(dict.fromkeys([date_col, symbol_col, value_col] + ([availability] if availability else [])))
        paths = source.get("paths", [source.get("path")])
        scans = []
        for path in paths:
            scan = pl.scan_parquet(path)
            missing = set(columns).difference(scan.collect_schema().names())
            if missing:
                raise SnapshotDataError("MISSING_EXPOSURE_COLUMN", str(sorted(missing)))
            schema = scan.collect_schema()
            if schema[symbol_col] != pl.String:
                raise SnapshotDataError("INVALID_EXPOSURE_KEY", "symbol column must contain strings preserving exchange and leading zeros")
            date_expr = (pl.col(date_col).str.to_date() if schema[date_col] == pl.String
                         else pl.col(date_col).cast(pl.Date))
            if scan.select(date_expr.is_null().any()).collect().item():
                raise SnapshotDataError("INVALID_EXPOSURE_KEY", "null observation date")
            scans.append(scan.select(columns).with_columns(date_expr.alias("_date"))
                         .filter(pl.col("_date").is_between(start.date(), end.date())))
        frame = pl.concat(scans).collect()
        if frame.select((pl.col(symbol_col).is_null() | (pl.col(symbol_col).str.strip_chars() == "")).any()).item():
            raise SnapshotDataError("INVALID_EXPOSURE_KEY", "null/empty symbol")
        frame = frame.with_columns(pl.col(symbol_col).cast(pl.String).map_elements(
            canonical_symbol, return_dtype=pl.String).alias("_symbol"))
        if frame.select(pl.struct(["_date", "_symbol"]).is_duplicated().any()).item():
            raise SnapshotDataError("DUPLICATE_EXPOSURE_KEY", "duplicate observation date/symbol")
        self.frame = frame

    def snapshot(self, date: pd.Timestamp, symbols: list[str], *, as_of: pd.Timestamp) -> np.ndarray:
        frame = self.frame.filter(pl.col("_date") == date.date())
        availability = self.source.get("available_at_column")
        if availability:
            # Never turn a naive source timestamp into an invented UTC time.
            known_times = []
            for value in frame[availability].to_list():
                try:
                    stamp = pd.Timestamp(value)
                except (TypeError, ValueError) as exc:
                    raise SnapshotDataError("INVALID_AVAILABLE_AT", "invalid publication timestamp") from exc
                if pd.isna(stamp) or stamp.tzinfo is None:
                    raise SnapshotDataError("INVALID_AVAILABLE_AT", "publication timestamp must include a timezone")
                known_times.append(stamp.tz_convert("UTC"))
            times = pd.DatetimeIndex(known_times, tz="UTC")
            cutoff = as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize("Asia/Shanghai").tz_convert("UTC")
            frame = frame.filter(pl.Series(np.asarray(times.notna() & (times <= cutoff))))
        values = dict(zip(frame["_symbol"].to_list(), frame[self.source["value_column"]].to_list()))
        missing = [s for s in symbols if values.get(canonical_symbol(s)) is None]
        if missing:
            raise SnapshotDataError("MISSING_INDUSTRY" if self.industry else "MISSING_SIZE_EXPOSURE",
                                    f"{len(missing)}/{len(symbols)} assets; first={missing[:3]}")
        raw = [values[canonical_symbol(s)] for s in symbols]
        if self.industry:
            converted = []
            for value in raw:
                if isinstance(value, (bool, np.bool_)):
                    raise SnapshotDataError("INVALID_INDUSTRY", "boolean industry code")
                if isinstance(value, (float, np.floating)):
                    if not np.isfinite(value) or value != int(value) or value <= 0:
                        raise SnapshotDataError("INVALID_INDUSTRY", "numeric industry codes must be positive integers")
                    value = str(int(value))
                elif isinstance(value, (int, np.integer)):
                    if value <= 0:
                        raise SnapshotDataError("INVALID_INDUSTRY", "numeric industry codes must be positive integers")
                    value = str(value)
                elif not isinstance(value, str):
                    raise SnapshotDataError("INVALID_INDUSTRY", "industry must be an exact integer code or string ID")
                if value.strip().lower() in {"", "none", "null", "nan", "inf", "+inf", "-inf", "infinity"}:
                    raise SnapshotDataError("INVALID_INDUSTRY", "unknown/zero industry code")
                try:
                    number = Decimal(value)
                except InvalidOperation:
                    pass  # Explicit string category IDs, such as 'bank', are valid.
                else:
                    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
                        raise SnapshotDataError("INVALID_INDUSTRY", "numeric industry codes must be positive integers")
                    # String IDs retain their exact spelling; no float roundtrip.
                converted.append(value)
            return np.asarray(converted, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in raw):
            raise SnapshotDataError("INVALID_SIZE_EXPOSURE", "boolean exposure is not numeric data")
        try:
            result = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SnapshotDataError("INVALID_SIZE_EXPOSURE", "non-numeric exposure") from exc
        if not np.isfinite(result).all():
            raise SnapshotDataError("MISSING_SIZE_EXPOSURE", "non-finite exposure")
        return result
