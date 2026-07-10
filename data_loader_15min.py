"""
15-minute data loading utilities.

Public API
----------
build_pool_15min(...)  ->  (BacktestDataset15Min, benchmark Series)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from data_loader import load_benchmark

FLAG_COLUMNS = [
    "date",
    "symbol",
    "market_cap_3",
    "log_size",
    "size_rank",
    "industry",
    "index",
    "limit_up_price",
    "limit_down_price",
    "listed_Satisfied",
    "is_ST",
    "normal_days",
    "can_open_base",
]


@dataclass(slots=True)
class BacktestDataset15Min:
    """Compact, array-backed 15-minute market and prediction data."""

    pool_frame: pl.DataFrame
    daily_snapshot_frame: pl.DataFrame
    bars: np.ndarray
    symbols: np.ndarray
    bar_offsets: np.ndarray
    bar_session_index: np.ndarray
    row_symbol_ids: np.ndarray
    vwap_ret: np.ndarray
    close_prev: np.ndarray  # ≈ open_t = close_{t-1} (连续竞价)
    close_curr: np.ndarray  # ≈ open_{t+1} = close_t
    vwap15: np.ndarray
    pred: np.ndarray
    size_rank: np.ndarray
    tradable: np.ndarray
    can_open: np.ndarray
    can_open_base: np.ndarray
    can_trade_buy: np.ndarray
    can_trade_sell: np.ndarray
    sort_cache: dict[bool, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def _normalize_symbol_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in frame.columns:
        return frame
    if "stock_code" in frame.columns:
        return frame.rename({"stock_code": "symbol"})
    raise ValueError("Input frame is missing symbol column ('symbol' or 'stock_code')")


def _prediction_files(preds_dir: Path, horizons: list[str]) -> list[Path]:
    parquet_files = sorted(preds_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet prediction files found in '{preds_dir}'")

    matched = [path for path in parquet_files if any(tag in path.stem for tag in horizons)]
    if matched:
        return matched

    logger.warning("15m predictions: no file matched horizons {}, fallback to all parquet files", horizons)
    return parquet_files


def load_15min_market(path: Path, start: str, end: str | None = None) -> pl.DataFrame:
    """Load 15-minute returns data and derive date from datetime."""
    frame = pl.read_parquet(path)
    frame = _normalize_symbol_column(frame)

    required = {"datetime", "close", "vwap15", "turnover", "open", "vwap_ret"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"15m market parquet must contain: close, vwap15, turnover, open, vwap_ret. Missing: {missing}"
        )

    if frame["datetime"].dtype == pl.String:
        frame = frame.with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
    else:
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None

    out = frame.with_columns(pl.col("datetime").dt.date().alias("date")).filter(pl.col("date") >= pl.lit(start_date))
    if end_date is not None:
        out = out.filter(pl.col("date") <= pl.lit(end_date))

    return out.select(["datetime", "date", "symbol", "vwap_ret", "vwap15", "open", "close", "turnover"]).sort([
        "datetime",
        "symbol",
    ])


def _normalize_prediction_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = set(frame.columns)

    if {"trade_date", "stock_code", "prediction"}.issubset(columns):
        out = frame.select(["trade_date", "stock_code", "prediction"]).rename({
            "trade_date": "datetime",
            "stock_code": "symbol",
            "prediction": "pred",
        })
        return out

    if {"datetime", "symbol", "pred"}.issubset(columns):
        return frame.select(["datetime", "symbol", "pred"])

    if "__index_level_0__" in columns:
        wide = frame
        idx = "__index_level_0__"
        if wide[idx].dtype == pl.String:
            wide = wide.with_columns(pl.col(idx).str.strptime(pl.Datetime, strict=False))
        else:
            wide = wide.with_columns(pl.col(idx).cast(pl.Datetime))

        long = wide.unpivot(index=idx, variable_name="symbol", value_name="pred").rename({idx: "datetime"})
        return long.select(["datetime", "symbol", "pred"])

    raise ValueError(
        "Unsupported prediction parquet schema. Expected either "
        "(trade_date, stock_code, prediction), (datetime, symbol, pred), or wide format with '__index_level_0__'."
    )


def load_15min_predictions(preds_dir: Path, horizons: list[str], start: str, end: str | None = None) -> pl.DataFrame:
    """Load 15-minute predictions and average duplicated (datetime, symbol) rows across files."""
    files = _prediction_files(preds_dir, horizons)
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None

    stacked: list[pl.DataFrame] = []
    for file_path in files:
        frame = _normalize_prediction_frame(pl.read_parquet(file_path))
        if frame["datetime"].dtype == pl.String:
            frame = frame.with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
        else:
            frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))

        frame = frame.with_columns(pl.col("datetime").dt.date().alias("date")).filter(
            pl.col("date") >= pl.lit(start_date)
        )
        if end_date is not None:
            frame = frame.filter(pl.col("date") <= pl.lit(end_date))
        stacked.append(frame.select(["datetime", "symbol", "pred"]))

    logger.info("15m predictions: {} file(s) loaded – {}", len(files), [f.name for f in files])

    return (
        pl
        .concat(stacked, how="vertical")
        .group_by(["datetime", "symbol"], maintain_order=True)
        .agg(pl.col("pred").mean().alias("pred"))
        .sort(["datetime", "symbol"])
    )


def load_daily_flags(
    path: Path,
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
    nosuspend_days: int,
) -> pl.DataFrame:
    """Load daily data and keep daily-level static constraints + limit prices."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None

    can_open_base_expr = pl.col("normal_days") >= nosuspend_days
    # 新增：必须满足 listed_Satisfied == 1
    can_open_base_expr &= pl.col("listed_Satisfied").cast(pl.Boolean)
    if not allow_st_open:
        can_open_base_expr &= ~pl.col("is_ST").fill_null(0).cast(pl.Boolean)
    if universe is not None:
        can_open_base_expr &= pl.col("index").is_in(universe)

    frame = pl.read_parquet(path).filter(pl.col("date") >= pl.lit(start_date))
    if end_date is not None:
        frame = frame.filter(pl.col("date") <= pl.lit(end_date))

    required_daily_cols = {"limit_up_price", "limit_down_price"}
    missing_daily_cols = sorted(required_daily_cols.difference(frame.columns))
    if missing_daily_cols:
        raise ValueError(f"Daily data parquet is missing required columns for 15m limit checks: {missing_daily_cols}")

    frame = (
        frame
        .with_columns(
            can_open_base=can_open_base_expr,
        )
        .select(FLAG_COLUMNS)
        .sort(["date", "symbol"])
    )

    return frame


def _dataset_from_frame(pool: pl.DataFrame, daily_snapshot_frame: pl.DataFrame) -> BacktestDataset15Min:
    symbol_values = pool["symbol"].unique().sort().to_list()
    symbol_map = pl.DataFrame({"symbol": symbol_values}).with_row_index("symbol_id")
    encoded = pool.join(symbol_map, on="symbol", how="left").with_row_index("row_idx")

    bar_counts = encoded.group_by("datetime", maintain_order=True).len().sort("datetime")
    bars = bar_counts["datetime"].to_numpy()
    counts = bar_counts["len"].to_numpy().astype(np.int64, copy=False)

    bar_offsets = np.empty(len(counts) + 1, dtype=np.int64)
    bar_offsets[0] = 0
    bar_offsets[1:] = np.cumsum(counts)

    # Session index: 0=morning (bars before 12:00), 1=afternoon (bars >= 12:00).
    # Mapped per bar, then factorized per (date, session) for T+1 freeze release.
    bar_datetimes = pd.to_datetime(bars)
    bar_session_half = (bar_datetimes.hour >= 12).astype(np.int32)
    bar_date_session = bar_datetimes.normalize().astype("int64") // 10**9 * 10 + bar_session_half
    bar_session_index = pd.factorize(bar_date_session)[0].astype(np.int32)

    # Compute close_prev and close_curr from actual close prices
    # close_prev = close of previous bar (shift +1 within each symbol)
    # close_curr = close of current bar
    encoded = encoded.sort(["symbol", "datetime"]).with_columns([
        pl.col("close").shift(1).over("symbol").alias("close_prev"),
        pl.col("close").alias("close_curr"),
    ])
    # Fill null close_prev (first bar per symbol) with open as fallback
    encoded = encoded.with_columns(pl.col("close_prev").fill_null(pl.col("open")))
    # Re-sort to original order (by datetime, symbol)
    encoded = encoded.sort(["datetime", "symbol"])

    return BacktestDataset15Min(
        pool_frame=encoded,
        daily_snapshot_frame=daily_snapshot_frame,
        bars=bars,
        symbols=np.asarray(symbol_values, dtype=object),
        bar_offsets=bar_offsets,
        bar_session_index=bar_session_index,
        row_symbol_ids=encoded["symbol_id"].to_numpy().astype(np.int32, copy=False),
        vwap_ret=encoded["vwap_ret"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        close_prev=encoded["close_prev"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        close_curr=encoded["close_curr"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        vwap15=encoded["vwap15"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        pred=encoded["pred"].fill_null(float("nan")).to_numpy().astype(np.float64, copy=False),
        size_rank=encoded["size_rank"].fill_null(999999).to_numpy().astype(np.int32, copy=False),
        tradable=encoded["tradable"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open=encoded["can_open"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open_base=encoded["can_open_base"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_buy=encoded["can_trade_buy"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_sell=encoded["can_trade_sell"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
    )


def build_pool_15min(
    data_15min_path: Path,
    preds_15min_dir: Path,
    daily_data_path: Path,
    bm_path: Path,
    horizons_15min: list[str],
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
    nosuspend_days: int,
) -> tuple[BacktestDataset15Min, pd.Series]:
    """
    Build 15-minute backtest dataset with daily constraint flags broadcast by (date, symbol).

    Returns
    -------
    pool_15m : BacktestDataset15Min
    bm_ret   : daily benchmark return Series
    """
    bm_ret = load_benchmark(bm_path)

    market_15m = load_15min_market(data_15min_path, start, end=end)
    daily_flags = load_daily_flags(daily_data_path, start, end, universe, allow_st_open, nosuspend_days)
    preds_15m = load_15min_predictions(preds_15min_dir, horizons_15min, start, end=end)
    # Match the external candidate-pool format:
    # first filter by the local daily base universe definition, then rank by
    # market_cap_3 within each date to get the top-POOL_SIZE names.
    # Intraday tradability such as turnover or limit hits must not affect rank.
    valid_universe_expr = (
        (pl.col("normal_days") >= nosuspend_days)
        & pl.col("listed_Satisfied").cast(pl.Boolean)
        & ~pl.col("is_ST").fill_null(0).cast(pl.Boolean)
    )
    if universe is not None:
        valid_universe_expr &= pl.col("index").is_in(universe)

    eligible_ranked_daily = (
        daily_flags
        .filter(valid_universe_expr & pl.col("market_cap_3").is_finite())
        .select(["date", "symbol", "market_cap_3"])
        .with_columns(
            pl.col("market_cap_3").rank(descending=True).over("date").cast(pl.Int32).alias("size_rank")
        )
        .select(["date", "symbol", "size_rank"])
    )
    ranked_daily = (
        daily_flags
        .drop("size_rank")
        .join(eligible_ranked_daily, on=["date", "symbol"], how="left")
        .with_columns(
            pl.col("size_rank").fill_null(999999).cast(pl.Int32)
        )
    )

    pool = (
        market_15m
        .join(ranked_daily, on=["date", "symbol"], how="left")
        .join(preds_15m, on=["datetime", "symbol"], how="left")
        .with_columns(
            limit_up_hit=((pl.col("vwap15") - pl.col("limit_up_price")).abs() <= 0.0005).fill_null(False),
            limit_down_hit=((pl.col("vwap15") - pl.col("limit_down_price")).abs() <= 0.0005).fill_null(False),
        )
        .with_columns(
            can_trade_buy=((pl.col("turnover") > 0) & ~pl.col("limit_up_hit")).fill_null(False),
            can_trade_sell=((pl.col("turnover") > 0) & ~pl.col("limit_down_hit")).fill_null(False),
        )
        .with_columns(
            is_limit_up=pl.col("limit_up_hit"),
            is_limit_down=pl.col("limit_down_hit"),
        )
        .with_columns(
            tradable=(pl.col("can_trade_buy") & pl.col("can_trade_sell")).fill_null(False),
        )
        .with_columns(
            can_open=(pl.col("tradable") & pl.col("can_open_base")).fill_null(False),
        )
        .drop(["limit_up_hit", "limit_down_hit"])
        .sort(["datetime", "symbol"])
    )

    dataset = _dataset_from_frame(pool, ranked_daily)
    return dataset, bm_ret
