"""
Data loading utilities.

Public API
----------
build_pool(...)  ->  (BacktestDataset, benchmark Series)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

MARKET_REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "turnover",
    "log_size",
    "size_rank",
    "industry",
    "index",
    "ret",
    "is_limit_up",
    "is_limit_down",
    "listed_Satisfied",
    "is_ST",
    "normal_days",
}


@dataclass(slots=True)
class BacktestDataset:
    """Compact, array-backed market and prediction data for the simulator."""

    pool_frame: pl.DataFrame
    dates: np.ndarray
    symbols: np.ndarray
    day_offsets: np.ndarray
    row_symbol_ids: np.ndarray
    ret: np.ndarray
    pred: np.ndarray
    size_rank: np.ndarray
    tradable: np.ndarray
    can_open: np.ndarray
    sort_cache: dict[bool, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


# ── Benchmark ─────────────────────────────────────────────────────────────────


def load_benchmark(path: Path) -> pd.Series:
    """Load a CSV of daily benchmark returns into a Series."""
    frame = pl.read_csv(path)
    if frame["date"].dtype == pl.String:
        frame = frame.with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%Y", strict=False))

    result = frame.sort("date").to_pandas().set_index("date")["ret"].rename("ret")
    return pd.Series(result)


# ── Predictions ───────────────────────────────────────────────────────────────


def _horizon_suffix(path: Path) -> str:
    """Extract the horizon tag after the last underscore (e.g. '3d' from 'model_3d.parquet')."""
    return path.stem.rsplit("_", 1)[-1] if "_" in path.stem else ""


def _normalize_prediction_dates(frame: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    index_name = "__index_level_0__"
    if index_name not in frame.columns:
        raise ValueError("Prediction parquet is missing the pandas index column '__index_level_0__'")

    if frame[index_name].dtype == pl.String:
        frame = frame.with_columns(pl.col(index_name).str.strptime(pl.Date, strict=False))
    else:
        frame = frame.with_columns(pl.col(index_name).cast(pl.Date))
    return frame, index_name


class PredictionAlignmentError(ValueError):
    pass


def _load_predictions_long_form(files: list[Path], start: str) -> pl.DataFrame:
    start_date = date.fromisoformat(start)
    stacked: list[pl.DataFrame] = []
    for file_path in files:
        wide, index_name = _normalize_prediction_dates(pl.read_parquet(file_path))
        long = (
            wide.filter(pl.col(index_name) >= pl.lit(start_date))
            .unpivot(index=index_name, variable_name="symbol", value_name="pred")
            .rename({index_name: "date"})
        )
        stacked.append(long)

    return (
        pl.concat(stacked, how="vertical")
        .group_by(["date", "symbol"], maintain_order=True)
        .agg(pl.col("pred").mean().alias("pred"))
        .sort(["date", "symbol"])
    )


def _load_predictions_wide_mean(files: list[Path], start: str) -> pl.DataFrame:
    start_date = date.fromisoformat(start)

    first, index_name = _normalize_prediction_dates(pl.read_parquet(files[0]))
    first = first.filter(pl.col(index_name) >= pl.lit(start_date))
    value_columns = [column for column in first.columns if column != index_name]
    base_dates = first[index_name].to_numpy()
    base_values = first.select(value_columns).to_numpy()
    sums = np.nan_to_num(base_values, copy=True, nan=0.0)
    counts = np.isfinite(base_values).astype(np.int16)

    for file_path in files[1:]:
        frame, current_index_name = _normalize_prediction_dates(pl.read_parquet(file_path))
        frame = frame.filter(pl.col(current_index_name) >= pl.lit(start_date))

        if frame.columns != first.columns:
            raise PredictionAlignmentError("Prediction parquet schemas differ across horizons")

        current_dates = frame[current_index_name].to_numpy()
        if not np.array_equal(current_dates, base_dates):
            raise PredictionAlignmentError("Prediction parquet date indexes differ across horizons")

        values = frame.select(value_columns).to_numpy()
        valid = np.isfinite(values)
        sums += np.where(valid, values, 0.0)
        counts += valid.astype(np.int16)

    mean_values = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )

    averaged = pl.DataFrame(mean_values, schema=value_columns)
    averaged.insert_column(0, pl.Series(index_name, base_dates))
    return averaged.unpivot(index=index_name, variable_name="symbol", value_name="pred").rename({index_name: "date"})


def load_predictions(preds_dir: Path, horizons: list, start: str) -> pl.DataFrame:
    """
    Load parquet prediction files matching the given horizon suffixes and average them.

    Each parquet file must be a wide DataFrame (index=date, columns=symbol).
    Returns a long Polars DataFrame with columns [date, symbol, pred].
    """
    files = [f for f in preds_dir.glob("*.parquet") if _horizon_suffix(f) in horizons]
    if not files:
        raise FileNotFoundError(f"No parquet files matching horizons {horizons} found in '{preds_dir}'")
    print(f"Predictions: {len(files)} file(s) loaded – {[f.name for f in files]}")

    try:
        return _load_predictions_wide_mean(files, start)
    except PredictionAlignmentError:
        return _load_predictions_long_form(files, start)


def _validate_market_schema(path: Path) -> None:
    schema = pl.read_parquet_schema(path)
    missing = sorted(MARKET_REQUIRED_COLUMNS.difference(schema))
    if not missing:
        return

    columns = list(schema)
    hint = ""
    if "__index_level_0__" in schema and "date" not in schema and "symbol" not in schema:
        hint = " The file looks like a wide prediction matrix; point data_path at long-form market data such as 'data/daily.pqt'."

    raise ValueError(
        f"Market data parquet '{path}' is missing required columns {missing}. First columns: {columns[:10]}.{hint}"
    )


# ── Market data ───────────────────────────────────────────────────────────────


def load_market_data(
    path: Path,
    start: str,
    universe: list | None,
    allow_st_open: bool,
) -> pl.DataFrame:
    """
    Load daily market data parquet and attach tradability flags.

    Columns added
    -------------
    tradable  : has volume AND open price is not at a limit
    can_open  : tradable + ≥10 consecutive normal days + optionally not ST + in universe
    """
    _validate_market_schema(path)
    start_date = date.fromisoformat(start)
    tradable_expr = (
        (pl.col("turnover") > 0) & ~pl.col("is_limit_up").cast(pl.Boolean) & ~pl.col("is_limit_down").cast(pl.Boolean)
    )
    can_open_expr = tradable_expr & (pl.col("normal_days") >= 10)
    if not allow_st_open:
        can_open_expr &= ~pl.col("is_ST").fill_null(0).cast(pl.Boolean)
    if universe is not None:
        can_open_expr &= pl.col("index").is_in(universe)

    return (
        pl.read_parquet(path)
        .filter(pl.col("date") >= pl.lit(start_date))
        .with_columns(
            tradable=tradable_expr,
            can_open=can_open_expr,
        )
        .sort(["date", "symbol"])
    )


# ── Combined entry point ───────────────────────────────────────────────────────


def _encode_dataset(pool: pl.DataFrame) -> BacktestDataset:
    symbol_values = pool["symbol"].unique().sort().to_list()
    symbol_map = pl.DataFrame({"symbol": symbol_values}).with_row_index("symbol_id")
    encoded = pool.join(symbol_map, on="symbol", how="left").with_row_index("row_idx")

    date_counts = encoded.group_by("date", maintain_order=True).len()
    dates = date_counts["date"].to_numpy()
    counts = date_counts["len"].to_numpy().astype(np.int64, copy=False)

    day_offsets = np.empty(len(counts) + 1, dtype=np.int64)
    day_offsets[0] = 0
    day_offsets[1:] = np.cumsum(counts)

    return BacktestDataset(
        pool_frame=encoded,
        dates=dates,
        symbols=np.asarray(symbol_values, dtype=object),
        day_offsets=day_offsets,
        row_symbol_ids=encoded["symbol_id"].to_numpy().astype(np.int32, copy=False),
        ret=encoded["ret"].to_numpy().astype(np.float64, copy=False),
        pred=encoded["pred"].fill_null(float("nan")).to_numpy().astype(np.float64, copy=False),
        size_rank=encoded["size_rank"].to_numpy().astype(np.int32, copy=False),
        tradable=encoded["tradable"].to_numpy().astype(np.bool_, copy=False),
        can_open=encoded["can_open"].to_numpy().astype(np.bool_, copy=False),
    )


def build_pool(
    data_path: Path,
    preds_dir: Path,
    bm_path: Path,
    horizons: list,
    start: str,
    universe: list | None,
    allow_st_open: bool,
) -> tuple[BacktestDataset, pd.Series]:
    """
    Assemble the full pool and benchmark return series.

    Returns
    -------
    pool   : BacktestDataset – market data joined with predictions and encoded for Numba
    bm_ret : Series of daily benchmark returns
    """
    bm_ret = load_benchmark(bm_path)
    preds = load_predictions(preds_dir, horizons, start)
    market = load_market_data(data_path, start, universe, allow_st_open)
    pool = market.join(preds, on=["date", "symbol"], how="left")
    return _encode_dataset(pool), bm_ret


if __name__ == "__main__":
    import config as cfg

    dataset, bm_ret = build_pool(
        data_path=cfg.DATA_PATH,
        preds_dir=cfg.PREDS_DIR,
        bm_path=cfg.BM_PATH,
        horizons=cfg.HORIZONS,
        start=cfg.START,
        universe=cfg.UNIVERSE,
        allow_st_open=cfg.ALLOW_ST_OPEN,
    )
    print(bm_ret.head())
    print(dataset.pool_frame.head())
