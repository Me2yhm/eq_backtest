"""
Data loading utilities.

Public API
----------
build_pool(...)  ->  (BacktestDataset, benchmark Series)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

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

POOL_CACHE_VERSION = 2
CACHED_POOL_REQUIRED_COLUMNS = MARKET_REQUIRED_COLUMNS | {
    "pred",
    "tradable",
    "can_open",
    "can_trade_buy",
    "can_trade_sell",
    "can_open_base",
    "row_idx",
    "symbol_id",
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
    can_trade_buy: np.ndarray
    can_trade_sell: np.ndarray
    can_open_base: np.ndarray
    sort_cache: dict[bool, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


# ── Benchmark ─────────────────────────────────────────────────────────────────


def load_benchmark(path: Path) -> pd.Series:
    """Load a CSV of daily benchmark returns into a Series."""
    frame = pl.read_csv(path)
    if "date" not in frame.columns:
        frame = frame.with_columns(pl.col(frame.columns[0]).cast(pl.Date).alias("date"))
    elif frame["date"].dtype == pl.String:
        frame = frame.with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%Y", strict=False))

    result = frame.sort("date").to_pandas().set_index("date")["ret"].rename("ret")
    return pd.Series(result)


# ── Predictions ───────────────────────────────────────────────────────────────


def _horizon_suffix(path: Path) -> str:
    """Extract the horizon tag after the last underscore (e.g. '3d' from 'model_3d.parquet')."""
    return path.stem.rsplit("_", 1)[-1] if "_" in path.stem else ""


def _prediction_files(preds_dir: Path, horizons: list) -> list[Path]:
    files = sorted(f for f in preds_dir.glob("*.parquet") if _horizon_suffix(f) in horizons)
    if not files:
        raise FileNotFoundError(f"No parquet files matching horizons {horizons} found in '{preds_dir}'")
    return files


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
    files = _prediction_files(preds_dir, horizons)
    logger.info("Predictions: {} file(s) loaded – {}", len(files), [f.name for f in files])

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
    tradable      : has volume AND is not at either price limit
    can_trade_buy : buy-side execution allowed on this day
    can_trade_sell: sell-side execution allowed on this day
    can_open      : symmetric open flag retained for diagnostics/backward compatibility
    """
    _validate_market_schema(path)
    start_date = date.fromisoformat(start)
    can_trade_buy_expr = (pl.col("turnover") > 0) & ~pl.col("is_limit_up").cast(pl.Boolean)
    can_trade_sell_expr = (pl.col("turnover") > 0) & ~pl.col("is_limit_down").cast(pl.Boolean)
    tradable_expr = can_trade_buy_expr & can_trade_sell_expr
    can_open_base_expr = pl.col("normal_days") >= 10
    if not allow_st_open:
        can_open_base_expr &= ~pl.col("is_ST").fill_null(0).cast(pl.Boolean)
    if universe is not None:
        can_open_base_expr &= pl.col("index").is_in(universe)

    can_open_expr = tradable_expr & can_open_base_expr

    return (
        pl.read_parquet(path)
        .filter(pl.col("date") >= pl.lit(start_date))
        .with_columns(
            tradable=tradable_expr,
            can_open=can_open_expr,
            can_trade_buy=can_trade_buy_expr,
            can_trade_sell=can_trade_sell_expr,
            can_open_base=can_open_base_expr,
        )
        .sort(["date", "symbol"])
    )


# ── Combined entry point ───────────────────────────────────────────────────────


def _dataset_from_encoded_frame(encoded: pl.DataFrame) -> BacktestDataset:
    missing = sorted(CACHED_POOL_REQUIRED_COLUMNS.difference(encoded.columns))
    if missing:
        raise ValueError(f"Cached pool frame is missing required columns {missing}")

    symbol_table = encoded.select(["symbol_id", "symbol"]).unique(maintain_order=True).sort("symbol_id")
    symbol_values = symbol_table["symbol"].to_list()

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
        can_trade_buy=encoded["can_trade_buy"].to_numpy().astype(np.bool_, copy=False),
        can_trade_sell=encoded["can_trade_sell"].to_numpy().astype(np.bool_, copy=False),
        can_open_base=encoded["can_open_base"].to_numpy().astype(np.bool_, copy=False),
    )


def _encode_dataset(pool: pl.DataFrame) -> BacktestDataset:
    symbol_values = pool["symbol"].unique().sort().to_list()
    symbol_map = pl.DataFrame({"symbol": symbol_values}).with_row_index("symbol_id")
    encoded = pool.join(symbol_map, on="symbol", how="left").with_row_index("row_idx")

    return _dataset_from_encoded_frame(encoded)


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _pool_cache_path(
    cache_dir: Path,
    data_path: Path,
    pred_files: list[Path],
    start: str,
    universe: list | None,
    allow_st_open: bool,
) -> Path:
    payload = {
        "version": POOL_CACHE_VERSION,
        "data": _file_signature(data_path),
        "predictions": [_file_signature(path) for path in pred_files],
        "start": start,
        "universe": list(universe) if universe is not None else None,
        "allow_st_open": allow_st_open,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"pool_{digest}.parquet"


def _read_cached_pool(cache_path: Path) -> BacktestDataset | None:
    if not cache_path.exists():
        return None

    try:
        encoded = pl.read_parquet(cache_path)
        return _dataset_from_encoded_frame(encoded)
    except Exception:
        return None


def build_pool(
    data_path: Path,
    preds_dir: Path,
    bm_path: Path,
    horizons: list,
    start: str,
    universe: list | None,
    allow_st_open: bool,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> tuple[BacktestDataset, pd.Series]:
    """
    Assemble the full pool and benchmark return series.

    Returns
    -------
    pool   : BacktestDataset – market data joined with predictions and encoded for Numba
    bm_ret : Series of daily benchmark returns
    """
    bm_ret = load_benchmark(bm_path)

    pred_files = _prediction_files(preds_dir, horizons)
    resolved_cache_dir = cache_dir if cache_dir is not None else data_path.parent / ".cache"
    cache_path = _pool_cache_path(resolved_cache_dir, data_path, pred_files, start, universe, allow_st_open)

    if use_cache:
        cached_dataset = _read_cached_pool(cache_path)
        if cached_dataset is not None:
            logger.debug("Pool cache: hit – {}", cache_path.name)
            return cached_dataset, bm_ret
        logger.debug("Pool cache: miss – {}", cache_path.name)

    logger.info("Predictions: {} file(s) loaded – {}", len(pred_files), [f.name for f in pred_files])
    try:
        preds = _load_predictions_wide_mean(pred_files, start)
    except PredictionAlignmentError:
        preds = _load_predictions_long_form(pred_files, start)

    market = load_market_data(data_path, start, universe, allow_st_open)
    pool = market.join(preds, on=["date", "symbol"], how="left")
    dataset = _encode_dataset(pool)

    if use_cache:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.pool_frame.write_parquet(cache_path)
        logger.debug("Pool cache: wrote – {}", cache_path.name)

    return dataset, bm_ret


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
        use_cache=cfg.USE_POOL_CACHE,
        cache_dir=cfg.POOL_CACHE_DIR,
    )
    logger.debug("bm_ret head:\n{}", bm_ret.head())
    logger.debug("pool_frame head:\n{}", dataset.pool_frame.head())
