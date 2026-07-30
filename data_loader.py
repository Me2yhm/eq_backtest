"""Frequency-independent market, prediction, and pool loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

POOL_CACHE_VERSION = 4
_BASE_COLUMNS = {
    "datetime",
    "date",
    "symbol",
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
    "pred",
    "tradable",
    "can_open",
    "can_open_base",
    "can_trade_buy",
    "can_trade_sell",
    "row_idx",
    "symbol_id",
}


@dataclass(slots=True)
class BacktestDataset:
    """Array-backed input shared by daily and intraday simulation."""

    pool_frame: pl.DataFrame
    bars: np.ndarray
    symbols: np.ndarray
    bar_offsets: np.ndarray
    row_symbol_ids: np.ndarray
    ret: np.ndarray
    pred: np.ndarray
    size_rank: np.ndarray
    tradable: np.ndarray
    can_open: np.ndarray
    can_open_base: np.ndarray
    can_trade_buy: np.ndarray
    can_trade_sell: np.ndarray
    sort_cache: dict[bool, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def load_benchmark(path: Path) -> pd.Series:
    frame = pl.read_csv(path)
    if "date" not in frame.columns:
        frame = frame.with_columns(pl.col(frame.columns[0]).cast(pl.Date).alias("date"))
    elif frame["date"].dtype == pl.String:
        frame = frame.with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%Y", strict=False))
    return pd.Series(frame.sort("date").to_pandas().set_index("date")["ret"], name="ret")


def load_external_benchmark(nav_path: Path) -> pd.Series:
    nav = pd.read_parquet(nav_path)
    result = nav["benchmark_return"].groupby(nav.index.normalize()).sum()
    result.index.name, result.name = "date", "ret"
    return result


def _frequency_name(freq_cfg: dict) -> str:
    """Resolve the configured frequency without duplicating it in every config item."""
    market_path = Path(freq_cfg["market_data"])
    name = market_path.name.lower()
    if "15min" in name or "15m" in name:
        return "15min"
    if "5min" in name or "5m" in name:
        return "5min"
    return "daily"


def _normalize_symbol_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in frame.columns:
        return frame
    if "stock_code" in frame.columns:
        return frame.rename({"stock_code": "symbol"})
    raise ValueError("Input frame is missing 'symbol' (or legacy 'stock_code') column")


def _prediction_files(preds_dir: Path, horizons: list[str]) -> list[Path]:
    files = sorted(preds_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet prediction files found in '{preds_dir}'")
    selected = [path for path in files if any(horizon in path.stem for horizon in horizons)]
    if selected:
        return selected
    # Intraday producers commonly use a fixed filename, so a one-file directory is valid.
    if len(files) == 1:
        return files
    raise FileNotFoundError(f"No prediction files in '{preds_dir}' match horizons {horizons}")


def _as_datetime(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = frame[column].dtype
    if dtype == pl.String:
        return frame.with_columns(pl.col(column).str.strptime(pl.Datetime, strict=False).alias(column))
    return frame.with_columns(pl.col(column).cast(pl.Datetime).alias(column))


def _filter_dates(frame: pl.DataFrame, datetime_column: str, start: str, end: str | None) -> pl.DataFrame:
    start_date = date.fromisoformat(start)
    out = frame.with_columns(pl.col(datetime_column).dt.date().alias("__date"))
    out = out.filter(pl.col("__date") >= pl.lit(start_date))
    if end:
        out = out.filter(pl.col("__date") <= pl.lit(date.fromisoformat(end)))
    return out.drop("__date")


def _normalize_prediction_frame(frame: pl.DataFrame, frequency: str) -> pl.DataFrame:
    columns = set(frame.columns)
    if {"trade_date", "stock_code", "prediction"}.issubset(columns):
        return _as_datetime(
            frame.select(["trade_date", "stock_code", "prediction"]).rename({
                "trade_date": "datetime",
                "stock_code": "symbol",
                "prediction": "pred",
            }),
            "datetime",
        )
    frame = (
        _normalize_symbol_column(frame) if "symbol" not in frame.columns and "stock_code" in frame.columns else frame
    )
    columns = set(frame.columns)
    if {"datetime", "symbol", "pred"}.issubset(columns):
        return _as_datetime(frame.select(["datetime", "symbol", "pred"]), "datetime")
    if {"datetime", "symbol", "prediction"}.issubset(columns):
        return _as_datetime(
            frame.select(["datetime", "symbol", "prediction"]).rename({"prediction": "pred"}), "datetime"
        )
    index_name = "__index_level_0__"
    if index_name not in columns:
        raise ValueError(
            "Unsupported prediction schema; expected long [datetime, symbol, pred] or a wide pandas parquet"
        )
    wide = _as_datetime(frame, index_name)
    return wide.unpivot(index=index_name, variable_name="symbol", value_name="pred").rename({index_name: "datetime"})


def load_predictions(freq_cfg: dict, start: str, end: str | None = None) -> pl.DataFrame:
    """Load every frequency into the canonical ``[datetime, symbol, pred]`` form."""
    frequency = _frequency_name(freq_cfg)
    files = _prediction_files(Path(freq_cfg["preds_dir"]), list(freq_cfg["horizons"]))
    frames = [_load_prediction_file(path, frequency, start, end) for path in files]
    logger.info("{} predictions: {} file(s) loaded – {}", frequency, len(files), [path.name for path in files])
    return (
        pl
        .concat(frames, how="vertical_relaxed")
        .group_by(["datetime", "symbol"], maintain_order=True)
        .agg(pl.col("pred").mean().alias("pred"))
        .sort(["datetime", "symbol"])
    )


def _load_prediction_file(path: Path, frequency: str, start: str, end: str | None) -> pl.DataFrame:
    """Read only the requested date range from a long prediction parquet.

    Wide daily matrices are intentionally handled by the established in-memory
    route: predicate pushdown cannot remove their symbol columns.
    """
    schema = pl.read_parquet_schema(path)
    columns = set(schema)
    if {"trade_date", "stock_code", "prediction"}.issubset(columns):
        lazy = pl.scan_parquet(path).select([
            pl.col("trade_date").cast(pl.Datetime).alias("datetime"),
            pl.col("stock_code").alias("symbol"),
            pl.col("prediction").alias("pred"),
        ])
    elif {"datetime", "symbol", "pred"}.issubset(columns):
        lazy = pl.scan_parquet(path).select([pl.col("datetime"), pl.col("symbol"), pl.col("pred")])
    elif {"datetime", "symbol", "prediction"}.issubset(columns):
        lazy = pl.scan_parquet(path).select([pl.col("datetime"), pl.col("symbol"), pl.col("prediction").alias("pred")])
    else:
        return _filter_dates(_normalize_prediction_frame(pl.read_parquet(path), frequency), "datetime", start, end)

    datetime_dtype = lazy.collect_schema()["datetime"]
    if datetime_dtype == pl.String:
        lazy = lazy.with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
    else:
        lazy = lazy.with_columns(pl.col("datetime").cast(pl.Datetime))
    lower = datetime.fromisoformat(start)
    lazy = lazy.filter(pl.col("datetime") >= pl.lit(lower))
    if end:
        lazy = lazy.filter(pl.col("datetime") < pl.lit(datetime.fromisoformat(end) + timedelta(days=1)))
    return lazy.collect()


def _can_open_base_expr(universe: list[str] | None, allow_st_open: bool, nosuspend_days: int) -> pl.Expr:
    expression = (pl.col("normal_days") >= nosuspend_days) & pl.col("listed_Satisfied").cast(pl.Boolean)
    if not allow_st_open:
        expression &= ~pl.col("is_ST").fill_null(0).cast(pl.Boolean)
    if universe is not None:
        expression &= pl.col("index").is_in(universe)
    return expression


def _validate_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def load_market_data_daily(
    freq_cfg: dict,
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
    nosuspend_days: int = 10,
) -> pl.DataFrame:
    """Read daily bars and normalize them to the common market schema."""
    frame = _normalize_symbol_column(pl.read_parquet(Path(freq_cfg["market_data"])))
    _validate_columns(
        frame,
        {
            "date",
            "symbol",
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
        },
        "Daily market parquet",
    )
    if frame["date"].dtype == pl.String:
        frame = frame.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    else:
        frame = frame.with_columns(pl.col("date").cast(pl.Date))
    start_date = date.fromisoformat(start)
    frame = frame.filter(pl.col("date") >= pl.lit(start_date))
    if end:
        frame = frame.filter(pl.col("date") <= pl.lit(date.fromisoformat(end)))
    can_buy = (pl.col("turnover") > 0) & ~pl.col("is_limit_up").fill_null(False).cast(pl.Boolean)
    can_sell = (pl.col("turnover") > 0) & ~pl.col("is_limit_down").fill_null(False).cast(pl.Boolean)
    can_open_base = _can_open_base_expr(universe, allow_st_open, nosuspend_days)
    return (
        frame
        .with_columns(
            pl.col("date").cast(pl.Datetime).alias("datetime"),
            can_trade_buy=can_buy,
            can_trade_sell=can_sell,
            can_open_base=can_open_base,
        )
        .with_columns(
            tradable=(pl.col("can_trade_buy") & pl.col("can_trade_sell")),
            can_open=(pl.col("can_trade_buy") & pl.col("can_trade_sell") & pl.col("can_open_base")),
        )
        .sort(["datetime", "symbol"])
    )


def load_daily_flags(
    daily_path: Path,
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
    nosuspend_days: int,
) -> pl.DataFrame:
    """Load daily eligibility/ranking constraints for broadcasting to intraday bars."""
    frame = _normalize_symbol_column(pl.read_parquet(daily_path))
    required = {"date", "symbol", "log_size", "industry", "index", "listed_Satisfied", "is_ST", "normal_days"}
    _validate_columns(frame, required, "Daily constraints parquet")
    if frame["date"].dtype == pl.String:
        frame = frame.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    start_date = date.fromisoformat(start)
    frame = frame.filter(pl.col("date") >= pl.lit(start_date))
    if end:
        frame = frame.filter(pl.col("date") <= pl.lit(date.fromisoformat(end)))
    base = _can_open_base_expr(universe, allow_st_open, nosuspend_days)
    rank_scope = _can_open_base_expr(universe, False, nosuspend_days)
    ranked = frame.with_columns(
        base.alias("can_open_base"),
        pl
        .when(rank_scope)
        .then(pl.col("log_size").rank(descending=True).over("date"))
        .otherwise(999999)
        .cast(pl.Int32)
        .alias("size_rank"),
    )
    keep = [
        column
        for column in [
            "date",
            "symbol",
            "log_size",
            "size_rank",
            "industry",
            "index",
            "listed_Satisfied",
            "is_ST",
            "normal_days",
            "can_open_base",
            "limit_up_price",
            "limit_down_price",
        ]
        if column in ranked.columns
    ]
    return ranked.select(keep).sort(["date", "symbol"])


def load_market_data_intraday(freq_cfg: dict, start: str, end: str | None) -> pl.DataFrame:
    """Read a 15-minute or 5-minute parquet into the common, flag-free schema."""
    path = Path(freq_cfg["market_data"])
    schema = pl.read_parquet_schema(path)
    symbol_column = "symbol" if "symbol" in schema else "stock_code" if "stock_code" in schema else None
    if symbol_column is None:
        raise ValueError("Intraday market parquet is missing 'symbol' (or legacy 'stock_code')")
    missing = {"datetime", "turnover", "vwap_ret"}.difference(schema)
    if missing:
        raise ValueError(f"Intraday market parquet is missing required columns: {sorted(missing)}")
    price_column = next((column for column in ("vwap15", "vwap5", "vwap", "close") if column in schema), None)
    if price_column is None:
        raise ValueError("Intraday market parquet needs one of vwap15, vwap5, vwap, or close for limit checks")
    frame = pl.scan_parquet(path).select([
        pl.col("datetime"),
        pl.col(symbol_column).alias("symbol"),
        pl.col("turnover"),
        pl.col("vwap_ret"),
        pl.col(price_column),
    ])
    if schema["datetime"] == pl.String:
        frame = frame.with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
    else:
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))
    frame = frame.filter(pl.col("datetime") >= pl.lit(datetime.fromisoformat(start)))
    if end:
        frame = frame.filter(pl.col("datetime") < pl.lit(datetime.fromisoformat(end) + timedelta(days=1)))
    return (
        frame
        .with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.col("vwap_ret").alias("ret"),
            pl.col(price_column).alias("execution_price"),
        )
        .collect()
        .sort(["datetime", "symbol"])
    )


def _dataset_from_encoded_frame(encoded: pl.DataFrame) -> BacktestDataset:
    missing = sorted(_BASE_COLUMNS.difference(encoded.columns))
    if missing:
        raise ValueError(f"Pool frame is missing canonical columns: {missing}")
    symbol_table = encoded.select(["symbol_id", "symbol"]).unique().sort("symbol_id")
    bar_counts = encoded.group_by("datetime", maintain_order=True).len().sort("datetime")
    counts = bar_counts["len"].to_numpy().astype(np.int64, copy=False)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0], offsets[1:] = 0, np.cumsum(counts)
    return BacktestDataset(
        pool_frame=encoded,
        bars=bar_counts["datetime"].to_numpy(),
        symbols=np.asarray(symbol_table["symbol"].to_list(), dtype=object),
        bar_offsets=offsets,
        row_symbol_ids=encoded["symbol_id"].to_numpy().astype(np.int32, copy=False),
        ret=encoded["ret"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        pred=encoded["pred"].fill_null(float("nan")).to_numpy().astype(np.float64, copy=False),
        size_rank=encoded["size_rank"].fill_null(999999).to_numpy().astype(np.int32, copy=False),
        tradable=encoded["tradable"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open=encoded["can_open"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open_base=encoded["can_open_base"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_buy=encoded["can_trade_buy"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_sell=encoded["can_trade_sell"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
    )


def _encode_dataset(pool: pl.DataFrame) -> BacktestDataset:
    symbols = pool["symbol"].unique().sort().to_list()
    symbol_map = pl.DataFrame({"symbol": symbols}).with_row_index("symbol_id")
    return _dataset_from_encoded_frame(pool.join(symbol_map, on="symbol", how="left").with_row_index("row_idx"))


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": path.resolve().as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _pool_cache_path(
    cache_dir: Path,
    freq_cfg: dict,
    daily_path: Path,
    pred_files: list[Path],
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
) -> Path:
    payload = {
        "version": POOL_CACHE_VERSION,
        "frequency": _frequency_name(freq_cfg),
        "market": _file_signature(Path(freq_cfg["market_data"])),
        "daily_flags": _file_signature(daily_path),
        "predictions": [_file_signature(path) for path in pred_files],
        "start": start,
        "end": end,
        "universe": list(universe) if universe is not None else None,
        "allow_st_open": allow_st_open,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir / f"pool_{_frequency_name(freq_cfg)}_{digest}.parquet"


def _read_cached_pool(path: Path) -> BacktestDataset | None:
    if not path.exists():
        return None
    try:
        return _dataset_from_encoded_frame(pl.read_parquet(path))
    except Exception as exc:
        logger.warning("Ignoring incompatible pool cache {}: {}", path.name, exc)
        return None


def build_pool(
    freq_cfg: dict,
    bm_path: Path,
    start: str,
    end: str | None,
    universe: list[str] | None,
    allow_st_open: bool,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    nosuspend_days: int = 10,
) -> tuple[BacktestDataset, pd.Series]:
    """Build a canonical pool for daily, 15-minute, or 5-minute backtests."""
    import config as cfg

    frequency = _frequency_name(freq_cfg)
    daily_path = Path(cfg.FREQ_CONFIG["daily"]["market_data"])
    pred_files = _prediction_files(Path(freq_cfg["preds_dir"]), list(freq_cfg["horizons"]))
    resolved_cache_dir = cache_dir or Path(freq_cfg["market_data"]).parent / ".cache"
    cache_path = _pool_cache_path(
        resolved_cache_dir, freq_cfg, daily_path, pred_files, start, end, universe, allow_st_open
    )
    bm_ret = load_external_benchmark(cfg.EXTERNAL_NAV_PATH) if cfg.USE_EXTERNAL_BENCHMARK else load_benchmark(bm_path)
    if use_cache:
        cached = _read_cached_pool(cache_path)
        if cached is not None:
            logger.info("Pool cache: hit – {}", cache_path.name)
            return cached, bm_ret
        logger.info("Pool cache: miss – {}", cache_path.name)

    predictions = load_predictions(freq_cfg, start, end)
    if frequency == "daily":
        market = load_market_data_daily(freq_cfg, start, end, universe, allow_st_open, nosuspend_days)
    else:
        market = load_market_data_intraday(freq_cfg, start, end)
        flags = load_daily_flags(daily_path, start, end, universe, allow_st_open, nosuspend_days)
        _validate_columns(
            flags, {"limit_up_price", "limit_down_price"}, "Daily constraints parquet for intraday limit checks"
        )
        market = (
            market
            .join(flags, on=["date", "symbol"], how="left")
            .with_columns(
                ((pl.col("execution_price") - pl.col("limit_up_price")).abs() <= 0.0005)
                .fill_null(False)
                .alias("is_limit_up"),
                ((pl.col("execution_price") - pl.col("limit_down_price")).abs() <= 0.0005)
                .fill_null(False)
                .alias("is_limit_down"),
            )
            .with_columns(
                ((pl.col("turnover") > 0) & ~pl.col("is_limit_up")).fill_null(False).alias("can_trade_buy"),
                ((pl.col("turnover") > 0) & ~pl.col("is_limit_down")).fill_null(False).alias("can_trade_sell"),
            )
            .with_columns(
                (pl.col("can_trade_buy") & pl.col("can_trade_sell")).alias("tradable"),
                (pl.col("can_trade_buy") & pl.col("can_trade_sell") & pl.col("can_open_base").fill_null(False)).alias(
                    "can_open"
                ),
            )
        )
    dataset = _encode_dataset(
        market.join(predictions, on=["datetime", "symbol"], how="left").sort(["datetime", "symbol"])
    )
    if use_cache:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.pool_frame.write_parquet(cache_path)
        logger.info("Pool cache: wrote – {}", cache_path.name)
    return dataset, bm_ret
