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

POOL_CACHE_VERSION = 7
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
    "vwap_ret",
    "execution_vwap",
    "prev_close",
    "bar_close",
    "row_idx",
    "symbol_id",
}


@dataclass(slots=True)
class BacktestDataset:
    """Array-backed input shared by daily and intraday simulation."""

    pool_frame: pl.DataFrame
    daily_snapshot_frame: pl.DataFrame
    bars: np.ndarray
    symbols: np.ndarray
    bar_offsets: np.ndarray
    bar_session_index: np.ndarray
    row_symbol_ids: np.ndarray
    vwap_ret: np.ndarray
    prev_close: np.ndarray
    bar_close: np.ndarray
    execution_vwap: np.ndarray
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


def _filter_exclude_period(
    frame: pl.DataFrame,
    datetime_column: str,
    exclude_period: tuple[str, str] | None,
) -> pl.DataFrame:
    """Remove an inclusive calendar-date interval from a frame."""
    if not exclude_period or frame.is_empty():
        return frame
    lo = date.fromisoformat(str(exclude_period[0])[:10])
    hi = date.fromisoformat(str(exclude_period[1])[:10])
    if lo > hi:
        raise ValueError(f"exclude_period start {lo} is after end {hi}")
    frame_date = pl.col(datetime_column).dt.date()
    return frame.filter(~frame_date.is_between(pl.lit(lo), pl.lit(hi), closed="both"))


def _filter_benchmark_period(
    series: pd.Series,
    exclude_period: tuple[str, str] | None,
) -> pd.Series:
    if not exclude_period or series.empty:
        return series
    lo = pd.Timestamp(str(exclude_period[0])[:10]).date()
    hi = pd.Timestamp(str(exclude_period[1])[:10]).date()
    dates = pd.to_datetime(series.index).date
    return series.loc[(dates < lo) | (dates > hi)]


def _recompute_intraday_vwap_ret(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute next retained-bar VWAP return after any date filtering.

    A symbol is only bridged when its next retained row is also the next
    retained global bar.  Filtering an exclusion interval therefore creates
    the intended direct pre-gap/post-gap transition, while ordinary missing
    rows (for example a suspension) do not create a synthetic return.
    """
    if frame.is_empty():
        return frame
    bars = frame.select("datetime").unique().sort("datetime").with_columns(
        pl.col("datetime").shift(-1).alias("__expected_next_datetime")
    )
    out = (
        frame.sort(["datetime", "symbol"])
        .with_columns([
            pl.col("execution_vwap").shift(-1).over("symbol").alias("__next_vwap"),
            pl.col("datetime").shift(-1).over("symbol").alias("__next_symbol_datetime"),
        ])
        .join(bars, on="datetime", how="left")
        .with_columns(
            pl.when(
                (pl.col("__next_symbol_datetime") == pl.col("__expected_next_datetime"))
                & (pl.col("execution_vwap") > 0)
                & pl.col("__next_vwap").is_not_null()
                & (pl.col("__next_vwap") > 0)
            )
            .then(pl.col("__next_vwap") / pl.col("execution_vwap") - 1.0)
            .otherwise(None)
            .alias("vwap_ret")
        )
        .drop(["__next_vwap", "__next_symbol_datetime", "__expected_next_datetime"])
    )
    return out


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


def _validate_prediction_frames(frames: list[pl.DataFrame], files: list[Path], merge_mode: str) -> None:
    if merge_mode not in {"concat_disjoint", "mean"}:
        raise ValueError("prediction_merge_mode must be 'concat_disjoint' or 'mean'")
    ranges: list[tuple[date, date, str]] = []
    for frame, path in zip(frames, files, strict=True):
        if frame.is_empty():
            continue
        duplicate = (
            frame.group_by(["datetime", "symbol"])
            .len()
            .filter(pl.col("len") > 1)
            .head(1)
        )
        if not duplicate.is_empty():
            raise ValueError(f"Prediction file contains duplicate (datetime, symbol) keys: {path}")
        min_date, max_date = frame.select([
            pl.col("datetime").dt.date().min().alias("min_date"),
            pl.col("datetime").dt.date().max().alias("max_date"),
        ]).row(0)
        ranges.append((min_date, max_date, path.name))
    if merge_mode == "concat_disjoint":
        for index, (lo_a, hi_a, name_a) in enumerate(ranges):
            for lo_b, hi_b, name_b in ranges[index + 1 :]:
                if lo_a <= hi_b and lo_b <= hi_a:
                    raise ValueError(
                        "Prediction date ranges overlap; use prediction_merge_mode='mean' explicitly: "
                        f"{name_a} [{lo_a}, {hi_a}] vs {name_b} [{lo_b}, {hi_b}]"
                    )
        logger.info("Prediction ranges: {}", ranges)


def load_predictions(
    freq_cfg: dict,
    start: str,
    end: str | None = None,
    exclude_period: tuple[str, str] | None = None,
    merge_mode: str = "concat_disjoint",
) -> pl.DataFrame:
    """Load every frequency into the canonical ``[datetime, symbol, pred]`` form."""
    frequency = _frequency_name(freq_cfg)
    files = _prediction_files(Path(freq_cfg["preds_dir"]), list(freq_cfg["horizons"]))
    frames = [
        _filter_exclude_period(_load_prediction_file(path, frequency, start, end), "datetime", exclude_period)
        for path in files
    ]
    _validate_prediction_frames(frames, files, merge_mode)
    logger.info("{} predictions: {} file(s) loaded – {}", frequency, len(files), [path.name for path in files])
    combined = pl.concat(frames, how="vertical_relaxed")
    if merge_mode == "mean":
        combined = (
            combined.group_by(["datetime", "symbol"], maintain_order=True)
            .agg(pl.col("pred").mean().alias("pred"))
        )
    return combined.sort(["datetime", "symbol"])


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
    exclude_period: tuple[str, str] | None = None,
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
    frame = _filter_exclude_period(frame, "date", exclude_period)
    can_buy = (pl.col("turnover") > 0) & ~pl.col("is_limit_up").fill_null(False).cast(pl.Boolean)
    can_sell = (pl.col("turnover") > 0) & ~pl.col("is_limit_down").fill_null(False).cast(pl.Boolean)
    can_open_base = _can_open_base_expr(universe, allow_st_open, nosuspend_days)
    return (
        frame
        .with_columns(
            pl.col("date").cast(pl.Datetime).alias("datetime"),
            pl.col("ret").alias("vwap_ret"),
            # Daily bars have no execution-price decomposition.  A unit VWAP
            # and close=1+ret preserves the daily weight-return P&L in the
            # common share-based simulator.
            pl.lit(1.0).alias("execution_vwap"),
            pl.lit(1.0).alias("prev_close"),
            (pl.lit(1.0) + pl.col("ret")).alias("bar_close"),
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
    base_frame = frame.with_columns(base.alias("can_open_base"))
    # Rank only the eligible universe.  Applying rank before masking ineligible
    # rows changes every remaining rank and breaks the reference portfolio.
    eligible_ranked = (
        base_frame.filter(rank_scope)
        .select(["date", "symbol", "log_size"])
        .with_columns(pl.col("log_size").rank(descending=True).over("date").cast(pl.Int32).alias("size_rank"))
        .select(["date", "symbol", "size_rank"])
    )
    ranked = (
        base_frame.drop("size_rank")
        .join(eligible_ranked, on=["date", "symbol"], how="left")
        .with_columns(pl.col("size_rank").fill_null(999999).cast(pl.Int32))
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


def load_market_data_intraday(
    freq_cfg: dict,
    start: str,
    end: str | None,
    exclude_period: tuple[str, str] | None = None,
) -> pl.DataFrame:
    """Read any intraday parquet into the common, flag-free schema."""
    path = Path(freq_cfg["market_data"])
    schema = pl.read_parquet_schema(path)
    symbol_column = "symbol" if "symbol" in schema else "stock_code" if "stock_code" in schema else None
    if symbol_column is None:
        raise ValueError("Intraday market parquet is missing 'symbol' (or legacy 'stock_code')")
    missing = {"datetime", "turnover", "vwap_ret"}.difference(schema)
    if missing:
        raise ValueError(f"Intraday market parquet is missing required columns: {sorted(missing)}")
    source_columns = freq_cfg.get("market_columns", {})
    price_column = source_columns.get("execution_vwap") or next(
        (column for column in ("vwap15", "vwap5", "vwap", "close") if column in schema), None
    )
    close_column = source_columns.get("bar_close", "close")
    if price_column is None:
        raise ValueError("Intraday market parquet needs an execution VWAP source column for limit checks")
    if close_column not in schema:
        raise ValueError(f"Intraday market parquet is missing configured bar close column '{close_column}'")
    frame = pl.scan_parquet(path).select([
        pl.col("datetime"),
        pl.col(symbol_column).alias("symbol"),
        pl.col("turnover"),
        pl.col("vwap_ret"),
        pl.col(close_column).alias("bar_close"),
        pl.col(price_column),
    ])
    if schema["datetime"] == pl.String:
        frame = frame.with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
    else:
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))
    frame = frame.filter(pl.col("datetime") >= pl.lit(datetime.fromisoformat(start)))
    if end:
        frame = frame.filter(pl.col("datetime") < pl.lit(datetime.fromisoformat(end) + timedelta(days=1)))
    frame = (
        frame
        .with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.col("vwap_ret").alias("ret"),
            pl.col(price_column).alias("execution_vwap"),
            pl.col(price_column).alias("execution_price"),
        )
        .collect()
        .sort(["datetime", "symbol"])
    )
    frame = _filter_exclude_period(frame, "datetime", exclude_period)
    return _recompute_intraday_vwap_ret(frame)


def _dataset_from_encoded_frame(encoded: pl.DataFrame) -> BacktestDataset:
    missing = sorted(_BASE_COLUMNS.difference(encoded.columns))
    if missing:
        raise ValueError(f"Pool frame is missing canonical columns: {missing}")
    symbol_table = encoded.select(["symbol_id", "symbol"]).unique().sort("symbol_id")
    bar_counts = encoded.group_by("datetime", maintain_order=True).len().sort("datetime")
    counts = bar_counts["len"].to_numpy().astype(np.int64, copy=False)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0], offsets[1:] = 0, np.cumsum(counts)
    bars = bar_counts["datetime"].to_numpy()
    bar_datetimes = pd.to_datetime(bars)
    if (bar_datetimes == bar_datetimes.normalize()).all():
        bar_session_index = np.arange(len(bars), dtype=np.int32)
    else:
        bar_session_half = (bar_datetimes.hour >= 12).astype(np.int32)
        bar_date_session = bar_datetimes.normalize().astype("int64") // 10**9 * 10 + bar_session_half
        bar_session_index = pd.factorize(bar_date_session)[0].astype(np.int32)
    snapshot_columns = [
        "date", "symbol", "log_size", "size_rank", "industry", "index", "listed_Satisfied", "is_ST", "normal_days",
    ]
    return BacktestDataset(
        pool_frame=encoded,
        daily_snapshot_frame=encoded.select(snapshot_columns).unique().sort(["date", "symbol"]),
        bars=bars,
        symbols=np.asarray(symbol_table["symbol"].to_list(), dtype=object),
        bar_offsets=offsets,
        bar_session_index=bar_session_index,
        row_symbol_ids=encoded["symbol_id"].to_numpy().astype(np.int32, copy=False),
        vwap_ret=encoded["vwap_ret"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        prev_close=encoded["prev_close"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        bar_close=encoded["bar_close"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        execution_vwap=encoded["execution_vwap"].fill_null(0.0).to_numpy().astype(np.float64, copy=False),
        pred=encoded["pred"].fill_null(float("nan")).to_numpy().astype(np.float64, copy=False),
        size_rank=encoded["size_rank"].fill_null(999999).to_numpy().astype(np.int32, copy=False),
        tradable=encoded["tradable"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open=encoded["can_open"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_open_base=encoded["can_open_base"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_buy=encoded["can_trade_buy"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
        can_trade_sell=encoded["can_trade_sell"].fill_null(False).to_numpy().astype(np.bool_, copy=False),
    )


def _encode_dataset(pool: pl.DataFrame, derive_prev_close: bool = False) -> BacktestDataset:
    symbols = pool["symbol"].unique().sort().to_list()
    symbol_map = pl.DataFrame({"symbol": symbols}).with_row_index("symbol_id")
    encoded = pool.join(symbol_map, on="symbol", how="left").with_row_index("row_idx")
    # The previous close is valid only if the symbol exists in the globally
    # previous bar; a symbol-level shift alone bridges suspension gaps.
    if derive_prev_close:
        encoded = encoded.sort(["symbol", "datetime"]).with_columns([
            pl.col("bar_close").shift(1).over("symbol").alias("prev_close_raw"),
            pl.col("datetime").shift(1).over("symbol").alias("prev_symbol_dt"),
        ])
        all_datetimes = encoded["datetime"].unique().sort()
        previous_datetime = {all_datetimes[index]: all_datetimes[index - 1] for index in range(1, len(all_datetimes))}
        encoded = encoded.with_columns(
            pl.col("datetime").replace_strict(previous_datetime, default=None).alias("expected_prev_dt")
        ).with_columns(
            pl.when(pl.col("expected_prev_dt").is_null() | (pl.col("prev_symbol_dt") != pl.col("expected_prev_dt")))
            .then(0.0)
            .otherwise(pl.col("prev_close_raw"))
            .alias("prev_close")
        ).sort(["datetime", "symbol"])
    return _dataset_from_encoded_frame(encoded)


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
    exclude_period: tuple[str, str] | None,
    prediction_merge_mode: str,
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
        "exclude_period": list(exclude_period) if exclude_period else None,
        "prediction_merge_mode": prediction_merge_mode,
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
    exclude_period: tuple[str, str] | None = None,
    prediction_merge_mode: str = "concat_disjoint",
) -> tuple[BacktestDataset, pd.Series]:
    """Build a canonical pool for daily, intraday, or 5-minute backtests."""
    import config as cfg

    frequency = _frequency_name(freq_cfg)
    daily_path = Path(cfg.FREQ_CONFIG["daily"]["market_data"])
    pred_files = _prediction_files(Path(freq_cfg["preds_dir"]), list(freq_cfg["horizons"]))
    resolved_cache_dir = cache_dir or Path(freq_cfg["market_data"]).parent / ".cache"
    cache_path = _pool_cache_path(
        resolved_cache_dir,
        freq_cfg,
        daily_path,
        pred_files,
        start,
        end,
        universe,
        allow_st_open,
        exclude_period,
        prediction_merge_mode,
    )
    bm_ret = load_external_benchmark(cfg.EXTERNAL_NAV_PATH) if cfg.USE_EXTERNAL_BENCHMARK else load_benchmark(bm_path)
    bm_ret = _filter_benchmark_period(bm_ret, exclude_period)
    if use_cache:
        cached = _read_cached_pool(cache_path)
        if cached is not None:
            logger.info("Pool cache: hit – {}", cache_path.name)
            return cached, bm_ret
        logger.info("Pool cache: miss – {}", cache_path.name)

    predictions = load_predictions(
        freq_cfg,
        start,
        end,
        exclude_period=exclude_period,
        merge_mode=prediction_merge_mode,
    )
    if frequency == "daily":
        market = load_market_data_daily(
            freq_cfg, start, end, universe, allow_st_open, nosuspend_days, exclude_period
        )
    else:
        market = load_market_data_intraday(freq_cfg, start, end, exclude_period)
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
        market.join(predictions, on=["datetime", "symbol"], how="left").sort(["datetime", "symbol"]),
        derive_prev_close=frequency != "daily",
    )
    if use_cache:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.pool_frame.write_parquet(cache_path)
        logger.info("Pool cache: wrote – {}", cache_path.name)
    return dataset, bm_ret
