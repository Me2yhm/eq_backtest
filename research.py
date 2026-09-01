"""Signal diagnostics and parameter sweeps for the backtest framework."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

import config as cfg
from data_loader import BacktestDataset, _encode_dataset, build_pool
from market_cache import load_benchmark_returns
from run import run_single_portfolio


def _signal_frame(pool: BacktestDataset, pool_size: int, is_short: bool = False) -> pl.DataFrame:
    """Return the minimum frame needed for same-timestamp signal validation."""
    columns = [
        "datetime",
        "symbol",
        "pred",
        "vwap_ret",
        "size_rank",
        "can_open_base",
        "can_open",
    ]
    if is_short:
        columns.append("borrow_available")
    frame = pool.pool_frame.select(columns)
    if frame.schema["datetime"] == pl.String:
        frame = frame.with_columns(pl.col("datetime").str.to_datetime(strict=False))
    else:
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))
    frame = frame.filter(
        pl.col("pred").is_not_null()
        & pl.col("pred").is_finite()
        & pl.col("vwap_ret").is_not_null()
        & pl.col("vwap_ret").is_finite()
    )
    if is_short:
        frame = frame.with_columns([
            (pl.col("can_open") & pl.col("borrow_available").fill_null(False)).alias("can_open"),
            (pl.lit(1) - pl.col("pred")).alias("__rank_signal"),
        ])
    else:
        frame = frame.with_columns(pl.col("pred").alias("__rank_signal"))
    return frame.with_columns([
        pl.col("__rank_signal").rank("ordinal", descending=True).over("datetime").alias("pred_rank"),
        pl.len().over("datetime").alias("cs_size"),
    ]).with_columns([
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month"),
    ])


def _validate_layer(frame: pl.DataFrame, layer: str, pool_size: int) -> pl.DataFrame:
    if layer == "all":
        return frame
    if layer == "pool":
        return frame.filter((pl.col("size_rank") <= pool_size) & pl.col("can_open_base"))
    if layer == "executable":
        return frame.filter(pl.col("can_open"))
    raise ValueError(f"Unknown validation layer: {layer}")


def signal_validation(
    pool: BacktestDataset,
    *,
    pool_size: int,
    top_n: int = 800,
    bottom_n: int = 800,
    is_short: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute per-bar and grouped signal effectiveness summaries.

    ``prediction`` and ``vwap_ret`` are joined by the same retained timestamp;
    no additional shift is applied.
    """
    base = _signal_frame(pool, pool_size, is_short=is_short)
    per_layer: list[pl.DataFrame] = []
    for layer in ("all", "pool", "executable"):
        frame = _validate_layer(base, layer, pool_size)
        if frame.is_empty():
            continue
        # Ranking is recomputed after layer filtering so the top/bottom groups
        # describe the actual validation universe for that layer.
        frame = frame.with_columns([
            pl.col("__rank_signal").rank("ordinal", descending=True).over("datetime").alias("layer_rank"),
            pl.len().over("datetime").alias("layer_size"),
            pl.col("__rank_signal").rank("average").over("datetime").alias("__pred_rank_avg"),
            pl.col("vwap_ret").rank("average").over("datetime").alias("__ret_rank_avg"),
        ])
        bars = (
            frame
            .group_by("datetime", maintain_order=True)
            .agg([
                pl.len().alias("n_obs"),
                pl.corr("__rank_signal", "vwap_ret").alias("ic"),
                pl.corr("__pred_rank_avg", "__ret_rank_avg").alias("rank_ic"),
                pl.col("vwap_ret").filter(pl.col("layer_rank") <= top_n).mean().alias("top_mean_ret"),
                pl
                .col("vwap_ret")
                .filter(pl.col("layer_rank") > pl.col("layer_size") - bottom_n)
                .mean()
                .alias("bottom_mean_ret"),
            ])
            .with_columns([
                (pl.col("top_mean_ret") - pl.col("bottom_mean_ret")).alias("top_bottom_spread"),
                pl.lit(layer).alias("layer"),
            ])
            .with_columns([
                pl.col("datetime").dt.year().alias("year"),
                pl.col("datetime").dt.month().alias("month"),
            ])
        )
        per_layer.append(bars)

    if not per_layer:
        empty = pl.DataFrame(
            schema={
                "datetime": pl.Datetime,
                "layer": pl.String,
                "n_obs": pl.Int64,
                "ic": pl.Float64,
                "rank_ic": pl.Float64,
                "top_mean_ret": pl.Float64,
                "bottom_mean_ret": pl.Float64,
                "top_bottom_spread": pl.Float64,
                "year": pl.Int32,
                "month": pl.Int8,
            }
        )
        return empty, empty

    per_bar = pl.concat(per_layer, how="vertical_relaxed").sort(["layer", "datetime"])
    summary = (
        per_bar
        .group_by(["layer", "year", "month"], maintain_order=True)
        .agg([
            pl.len().alias("n_bars"),
            pl.col("n_obs").mean().alias("avg_n_obs"),
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("rank_ic").mean().alias("mean_rank_ic"),
            pl.col("top_mean_ret").mean().alias("mean_top_ret"),
            pl.col("bottom_mean_ret").mean().alias("mean_bottom_ret"),
            pl.col("top_bottom_spread").mean().alias("mean_top_bottom_spread"),
        ])
        .sort(["layer", "year", "month"])
    )
    return per_bar, summary


def _slice_dataset(pool: BacktestDataset, start: str, end: str) -> BacktestDataset:
    frame = pool.pool_frame.filter(
        (pl.col("datetime") >= pl.lit(pd.Timestamp(start))) & (pl.col("datetime") < pl.lit(pd.Timestamp(end)))
    ).drop([column for column in ("row_idx", "symbol_id") if column in pool.pool_frame.columns])
    if frame.is_empty():
        return _encode_dataset(frame, derive_prev_close=True)
    return _encode_dataset(frame, derive_prev_close=True)


def _research_is_short() -> bool:
    modes = cfg.STRATEGY_MODES
    if len(modes) != 1 or modes[0] not in {"long_only", "short_only"}:
        raise ValueError("research.py supports exactly one sleeve: long_only or short_only")
    return modes[0] == "short_only"


def parameter_sweep(
    pool: BacktestDataset,
    bm_ret: pd.Series,
    *,
    splits: dict[str, tuple[str, str]],
    pool_sizes: tuple[int, ...] = (2200, 3300, 4400, 5500),
    port_sizes: tuple[int, ...] = tuple(i for i in range(10, 100, 10)),
    buffers: tuple[int, ...] = (0, 100, 300, 600, 1000, 1200, 1400),
    costs: tuple[float, ...] = (0.00045,),
    weight_modes: tuple[str, ...] = ("equal",),
    is_short: bool = False,
    sbl_enabled: bool = False,
) -> pd.DataFrame:
    """Run the configured grid independently on each time split."""
    if is_short and not sbl_enabled:
        raise ValueError("Short sweeps require a pool built with explicit short_borrow_sources")
    rows: list[dict] = []
    split_datasets = {name: _slice_dataset(pool, *period) for name, period in splits.items()}
    for split_name, split_pool in split_datasets.items():
        if split_pool.pool_frame.is_empty():
            logger.warning("Skipping empty sweep split {}", split_name)
            continue
        split_bm = bm_ret.loc[
            (pd.to_datetime(bm_ret.index) >= pd.Timestamp(splits[split_name][0]))
            & (pd.to_datetime(bm_ret.index) < pd.Timestamp(splits[split_name][1]))
        ]
        for pool_size, port_size, buffer, cost, weight_mode in product(
            pool_sizes, port_sizes, buffers, costs, weight_modes
        ):
            if port_size >= pool_size:
                continue
            evaluation = run_single_portfolio(
                split_pool,
                split_bm,
                port_size=port_size,
                pool_size=pool_size,
                thresh_out_buffer=buffer,
                frequency=cfg.FREQUENCY,
                agg_mode=cfg.AGG_MODE,
                is_short=is_short,
                trade_on_next_bar=cfg.trade_on_next_bar_for(cfg.FREQUENCY),
                strict_first_bar_top_n=cfg.STRICT_FIRST_BAR_TOP_N,
                close_on_size_drop=cfg.CLOSE_ON_SIZE_DROP,
                cost_per_turnover=cost,
                compounding=cfg.COMPOUNDING,
                weight_mode=weight_mode,
                max_weight_multiple=cfg.MAX_WEIGHT_MULTIPLE,
                sbl_enabled=sbl_enabled,
            )
            row = {
                "split": split_name,
                "pool_size": pool_size,
                "port_size": port_size,
                "exit_buffer": buffer,
                "cost_per_turnover": cost,
                "weight_mode": weight_mode,
                **evaluation.metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def select_robust_candidates(
    sweep: pd.DataFrame,
    *,
    tune_split: str = "tune_2024",
    test_split: str = "test_2025",
    baseline: dict | None = None,
) -> pd.DataFrame:
    """Apply the agreed OOS and robustness gate to sweep results."""
    if sweep.empty:
        return sweep.assign(robust=False)
    baseline = baseline or {
        "pool_size": 4400,
        "port_size": 800,
        "exit_buffer": 600,
        "weight_mode": "equal",
    }
    keys = ["pool_size", "port_size", "exit_buffer", "weight_mode", "cost_per_turnover"]
    base_mask = pd.Series(True, index=sweep.index)
    for key, value in baseline.items():
        base_mask &= sweep[key].eq(value)

    base_tune = sweep[base_mask & sweep["split"].eq(tune_split)].set_index("cost_per_turnover")
    base_test = sweep[base_mask & sweep["split"].eq(test_split)].set_index("cost_per_turnover")
    output = sweep.copy()
    output["robust"] = False
    output["reason"] = "not evaluated"
    for index, candidate in output[output["split"].eq(test_split)].iterrows():
        cost = candidate["cost_per_turnover"]
        if cost not in base_test.index or cost not in base_tune.index:
            output.loc[index, "reason"] = "baseline missing"
            continue
        test_base = base_test.loc[cost]
        tune_base = base_tune.loc[cost]
        candidate_mask = pd.Series(True, index=sweep.index)
        for key in keys:
            candidate_mask &= sweep[key].eq(candidate[key])
        candidate_tune_rows = sweep[candidate_mask & sweep["split"].eq(tune_split)]
        if candidate_tune_rows.empty:
            output.loc[index, "reason"] = "candidate tune result missing"
            continue
        candidate_tune = candidate_tune_rows.iloc[0]
        test_pass = (
            candidate["Ann. Return"] > test_base["Ann. Return"]
            and abs(candidate["Max Drawdown"]) <= abs(test_base["Max Drawdown"]) * 1.10
            and candidate["Ann. Turnover"] <= test_base["Ann. Turnover"] * 1.20
        )
        tune_pass = candidate_tune["Ann. Return"] >= tune_base["Ann. Return"]
        output.loc[index, "robust"] = bool(test_pass and tune_pass)
        output.loc[index, "reason"] = "passed" if test_pass and tune_pass else "gate failed"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal diagnostics and parameter sweep")
    parser.add_argument("--output-dir", type=Path, default=Path("research_output"))
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--include-weight-modes", action="store_true")
    parser.add_argument(
        "--no-sort-sweep-by-return",
        dest="sort_sweep_by_return",
        action="store_false",
        help="参数寻优输出不按收益率降序排序（默认按收益率降序排序）",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    is_short = _research_is_short()
    if is_short and not cfg.SHORT_BORROW_SOURCES:
        raise ValueError("short_only research requires short_borrow_sources")


    pool = build_pool(
        freq_cfg=cfg.FREQ_CONFIG[cfg.FREQUENCY],
        start=cfg.START,
        end=cfg.END,
        universe=cfg.UNIVERSE,
        allow_st_open=cfg.ALLOW_ST_OPEN,
        use_cache=cfg.USE_POOL_CACHE,
        cache_dir=cfg.POOL_CACHE_DIR,
        nosuspend_days=cfg.NOSUSPEND_DAYS,
        short_borrow_sources=cfg.SHORT_BORROW_SOURCES if is_short else None,
        borrow_selection=cfg.BORROW_SELECTION,
        prediction_merge_mode=cfg.PREDICTION_MERGE_MODE,
    )
    bm_ret = load_benchmark_returns(
        cfg.MARKET_CACHE_DIR,
        cfg.BENCHMARK_SYMBOL,
        start=cfg.START,
        end=cfg.END,
    )
    per_bar, summary = signal_validation(pool, pool_size=cfg.POOL_SIZE, top_n=cfg.PORT_SIZES[0], is_short=is_short)
    per_bar.write_csv(args.output_dir / "signal_validation_per_bar.csv")
    summary.write_csv(args.output_dir / "signal_validation_summary.csv")

    if not args.validation_only:
        weight_modes = ("equal", "rank_linear", "rank_square") if args.include_weight_modes else ("equal",)
        sweep = parameter_sweep(
            pool,
            bm_ret,
            splits={"tune_2024": ("2024-01-01", "2025-01-01"), "test_2025": ("2025-01-01", "2026-01-01")},
            weight_modes=weight_modes,
            is_short=is_short,
            sbl_enabled=is_short,
        )
        if args.sort_sweep_by_return and not sweep.empty and "Ann. Return" in sweep.columns:
            sweep = sweep.sort_values("Ann. Return", ascending=False)
        sweep.to_csv(args.output_dir / "parameter_sweep.csv", index=False)
        candidates = select_robust_candidates(sweep)
        if args.sort_sweep_by_return and not candidates.empty and "Ann. Return" in candidates.columns:
            candidates = candidates.sort_values("Ann. Return", ascending=False)
        candidates.to_csv(args.output_dir / "parameter_sweep_candidates.csv", index=False)


if __name__ == "__main__":
    main()
