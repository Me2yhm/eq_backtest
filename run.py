"""
Backtest entry point.

Usage
-----
    python run.py runs/my-backtest
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger


def _configure_run_directory() -> None:
    """Require an isolated run directory before importing its configuration."""
    parser = argparse.ArgumentParser(description="Run an isolated EQ backtest")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Existing directory that contains this run's config.yml and receives its results.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    run_dir = run_dir.resolve()
    repo_dir = Path(__file__).resolve().parent
    if run_dir == repo_dir:
        parser.error("run_dir must be a dedicated directory outside the repository root")
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    if not (run_dir / "config.yml").is_file():
        parser.error(f"run directory is missing config.yml: {run_dir}")
    os.environ["EQ_BACKTEST_RUN_DIR"] = str(run_dir)


if __name__ == "__main__":
    _configure_run_directory()


import config as cfg
from data_loader import build_pool
from market_cache import load_benchmark_returns
from metrics import holding_period_stats, portfolio_metrics
from plotting import (
    plot_holding_periods,
    plot_metrics_table,
    plot_portfolio_results,
)
from portfolio import PortfolioResult, generate_portfolio


@dataclass(slots=True)
class PortfolioEvaluation:
    result: PortfolioResult
    portfolio_returns: pd.Series
    turnover: pd.Series
    close_counts: pd.DataFrame
    excess: pd.Series
    metrics: dict
    portfolio_pnl: pd.DataFrame

# ── Return computation ─────────────────────────────────────────────────────────


def compute_returns(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    is_short: bool,
    exclude_period: tuple | None,
) -> tuple:
    """
    Compute daily excess returns and one-way turnover for a single portfolio.

    Excess return = benchmark - portfolio (short) or portfolio - benchmark (long),
    Transaction costs are already deducted by the share-based simulator.

    Returns
    -------
    excess_ret : daily excess return Series
    turnover   : daily one-way turnover Series for reporting
    """
    bm_aligned = bm_ret.reindex(portfolio_returns.index)
    excess = (bm_aligned - portfolio_returns) if is_short else (portfolio_returns - bm_aligned)

    # Keep the merged-branch evaluation convention: the exclusion interval is
    # removed only from the metric series, after the full backtest timeline has
    # been simulated.
    if exclude_period:
        lo = pd.to_datetime(exclude_period[0])
        hi = pd.to_datetime(exclude_period[1])
        excess = excess[(excess.index < lo) | (excess.index >= hi)]

    return excess, turnover


def daily_returns_from_intraday(returns_intraday: pd.Series, agg_mode: str = "simple") -> pd.Series:
    """Aggregate intraday returns to daily returns.

    agg_mode: "simple" = sum(bar_ret), "compound" = (1+x).prod()-1
    """
    if returns_intraday.empty:
        return returns_intraday
    grouped = returns_intraday.groupby(returns_intraday.index.normalize())
    if agg_mode == "compound":
        return grouped.apply(lambda x: (1.0 + x).prod() - 1.0).rename(returns_intraday.name)
    else:
        return grouped.sum().rename(returns_intraday.name)


def daily_sum_from_intraday(series_intraday: pd.Series, name: str) -> pd.Series:
    """Aggregate additive intraday series (e.g. turnover, close counts) to daily sums."""
    if series_intraday.empty:
        return series_intraday.rename(name)
    return series_intraday.groupby(series_intraday.index.normalize()).sum().rename(name)


def _align_benchmark_to_index(bm_ret: pd.Series, index: pd.Index) -> pd.Series:
    """Align daily benchmark returns to target index (daily or intraday datetime index)."""
    bm_daily = bm_ret.copy()
    bm_daily.index = pd.to_datetime(bm_daily.index).normalize()
    if bm_daily.index.has_duplicates:
        bm_daily = bm_daily.groupby(level=0).sum()
    if isinstance(index, pd.DatetimeIndex):
        aligned_values = bm_daily.reindex(index.normalize()).to_numpy()
        return pd.Series(aligned_values, index=index, name=bm_ret.name)
    target_index = pd.to_datetime(index)
    aligned_values = bm_daily.reindex(target_index.normalize()).to_numpy()
    return pd.Series(aligned_values, index=index, name=bm_ret.name)


def _write_positions_csv(positions: pd.DataFrame, output_path: str) -> None:
    df = positions.reset_index()
    # 直接将 DataFrame 转换为 Polars，不改变 date 列的类型
    frame = pl.from_pandas(df)
    frame.write_csv(output_path)
    # frame = pl.from_pandas(positions.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    # frame.write_csv(output_path)


def _write_target_weights(target_weights: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
    frame = pl.from_pandas(target_weights.reset_index())
    frame.write_csv(csv_path)
    frame.write_parquet(parquet_path)


def _build_portfolio_pnl_frame(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    close_counts: pd.DataFrame,
    is_short: bool,
    exclude_period: tuple | None,
) -> pd.DataFrame:
    daily_benchmark = _align_benchmark_to_index(bm_ret, portfolio_returns.index).fillna(0.0).rename("daily_benchmark")
    daily_tto = turnover.rename("daily_tto").copy()
    daily_strategy = portfolio_returns.rename("daily_strategy")
    daily_alpha = (daily_strategy + daily_benchmark) if is_short else (daily_strategy - daily_benchmark)
    daily_alpha = daily_alpha.rename("daily_alpha")

    all_pl = daily_strategy.cumsum().rename("all_pl")
    alpha_pl = daily_alpha.cumsum().rename("alpha_pl")
    benchmark = daily_benchmark.cumsum().rename("benchmark")
    close_count = close_counts["n_closed"].reindex(portfolio_returns.index).fillna(0).astype(int).rename("close_count")

    if exclude_period:
        lo = pd.to_datetime(exclude_period[0])
        hi = pd.to_datetime(exclude_period[1])
        keep = (daily_strategy.index < lo) | (daily_strategy.index >= hi)
        all_pl = all_pl[keep]
        alpha_pl = alpha_pl[keep]
        benchmark = benchmark[keep]
        daily_strategy = daily_strategy[keep]
        daily_benchmark = daily_benchmark[keep]
        daily_alpha = daily_alpha[keep]
        daily_tto = daily_tto[keep]
        close_count = close_count[keep]

    frame = pd.concat(
        [
            all_pl,
            alpha_pl,
            benchmark,
            daily_strategy,
            daily_benchmark,
            daily_alpha,
            daily_tto,
            close_count,
        ],
        axis=1,
    )
    frame.index.name = "date"
    return frame


def evaluate_portfolio_result(
    result: PortfolioResult,
    bm_ret: pd.Series,
    frequency: str,
    agg_mode: str,
    is_short: bool,
    compounding: bool,
) -> PortfolioEvaluation:
    """Evaluate one simulator result using the project's excess-return metric."""
    is_intraday = frequency != "daily"
    if is_intraday:
        portfolio_returns = daily_returns_from_intraday(result.portfolio_returns, agg_mode=agg_mode)
        turnover = daily_sum_from_intraday(result.turnover, "turnover")
        close_counts = pd.DataFrame({
            "n_closed": daily_sum_from_intraday(result.close_counts["n_closed"], "n_closed").astype(int)
        })
    else:
        portfolio_returns = result.portfolio_returns
        turnover = result.turnover
        close_counts = result.close_counts

    excess, _ = compute_returns(
        portfolio_returns,
        result.cost_turnover,
        turnover,
        bm_ret,
        is_short=is_short,
        exclude_period=cfg.EXCLUDE_PERIOD,
    )
    portfolio_pnl = _build_portfolio_pnl_frame(
        portfolio_returns=portfolio_returns,
        cost_turnover=result.cost_turnover,
        turnover=turnover,
        bm_ret=bm_ret,
        close_counts=close_counts,
        is_short=is_short,
        exclude_period=cfg.EXCLUDE_PERIOD,
    )
    metrics = portfolio_metrics(excess, compounding=compounding)
    metrics["Ann. Turnover"] = float(turnover.mean() * 242) if not turnover.empty else 0.0
    return PortfolioEvaluation(
        result=result,
        portfolio_returns=portfolio_returns,
        turnover=turnover,
        close_counts=close_counts,
        excess=excess,
        metrics=metrics,
        portfolio_pnl=portfolio_pnl,
    )


def run_single_portfolio(
    pool,
    bm_ret: pd.Series,
    *,
    port_size: int,
    pool_size: int,
    thresh_out_buffer: int,
    frequency: str,
    agg_mode: str,
    is_short: bool,
    trade_on_next_bar: bool,
    strict_first_bar_top_n: bool,
    close_on_size_drop: bool,
    cost_per_turnover: float,
    compounding: bool,
    weight_mode: str = "equal",
    max_weight_multiple: float = 2.0,
    output_dir: str = "",
) -> PortfolioEvaluation:
    result = generate_portfolio(
        pool=pool,
        port_size=port_size,
        thresh_out_buffer=thresh_out_buffer,
        size_cut=pool_size,
        close_on_size_drop=close_on_size_drop,
        trade_on_next_bar=trade_on_next_bar,
        strict_first_bar_top_n=strict_first_bar_top_n,
        is_short=is_short,
        output_dir=output_dir,
        plot_heatmap=False,
        record_target_weights=False,
        cost_per_turnover=cost_per_turnover,
        portfolio_initial_value=cfg.PORTFOLIO_INITIAL_VALUE,
        weight_mode=weight_mode,
        max_weight_multiple=max_weight_multiple,
    )
    return evaluate_portfolio_result(result, bm_ret, frequency, agg_mode, is_short, compounding)


# ── Logging helper ─────────────────────────────────────────────────────────────


def _log_holding_stats(stats: dict, port_size: int) -> None:
    avg_open = stats["avg_open"]
    avg_closed = stats["avg_closed"]
    logger.info("Holding (open):   avg {:.1f} d, median {:.1f} d", avg_open.mean(), avg_open.median())
    if not avg_closed.empty:
        logger.info("Holding (closed): avg {:.1f} d", avg_closed.mean())


# ── Main ───────────────────────────────────────────────────────────────────────


def run() -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_dir = str(cfg.OUTPUT_DIR) + os.sep
    logger.info("Run directory: {} (config: {})", cfg.RUN_DIR, cfg.CONFIG_PATH)

    # ── Load data ─────────────────────────────────────────────────────────────
    bm_ret = load_benchmark_returns(
        cfg.MARKET_CACHE_DIR,
        cfg.BENCHMARK_SYMBOL,
        start=cfg.START,
        end=cfg.END,
    )
    pool = build_pool(
        freq_cfg=cfg.FREQ_CONFIG[cfg.FREQUENCY],
        start=cfg.START,
        end=cfg.END,
        universe=cfg.UNIVERSE,
        allow_st_open=cfg.ALLOW_ST_OPEN,
        use_cache=cfg.USE_POOL_CACHE,
        cache_dir=cfg.POOL_CACHE_DIR,
        nosuspend_days=cfg.NOSUSPEND_DAYS,
        prediction_merge_mode=cfg.PREDICTION_MERGE_MODE,
    )

    # ── Run for each portfolio size ───────────────────────────────────────────
    all_metrics = {}
    all_ret = {}
    all_cumrets = {}
    all_port_sizes = {}
    all_closes = {}

    for port_size in cfg.PORT_SIZES:
        logger.info("── Portfolio size: {} ──", port_size)

        result = generate_portfolio(
            pool=pool,
            port_size=port_size,
            thresh_out_buffer=cfg.THRESH_OUT_BUFFER,
            size_cut=cfg.POOL_SIZE,
            close_on_size_drop=cfg.CLOSE_ON_SIZE_DROP,
            trade_on_next_bar=cfg.trade_on_next_bar_for(cfg.FREQUENCY),
            strict_first_bar_top_n=cfg.STRICT_FIRST_BAR_TOP_N,
            is_short=cfg.IS_SHORT,
            output_dir=output_dir,
            record_target_weights=cfg.FREQUENCY != "daily",
            debug_mode=cfg.DEBUG,
            debug_symbol=cfg.DEBUG_SYMBOL,
            debug_datetime=cfg.DEBUG_DATETIME,
            cost_per_turnover=cfg.COST_PER_TURNOVER,
            portfolio_initial_value=cfg.PORTFOLIO_INITIAL_VALUE,
            weight_mode=cfg.WEIGHT_MODE,
            max_weight_multiple=cfg.MAX_WEIGHT_MULTIPLE,
        )

        positions = result.positions
        close_counts = result.close_counts
        _write_positions_csv(positions, f"{output_dir}positions_{port_size}.csv")
        if result.target_weights is not None:
            _write_target_weights(
                result.target_weights,
                f"{output_dir}target_weights_{port_size}.csv",
                f"{output_dir}target_weights_{port_size}.parquet",
            )
        evaluation = evaluate_portfolio_result(
            result=result,
            bm_ret=bm_ret,
            frequency=cfg.FREQUENCY,
            agg_mode=cfg.AGG_MODE,
            is_short=cfg.IS_SHORT,
            compounding=cfg.COMPOUNDING,
        )
        excess = evaluation.excess
        turnover = evaluation.turnover
        close_counts_eval = evaluation.close_counts
        portfolio_pnl = evaluation.portfolio_pnl
        portfolio_pnl.to_csv(
            f"{output_dir}portfolio_pnl_{port_size}.csv",
            index_label="date",
            float_format="%.8f",
        )

        m = evaluation.metrics
        all_metrics[port_size] = m
        all_ret[port_size] = excess.rename(f"Port_{port_size}")
        all_cumrets[port_size] = excess.cumsum().rename(f"Port_{port_size}")
        all_port_sizes[port_size] = result.held_counts.rename(f"Port_{port_size}")

        if not close_counts_eval.empty:
            close_counts_eval.columns = [f"Port_{port_size}"]
            all_closes[port_size] = close_counts_eval

        # Holding-period analysis
        hp = holding_period_stats(positions)
        plot_holding_periods(hp, port_size, is_short=cfg.IS_SHORT, output_path=output_dir)
        _log_holding_stats(hp, port_size)

    # ── Aggregate, save, and plot ─────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics).T
    cumrets_df = pd.DataFrame(all_cumrets)
    ret_df = pd.DataFrame(all_ret)
    port_sizes_df = pd.DataFrame(all_port_sizes)
    close_df = pd.concat(list(all_closes.values()), axis=1) if all_closes else pd.DataFrame()

    tag = f"{cfg.POOL_SIZE}_{cfg.BENCHMARK_SYMBOL}"
    metrics_df.to_csv(f"{output_dir}metrics_{tag}.csv")
    cumrets_df.to_csv(f"{output_dir}cumrets_{tag}.csv")
    ret_df.to_csv(f"{output_dir}returns_{tag}.csv")

    logger.info("── Metrics ──\n{}", metrics_df.to_string())

    plot_portfolio_results(
        cumrets_df,
        port_sizes_df,
        close_df,
        is_short=cfg.IS_SHORT,
        output_path=output_dir,
    )
    plot_metrics_table(
        metrics_df,
        cfg.POOL_SIZE,
        cfg.BENCHMARK_SYMBOL,
        is_short=cfg.IS_SHORT,
        output_path=output_dir,
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    run()
    end_time = time.time()
    logger.info("Total execution time: {:.2f} seconds", end_time - start_time)
