"""
Backtest entry point.

Usage
-----
    python run.py
"""

from __future__ import annotations

import os

import pandas as pd
import polars as pl
from loguru import logger

import config as cfg
from data_loader import build_pool
from metrics import holding_period_stats, portfolio_metrics
from plotting import (
    plot_holding_periods,
    plot_metrics_table,
    plot_portfolio_results,
)
from portfolio import generate_portfolio

# ── Return computation ─────────────────────────────────────────────────────────


def compute_returns(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    is_short: bool,
    cost: float,
    exclude_period: tuple | None,
) -> tuple:
    """
    Compute daily excess returns and one-way turnover for a single portfolio.

    Excess return = benchmark - portfolio (short) or portfolio - benchmark (long),
    minus transaction cost.

    Returns
    -------
    excess_ret : daily excess return Series
    turnover   : daily one-way turnover Series for reporting
    """
    bm_aligned = bm_ret.reindex(portfolio_returns.index)
    excess = (bm_aligned - portfolio_returns) if is_short else (portfolio_returns - bm_aligned)
    excess -= cost_turnover.mul(cost)

    if exclude_period:
        lo = pd.to_datetime(exclude_period[0])
        hi = pd.to_datetime(exclude_period[1])
        excess.loc[(excess.index >= lo) & (excess.index < hi)] = 0.0

    return excess, turnover


def _write_positions_csv(positions: pd.DataFrame, output_path: str) -> None:
    frame = pl.from_pandas(positions.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    frame.write_csv(output_path)


def _build_portfolio_pnl_frame(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    close_counts: pd.DataFrame,
    is_short: bool,
    cost: float,
    exclude_period: tuple | None,
) -> pd.DataFrame:
    daily_benchmark = bm_ret.reindex(portfolio_returns.index).fillna(0.0).rename("daily_benchmark")
    daily_tto = turnover.rename("daily_tto").copy()
    daily_strategy = portfolio_returns.sub(cost_turnover.mul(cost)).rename("daily_strategy")
    daily_alpha = (daily_strategy + daily_benchmark) if is_short else (daily_strategy - daily_benchmark)
    daily_alpha = daily_alpha.rename("daily_alpha")

    if exclude_period:
        lo = pd.to_datetime(exclude_period[0])
        hi = pd.to_datetime(exclude_period[1])
        mask = (daily_strategy.index >= lo) & (daily_strategy.index < hi)
        daily_strategy.loc[mask] = 0.0
        daily_alpha.loc[mask] = 0.0
        daily_tto.loc[mask] = 0.0

    all_pl = daily_strategy.cumsum().rename("all_pl")
    alpha_pl = daily_alpha.cumsum().rename("alpha_pl")
    benchmark = daily_benchmark.cumsum().rename("benchmark")
    close_count = close_counts["n_closed"].reindex(portfolio_returns.index).fillna(0).astype(int).rename("close_count")

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

    # ── Load data ─────────────────────────────────────────────────────────────
    pool, bm_ret = build_pool(
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
            close_on_size_drop=True,
            trade_on_next_day=cfg.TRADE_ON_NEXT_DAY,
            strict_first_day_top_n=cfg.STRICT_FIRST_DAY_TOP_N,
            is_short=cfg.IS_SHORT,
            output_dir=output_dir,
        )

        positions = result.positions
        close_counts = result.close_counts
        _write_positions_csv(positions, f"{output_dir}positions_{port_size}.csv")

        excess, turnover = compute_returns(
            result.portfolio_returns,
            result.cost_turnover,
            result.turnover,
            bm_ret,
            is_short=cfg.IS_SHORT,
            cost=cfg.COST_PER_TURNOVER,
            exclude_period=cfg.EXCLUDE_PERIOD,
        )
        portfolio_pnl = _build_portfolio_pnl_frame(
            portfolio_returns=result.portfolio_returns,
            cost_turnover=result.cost_turnover,
            turnover=result.turnover,
            bm_ret=bm_ret,
            close_counts=close_counts,
            is_short=cfg.IS_SHORT,
            cost=cfg.COST_PER_TURNOVER,
            exclude_period=cfg.EXCLUDE_PERIOD,
        )
        portfolio_pnl.to_csv(
            f"{output_dir}portfolio_pnl_{port_size}.csv",
            index_label="date",
            float_format="%.8f",
        )

        m = portfolio_metrics(excess)
        m["Ann. Turnover"] = float(turnover.mean() * 242)
        all_metrics[port_size] = m
        all_ret[port_size] = excess.rename(f"Port_{port_size}")
        all_cumrets[port_size] = excess.cumsum().rename(f"Port_{port_size}")
        all_port_sizes[port_size] = result.held_counts.rename(f"Port_{port_size}")

        if not close_counts.empty:
            close_counts.columns = [f"Port_{port_size}"]
            all_closes[port_size] = close_counts

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

    tag = f"{cfg.POOL_SIZE}_{cfg.BM_NAME}"
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
        cfg.BM_NAME,
        is_short=cfg.IS_SHORT,
        output_path=output_dir,
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    run()
    end_time = time.time()
    logger.info("Total execution time: {:.2f} seconds", end_time - start_time)
