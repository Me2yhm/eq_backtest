"""
Backtest entry point.

Usage
-----
    python run.py
"""

from __future__ import annotations

import os

import pandas as pd

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
    positions: pd.DataFrame,
    bm_ret: pd.Series,
    port_size: int,
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
    turnover   : daily one-way turnover Series
    """
    port_ret = positions.groupby("date")["ret"].sum() / port_size

    # Equal weights (1/N per stock, 0 when not held)
    weights = (
        pd.Series(1 / port_size, index=positions.index, name="w")
        .reset_index()
        .pivot(index="date", columns="symbol")
        .fillna(0)
    )
    trades = weights.diff().abs()
    trades.iloc[0] = weights.iloc[0].abs()
    turnover = trades.sum(axis=1)

    excess = (bm_ret - port_ret) if is_short else (port_ret - bm_ret)
    excess -= turnover.mul(cost)

    if exclude_period:
        lo = pd.to_datetime(exclude_period[0])
        hi = pd.to_datetime(exclude_period[1])
        excess.loc[(excess.index >= lo) & (excess.index < hi)] = 0.0

    return excess, turnover


# ── Logging helper ─────────────────────────────────────────────────────────────


def _log_holding_stats(stats: dict, port_size: int) -> None:
    avg_open = stats["avg_open"]
    avg_closed = stats["avg_closed"]
    print(f"  Holding (open):   avg {avg_open.mean():.1f} d, median {avg_open.median():.1f} d")
    if not avg_closed.empty:
        print(f"  Holding (closed): avg {avg_closed.mean():.1f} d")


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
    )

    # ── Run for each portfolio size ───────────────────────────────────────────
    all_metrics = {}
    all_ret = {}
    all_cumrets = {}
    all_port_sizes = {}
    all_closes = {}

    for port_size in cfg.PORT_SIZES:
        print(f"\n── Portfolio size: {port_size} ──")

        positions, close_counts = generate_portfolio(
            pool=pool,
            port_size=port_size,
            thresh_out_buffer=cfg.THRESH_OUT_BUFFER,
            size_cut=cfg.POOL_SIZE,
            close_on_size_drop=True,
            is_short=cfg.IS_SHORT,
            output_dir=output_dir,
        )

        positions.to_csv(f"{output_dir}positions_{port_size}.csv")

        excess, turnover = compute_returns(
            positions,
            bm_ret,
            port_size,
            is_short=cfg.IS_SHORT,
            cost=cfg.COST_PER_TURNOVER,
            exclude_period=cfg.EXCLUDE_PERIOD,
        )

        m = portfolio_metrics(excess)
        m["Ann. Turnover"] = float(turnover.mean() * 242)
        all_metrics[port_size] = m
        all_ret[port_size] = excess.rename(f"Port_{port_size}")
        all_cumrets[port_size] = excess.cumsum().rename(f"Port_{port_size}")
        all_port_sizes[port_size] = positions.groupby("date").size().rename(f"Port_{port_size}")

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

    print("\n── Metrics ──")
    print(metrics_df.to_string())

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
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
