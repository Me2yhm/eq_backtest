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
    deduct_cost: bool = True,
) -> tuple:
    """
    Compute daily excess returns and one-way turnover for a single portfolio.

    Excess return = benchmark - portfolio (short) or portfolio - benchmark (long),
    minus transaction cost (only when deduct_cost=True; 15min 模式引擎已扣成本).

    Returns
    -------
    excess_ret : daily excess return Series
    turnover   : daily one-way turnover Series for reporting
    """
    bm_aligned = bm_ret.reindex(portfolio_returns.index)
    excess = (bm_aligned - portfolio_returns) if is_short else (portfolio_returns - bm_aligned)
    if deduct_cost:
        excess -= cost_turnover.mul(cost)

    # exclude_period: 删除行 (对齐 evaluation/metrics.py L252-259 _apply_metric_filters)
    # 旧口径: 置零 (excess.loc[...] = 0.0)
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
    if isinstance(index, pd.DatetimeIndex):
        bm_daily = bm_ret.copy()
        bm_daily.index = pd.to_datetime(bm_daily.index).normalize()
        if (index != index.normalize()).any():
            aligned_values = bm_daily.reindex(index.normalize()).to_numpy()
            return pd.Series(aligned_values, index=index, name=bm_ret.name)
    return bm_ret.reindex(index)


def _write_positions_csv(positions: pd.DataFrame, output_path: str) -> None:
    df = positions.reset_index()
    # 直接将 DataFrame 转换为 Polars，不改变 date 列的类型
    frame = pl.from_pandas(df)
    frame.write_csv(output_path)
    # frame = pl.from_pandas(positions.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    # frame.write_csv(output_path)


def _build_portfolio_pnl_frame(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    close_counts: pd.DataFrame,
    is_short: bool,
    cost: float,
    exclude_period: tuple | None,
    deduct_cost: bool = True,
) -> pd.DataFrame:
    daily_benchmark = _align_benchmark_to_index(bm_ret, portfolio_returns.index).fillna(0.0).rename("daily_benchmark")
    daily_tto = turnover.rename("daily_tto").copy()
    if deduct_cost:
        daily_strategy = portfolio_returns.sub(cost_turnover.mul(cost)).rename("daily_strategy")
    else:
        # Retained for callers that pass already-net returns.
        daily_strategy = portfolio_returns.rename("daily_strategy")
    daily_alpha = (daily_strategy + daily_benchmark) if is_short else (daily_strategy - daily_benchmark)
    daily_alpha = daily_alpha.rename("daily_alpha")

    all_pl = daily_strategy.cumsum().rename("all_pl")
    alpha_pl = daily_alpha.cumsum().rename("alpha_pl")
    benchmark = daily_benchmark.cumsum().rename("benchmark")
    close_count = close_counts["n_closed"].reindex(portfolio_returns.index).fillna(0).astype(int).rename("close_count")

    # exclude_period: 删除行 (对齐 evaluation/metrics.py L252-259)
    # 旧口径: 置零 daily_strategy/daily_alpha/daily_tto
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
        freq_cfg=cfg.FREQ_CONFIG[cfg.FREQUENCY],
        bm_path=cfg.BM_PATH,
        start=cfg.START,
        end=cfg.END,
        universe=cfg.UNIVERSE,
        allow_st_open=cfg.ALLOW_ST_OPEN,
        use_cache=cfg.USE_POOL_CACHE,
        cache_dir=cfg.POOL_CACHE_DIR,
        nosuspend_days=cfg.NOSUSPEND_DAYS,
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
            trade_on_next_bar=cfg.TRADE_ON_NEXT_BAR,
            strict_first_bar_top_n=cfg.STRICT_FIRST_BAR_TOP_N,
            is_short=cfg.IS_SHORT,
            output_dir=output_dir,
            debug_mode=cfg.DEBUG,
            debug_symbol=cfg.DEBUG_SYMBOL,
            debug_datetime=cfg.DEBUG_DATETIME,
        )

        positions = result.positions
        close_counts = result.close_counts
        _write_positions_csv(positions, f"{output_dir}positions_{port_size}.csv")
        is_intraday = cfg.FREQUENCY != "daily"
        if is_intraday:
            portfolio_returns_eval = daily_returns_from_intraday(result.portfolio_returns, agg_mode=cfg.AGG_MODE)
            cost_turnover_eval = daily_sum_from_intraday(result.cost_turnover, "cost_turnover")
            turnover_eval = daily_sum_from_intraday(result.turnover, "turnover")
            close_counts_eval = pd.DataFrame({
                "n_closed": daily_sum_from_intraday(close_counts["n_closed"], "n_closed").astype(int)
            })
        else:
            portfolio_returns_eval = result.portfolio_returns
            cost_turnover_eval = result.cost_turnover
            turnover_eval = result.turnover
            close_counts_eval = close_counts

        # The common engine always reports gross P&L and separate one-way turnover.
        deduct_cost = True
        excess, turnover = compute_returns(
            portfolio_returns_eval,
            cost_turnover_eval,
            turnover_eval,
            bm_ret,
            is_short=cfg.IS_SHORT,
            cost=cfg.COST_PER_TURNOVER,
            exclude_period=cfg.EXCLUDE_PERIOD,
            deduct_cost=deduct_cost,
        )
        portfolio_pnl = _build_portfolio_pnl_frame(
            portfolio_returns=portfolio_returns_eval,
            cost_turnover=cost_turnover_eval,
            turnover=turnover_eval,
            bm_ret=bm_ret,
            close_counts=close_counts_eval,
            is_short=cfg.IS_SHORT,
            cost=cfg.COST_PER_TURNOVER,
            exclude_period=cfg.EXCLUDE_PERIOD,
            deduct_cost=deduct_cost,
        )
        portfolio_pnl.to_csv(
            f"{output_dir}portfolio_pnl_{port_size}.csv",
            index_label="date",
            float_format="%.8f",
        )

        m = portfolio_metrics(excess, compounding=cfg.COMPOUNDING)
        m["Ann. Turnover"] = float(turnover.mean() * 242)
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
