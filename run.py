"""
Backtest entry point.

Usage
-----
    python run.py runs/my-backtest
"""

from __future__ import annotations

import argparse
import json
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
from optimizer_adapter import prepare_optimizer_targets
from optimizer_client import create_optimizer_client


@dataclass(slots=True)
class PortfolioEvaluation:
    result: PortfolioResult
    portfolio_returns: pd.Series
    borrow_cost: pd.Series
    turnover: pd.Series
    close_counts: pd.DataFrame
    excess: pd.Series
    metrics: dict
    portfolio_pnl: pd.DataFrame

# ── Return computation ─────────────────────────────────────────────────────────



def _strategy_mode(strategy_mode: str | None, is_short: bool | None) -> str:
    mode = strategy_mode or ("short_only" if is_short else "long_only")
    if mode not in {"long_only", "short_only", "long_short"}:
        raise ValueError(f"Unsupported strategy mode: {mode!r}")
    return mode


def _evaluation_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series, mode: str) -> pd.Series:
    if mode == "long_only":
        return portfolio_returns - benchmark_returns
    if mode == "short_only":
        return portfolio_returns + benchmark_returns
    return portfolio_returns


def compute_returns(
    portfolio_returns: pd.Series,
    cost_turnover: pd.Series,
    turnover: pd.Series,
    bm_ret: pd.Series,
    is_short: bool | None = None,
    exclude_period: tuple | None = None,
    strategy_mode: str | None = None,
    benchmark_missing_return_policy: str | None = None,
) -> tuple:
    """
    Compute daily excess returns and one-way turnover for a single portfolio.

    Evaluation return is strategy minus benchmark for long-only, benchmark plus
    signed short P&L for short-only, and combined signed P&L for long-short.
    Transaction costs are already deducted by the share-based simulator.

    Returns
    -------
    excess_ret : daily excess return Series
    turnover   : daily one-way turnover Series for reporting
    """
    mode = _strategy_mode(strategy_mode, is_short)
    benchmark = _align_benchmark_to_index(
        bm_ret,
        portfolio_returns.index,
        missing_return_policy=benchmark_missing_return_policy,
    )
    excess = _evaluation_returns(portfolio_returns, benchmark, mode)

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


def _align_benchmark_to_index(
    bm_ret: pd.Series,
    index: pd.Index,
    *,
    missing_return_policy: str | None = None,
) -> pd.Series:
    """Align daily benchmark returns and apply the configured missing-return policy."""
    policy = missing_return_policy or cfg.BENCHMARK_MISSING_RETURN_POLICY
    if policy not in {"error", "zero"}:
        raise ValueError("benchmark_missing_return_policy must be 'error' or 'zero'")
    bm_daily = bm_ret.copy()
    bm_daily.index = pd.to_datetime(bm_daily.index).normalize()
    if bm_daily.index.has_duplicates:
        bm_daily = bm_daily.groupby(level=0).sum()
    target_index = index.normalize() if isinstance(index, pd.DatetimeIndex) else pd.to_datetime(index).normalize()
    aligned = pd.Series(bm_daily.reindex(target_index).to_numpy(), index=index, name=bm_ret.name)
    missing = aligned.isna()
    if not missing.any():
        return aligned
    if policy == "zero":
        return aligned.fillna(0.0)
    missing_dates = pd.DatetimeIndex(aligned.index[missing]).normalize().unique()
    preview = ", ".join(stamp.strftime("%Y-%m-%d") for stamp in missing_dates[:5])
    suffix = "" if len(missing_dates) <= 5 else ", ..."
    raise ValueError(
        f"Benchmark returns are missing for {len(missing_dates)} backtest date(s): {preview}{suffix}. "
        "Set benchmark_missing_return_policy: zero only when zero is intended."
    )


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
    borrow_cost: pd.Series,
    bm_ret: pd.Series,
    close_counts: pd.DataFrame,
    is_short: bool | None = None,
    exclude_period: tuple | None = None,
    strategy_mode: str | None = None,
    benchmark_missing_return_policy: str | None = None,
) -> pd.DataFrame:
    daily_benchmark = _align_benchmark_to_index(
        bm_ret,
        portfolio_returns.index,
        missing_return_policy=benchmark_missing_return_policy,
    ).rename("daily_benchmark")
    mode = _strategy_mode(strategy_mode, is_short)
    daily_tto = turnover.rename("daily_tto").copy()
    daily_borrow_cost = borrow_cost.reindex(portfolio_returns.index).fillna(0.0).rename("daily_borrow_cost")
    daily_strategy = portfolio_returns.rename("daily_strategy")
    daily_alpha = _evaluation_returns(portfolio_returns, daily_benchmark, mode).rename("daily_alpha")

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
        daily_borrow_cost = daily_borrow_cost[keep]
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
            daily_borrow_cost,
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
    is_short: bool | None,
    compounding: bool,
    strategy_mode: str | None = None,
    benchmark_missing_return_policy: str | None = None,
) -> PortfolioEvaluation:
    """Evaluate one simulator result using the project's excess-return metric."""
    is_intraday = frequency != "daily"
    if is_intraday:
        portfolio_returns = daily_returns_from_intraday(result.portfolio_returns, agg_mode=agg_mode)
        turnover = daily_sum_from_intraday(result.turnover, "turnover")
        borrow_cost = daily_sum_from_intraday(result.borrow_cost, "borrow_cost")
        close_counts = pd.DataFrame({
            "n_closed": daily_sum_from_intraday(result.close_counts["n_closed"], "n_closed").astype(int)
        })
    else:
        portfolio_returns = result.portfolio_returns
        turnover = result.turnover
        borrow_cost = result.borrow_cost
        close_counts = result.close_counts

    excess, _ = compute_returns(
        portfolio_returns,
        result.cost_turnover,
        turnover,
        bm_ret,
        is_short=is_short,
        exclude_period=cfg.EXCLUDE_PERIOD,
        strategy_mode=strategy_mode,
        benchmark_missing_return_policy=benchmark_missing_return_policy,
    )
    portfolio_pnl = _build_portfolio_pnl_frame(
        portfolio_returns=portfolio_returns,
        cost_turnover=result.cost_turnover,
        turnover=turnover,
        borrow_cost=borrow_cost,
        bm_ret=bm_ret,
        close_counts=close_counts,
        is_short=is_short,
        exclude_period=cfg.EXCLUDE_PERIOD,
        strategy_mode=strategy_mode,
        benchmark_missing_return_policy=benchmark_missing_return_policy,
    )
    metrics = portfolio_metrics(excess, compounding=compounding)
    metrics["Ann. Turnover"] = float(turnover.mean() * 242) if not turnover.empty else 0.0
    return PortfolioEvaluation(
        result=result,
        portfolio_returns=portfolio_returns,
        turnover=turnover,
        borrow_cost=borrow_cost,
        close_counts=close_counts,
        excess=excess,
        metrics=metrics,
        portfolio_pnl=portfolio_pnl,
    )



def _with_sleeve(result: PortfolioResult, sleeve: str) -> PortfolioResult:
    positions = result.positions.copy()
    positions["sleeve"] = sleeve
    target_weights = None
    if result.target_weights is not None:
        target_weights = result.target_weights.copy()
        target_weights["sleeve"] = sleeve
    return PortfolioResult(
        positions=positions,
        close_counts=result.close_counts,
        portfolio_returns=result.portfolio_returns,
        cost_turnover=result.cost_turnover,
        borrow_cost=result.borrow_cost,
        turnover=result.turnover,
        held_counts=result.held_counts,
        target_weights=target_weights,
        optimizer_calls=result.optimizer_calls,
        decision_targets=result.decision_targets,
    )


def _combine_sleeves(long_result: PortfolioResult, short_result: PortfolioResult) -> PortfolioResult:
    positions = pd.concat([long_result.positions, short_result.positions]).sort_index()
    target_weights = None
    if long_result.target_weights is not None and short_result.target_weights is not None:
        target_weights = pd.concat([long_result.target_weights, short_result.target_weights]).sort_index()
    return PortfolioResult(
        positions=positions,
        close_counts=long_result.close_counts.add(short_result.close_counts, fill_value=0).astype(int),
        portfolio_returns=long_result.portfolio_returns.add(short_result.portfolio_returns, fill_value=0.0),
        cost_turnover=long_result.cost_turnover.add(short_result.cost_turnover, fill_value=0.0),
        borrow_cost=long_result.borrow_cost.add(short_result.borrow_cost, fill_value=0.0),
        turnover=long_result.turnover.add(short_result.turnover, fill_value=0.0),
        held_counts=long_result.held_counts.add(short_result.held_counts, fill_value=0).astype(int),
        target_weights=target_weights,
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
    sbl_enabled: bool = False,
) -> PortfolioEvaluation:
    if is_short and not sbl_enabled:
        raise ValueError("Short portfolios require a pool built with explicit short_borrow_sources")

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
    return evaluate_portfolio_result(
        result,
        bm_ret,
        frequency,
        agg_mode,
        is_short,
        compounding,

        strategy_mode="short_only" if is_short else "long_only",
    )

# ── Logging helper ─────────────────────────────────────────────────────────────


def _log_holding_stats(stats: dict, port_size: int) -> None:
    avg_open = stats["avg_open"]
    avg_closed = stats["avg_closed"]
    logger.info("Holding (open):   avg {:.1f} sessions, median {:.1f} sessions", avg_open.mean(), avg_open.median())
    if not avg_closed.empty:
        logger.info("Holding (closed): avg {:.1f} sessions", avg_closed.mean())


# ── Main ───────────────────────────────────────────────────────────────────────




def _short_sleeve_parameters(port_size: int) -> tuple[int, int]:
    short_port_size = cfg.SHORT_PORT_SIZE or port_size
    # Short entries use the top short_port_size ranks and matched reverse-rank
    # covers. Keep no exit buffer in the actual short rebalancing path.
    return short_port_size, 0


def _mode_result(
    pool,
    mode: str,
    *,
    port_size: int,
    output_dir: str,
    optimizer_client: OptimizerClient | None = None,
) -> PortfolioResult:
    common = {
        "pool": pool,
        "size_cut": cfg.POOL_SIZE,
        "close_on_size_drop": cfg.CLOSE_ON_SIZE_DROP,
        "trade_on_next_bar": cfg.trade_on_next_bar_for(cfg.FREQUENCY),
        "strict_first_bar_top_n": cfg.STRICT_FIRST_BAR_TOP_N,
        "output_dir": output_dir,
        "plot_heatmap": False,
        "record_target_weights": cfg.FREQUENCY != "daily" or optimizer_client is not None,
        "debug_mode": cfg.DEBUG,
        "debug_symbol": cfg.DEBUG_SYMBOL,
        "debug_datetime": cfg.DEBUG_DATETIME,
        "cost_per_turnover": cfg.COST_PER_TURNOVER,
        "portfolio_initial_value": cfg.PORTFOLIO_INITIAL_VALUE,
        "weight_mode": cfg.WEIGHT_MODE,
        "max_weight_multiple": cfg.MAX_WEIGHT_MULTIPLE,
    }
    long_common = {
        **common,
        "port_size": port_size,
        "thresh_out_buffer": cfg.THRESH_OUT_BUFFER,
    }
    if mode == "long_only":
        optimizer_plan = None
        if optimizer_client is not None:
            optimizer_plan = prepare_optimizer_targets(
                pool,
                optimizer_client,
                portfolio_id=f"long:{port_size}",
                port_size=port_size,
                thresh_out_buffer=cfg.THRESH_OUT_BUFFER,
                size_cut=cfg.POOL_SIZE,
                close_on_size_drop=cfg.CLOSE_ON_SIZE_DROP,
                model=cfg.OPTIMIZER["model"],
                timeout_ms=cfg.OPTIMIZER["timeout_ms"],
                audit_path=Path(output_dir) / "optimizer_calls.jsonl",
            )
        try:
            return _with_sleeve(
                generate_portfolio(is_short=False, optimizer_plan=optimizer_plan, **long_common),
                "long",
            )
        finally:
            if optimizer_plan is not None:
                optimizer_plan.close()
    short_port_size, short_thresh_out_buffer = _short_sleeve_parameters(port_size)
    short_common = {
        **common,
        "port_size": short_port_size,
        "thresh_out_buffer": short_thresh_out_buffer,
    }
    short_result = _with_sleeve(generate_portfolio(is_short=True, **short_common), "short")
    if mode == "short_only":
        return short_result
    long_result = _with_sleeve(generate_portfolio(is_short=False, **long_common), "long")
    return _combine_sleeves(long_result, short_result)


def _run(optimizer_client: OptimizerClient | None) -> None:
    logger.info("Run directory: {} (config: {})", cfg.RUN_DIR, cfg.CONFIG_PATH)
    needs_short = any(mode != "long_only" for mode in cfg.STRATEGY_MODES)
    if needs_short and not cfg.SHORT_BORROW_SOURCES:
        raise ValueError("short_only and long_short modes require short_borrow_sources")
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
        short_borrow_sources=cfg.SHORT_BORROW_SOURCES if needs_short else None,
        borrow_selection=cfg.BORROW_SELECTION,
    )

    for mode in cfg.STRATEGY_MODES:
        mode_dir = cfg.output_dir_for_mode(mode)
        mode_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(mode_dir) + os.sep
        if cfg.OPTIMIZER["enabled"]:
            (mode_dir / "optimizer_calls.jsonl").write_text("", encoding="utf-8")
        decision_target_frames: list[pd.DataFrame] = []
        all_metrics, all_ret, all_cumrets, all_port_sizes, all_closes = {}, {}, {}, {}, {}
        for port_size in cfg.PORT_SIZES:
            logger.info("── {} portfolio size: {} ──", mode, port_size)
            result = _mode_result(
                pool, mode, port_size=port_size, output_dir=output_dir,
                optimizer_client=optimizer_client,
            )
            _write_positions_csv(result.positions, f"{output_dir}positions_{port_size}.csv")
            if result.target_weights is not None:
                _write_target_weights(
                    result.target_weights,
                    f"{output_dir}target_weights_{port_size}.csv",
                    f"{output_dir}target_weights_{port_size}.parquet",
                )
            if result.decision_targets is not None:
                decision_target_frames.append(result.decision_targets)
                pl.from_pandas(pd.concat(decision_target_frames, ignore_index=True)).write_parquet(
                    mode_dir / "decision_targets.parquet"
                )
            evaluation = evaluate_portfolio_result(
                result=result,
                bm_ret=bm_ret,
                frequency=cfg.FREQUENCY,
                agg_mode=cfg.AGG_MODE,
                is_short=mode == "short_only",
                compounding=cfg.COMPOUNDING,
                strategy_mode=mode,
            )
            evaluation.portfolio_pnl.to_csv(
                f"{output_dir}portfolio_pnl_{port_size}.csv",
                index_label="date",
                float_format="%.8f",
            )
            all_metrics[port_size] = evaluation.metrics
            all_ret[port_size] = evaluation.excess.rename(f"Port_{port_size}")
            all_cumrets[port_size] = evaluation.excess.cumsum().rename(f"Port_{port_size}")
            all_port_sizes[port_size] = result.held_counts.rename(f"Port_{port_size}")
            if not evaluation.close_counts.empty:
                all_closes[port_size] = evaluation.close_counts.rename(columns={"n_closed": f"Port_{port_size}"})
            hp = holding_period_stats(result.positions)
            plot_holding_periods(
                hp,
                port_size,
                is_short=mode == "short_only",
                output_path=output_dir,
                strategy_mode=mode,
            )
            _log_holding_stats(hp, port_size)

        metrics_df = pd.DataFrame(all_metrics).T
        cumrets_df = pd.DataFrame(all_cumrets)
        ret_df = pd.DataFrame(all_ret)
        port_sizes_df = pd.DataFrame(all_port_sizes)
        close_df = pd.concat(list(all_closes.values()), axis=1) if all_closes else pd.DataFrame()
        tag = f"{cfg.POOL_SIZE}_{cfg.BENCHMARK_SYMBOL}"
        metrics_df.to_csv(f"{output_dir}metrics_{tag}.csv")
        cumrets_df.to_csv(f"{output_dir}cumrets_{tag}.csv")
        ret_df.to_csv(f"{output_dir}returns_{tag}.csv")
        logger.info("── {} metrics ──\n{}", mode, metrics_df.to_string())
        plot_portfolio_results(
            cumrets_df,
            port_sizes_df,
            close_df,
            is_short=mode == "short_only",
            output_path=output_dir,
            strategy_mode=mode,
        )
        plot_metrics_table(
            metrics_df,
            cfg.POOL_SIZE,
            cfg.BENCHMARK_SYMBOL,
            is_short=mode == "short_only",
            output_path=output_dir,
            strategy_mode=mode,
        )


def run() -> None:
    if not cfg.OPTIMIZER["enabled"]:
        _run(None)
        return
    status_path = cfg.RUN_DIR / "optimizer_run_status.json"
    status_path.write_text(json.dumps({"status": "running", "complete": False}), encoding="utf-8")
    try:
        # Validate the selected backend before loading external market data.
        with create_optimizer_client(cfg.OPTIMIZER) as optimizer_client:
            optimizer_client.validate_capacity(max(cfg.PORT_SIZES))
            _run(optimizer_client)
    except Exception as exc:
        status_path.write_text(json.dumps({
            "status": "failed", "complete": False,
            "error_type": type(exc).__name__, "error": str(exc),
        }, ensure_ascii=False), encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"status": "success", "complete": True}), encoding="utf-8")


if __name__ == "__main__":
    import time

    start_time = time.time()
    run()
    end_time = time.time()
    logger.info("Total execution time: {:.2f} seconds", end_time - start_time)
