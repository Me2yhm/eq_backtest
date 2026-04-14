"""
Portfolio simulation engine.

Public API
----------
generate_portfolio(pool, port_size, ...)  ->  PortfolioResult
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from numba import njit

from data_loader import BacktestDataset
from plotting import plot_position_heatmap


@dataclass(slots=True)
class PortfolioResult:
    positions: pd.DataFrame
    close_counts: pd.DataFrame
    portfolio_returns: pd.Series
    turnover: pd.Series
    held_counts: pd.Series


def _build_day_orders(pool: BacktestDataset, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
    cached = pool.sort_cache.get(ascending)
    if cached is not None:
        return cached

    counts = np.empty(len(pool.dates), dtype=np.int64)
    ordered_days: list[np.ndarray] = []
    pred = pool.pred

    for day_idx in range(len(pool.dates)):
        start = int(pool.day_offsets[day_idx])
        end = int(pool.day_offsets[day_idx + 1])
        day_rows = np.arange(start, end, dtype=np.int32)
        mask = np.isfinite(pred[start:end])
        ranked_rows = day_rows[mask]
        if ranked_rows.size:
            local_order = np.arange(ranked_rows.size, dtype=np.int32)
            day_pred = pred[ranked_rows]
            sorter = np.lexsort((local_order, day_pred if ascending else -day_pred))
            ranked_rows = ranked_rows[sorter]
        ordered_days.append(ranked_rows)
        counts[day_idx] = ranked_rows.size

    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    sorted_rows = np.concatenate(ordered_days) if ordered_days else np.empty(0, dtype=np.int32)
    pool.sort_cache[ascending] = (sorted_rows, offsets)
    return sorted_rows, offsets


@njit(cache=True)
def _simulate_portfolio_core(
    day_offsets: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_offsets: np.ndarray,
    row_symbol_ids: np.ndarray,
    ret: np.ndarray,
    size_rank: np.ndarray,
    tradable: np.ndarray,
    can_close: np.ndarray,
    can_open_base: np.ndarray,
    can_open: np.ndarray,
    n_symbols: int,
    port_size: int,
    thresh_out: int,
    size_cut: int,
    close_on_size_drop: bool,
    strict_first_day_top_n: bool,
    trade_on_next_day: bool,
) -> tuple:
    n_days = len(day_offsets) - 1
    first_execution_day = 1 if trade_on_next_day else 0

    held = np.zeros(n_symbols, dtype=np.bool_)
    current_row = np.full(n_symbols, -1, dtype=np.int64)
    last_row = np.full(n_symbols, -1, dtype=np.int64)
    rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    signal_in_size_pool = np.zeros(n_symbols, dtype=np.bool_)
    signal_can_open_pool = np.zeros(n_symbols, dtype=np.bool_)
    held_symbols = np.full(port_size, -1, dtype=np.int32)

    held_count = 0
    close_counts = np.zeros(n_days, dtype=np.int32)
    daily_turnover = np.zeros(n_days, dtype=np.float64)
    daily_returns = np.zeros(n_days, dtype=np.float64)
    held_counts = np.zeros(n_days, dtype=np.int32)

    record_capacity = n_days * port_size
    rec_date_idx = np.empty(record_capacity, dtype=np.int32)
    rec_source_row = np.empty(record_capacity, dtype=np.int64)
    rec_pred_rank = np.empty(record_capacity, dtype=np.float64)
    rec_size_rank = np.empty(record_capacity, dtype=np.float64)
    rec_ret = np.empty(record_capacity, dtype=np.float64)
    rec_tradable = np.empty(record_capacity, dtype=np.bool_)
    rec_count = 0

    for day_idx in range(n_days):
        prev_held = held.copy()
        current_row[:] = -1
        rank_by_symbol[:] = 0
        signal_in_size_pool[:] = False
        signal_can_open_pool[:] = False

        day_start = day_offsets[day_idx]
        day_end = day_offsets[day_idx + 1]
        for row_idx in range(day_start, day_end):
            symbol_id = row_symbol_ids[row_idx]
            current_row[symbol_id] = row_idx

        signal_day_idx = day_idx - 1 if trade_on_next_day else day_idx
        has_signal = signal_day_idx >= 0
        order_start = 0
        order_end = 0

        if has_signal:
            rank = 0
            order_start = sorted_offsets[signal_day_idx]
            order_end = sorted_offsets[signal_day_idx + 1]
            for pos in range(order_start, order_end):
                signal_row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row_idx]
                if size_rank[signal_row_idx] < size_cut:
                    signal_in_size_pool[symbol_id] = True
                    if can_open_base[signal_row_idx]:
                        signal_can_open_pool[symbol_id] = True
                if signal_can_open_pool[symbol_id] or held[symbol_id]:
                    rank += 1
                    rank_by_symbol[symbol_id] = rank

        n_closed = 0
        if has_signal:
            new_count = 0
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                should_close = False
                if row_idx != -1:
                    last_row[symbol_id] = row_idx
                    if can_close[row_idx]:
                        if rank_by_symbol[symbol_id] == 0 or rank_by_symbol[symbol_id] > thresh_out:
                            should_close = True
                        if close_on_size_drop and not signal_in_size_pool[symbol_id]:
                            should_close = True

                if should_close:
                    held[symbol_id] = False
                    n_closed += 1
                else:
                    held_symbols[new_count] = symbol_id
                    new_count += 1

            held_count = new_count

        n_opened = 0
        if has_signal:
            for pos in range(order_start, order_end):
                if held_count == port_size:
                    break

                signal_row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row_idx]
                exec_row_idx = current_row[symbol_id]

                if day_idx == first_execution_day and strict_first_day_top_n and rank_by_symbol[symbol_id] > port_size:
                    continue

                if (
                    signal_can_open_pool[symbol_id]
                    and exec_row_idx != -1
                    and can_open[exec_row_idx]
                    and not held[symbol_id]
                ):
                    held[symbol_id] = True
                    held_symbols[held_count] = symbol_id
                    held_count += 1
                    last_row[symbol_id] = exec_row_idx
                    n_opened += 1

        close_counts[day_idx] = n_closed
        changed_count = 0
        for symbol_id in range(n_symbols):
            if held[symbol_id] != prev_held[symbol_id]:
                changed_count += 1
        daily_turnover[day_idx] = changed_count / port_size
        held_counts[day_idx] = held_count

        day_ret_sum = 0.0
        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]

            rec_date_idx[rec_count] = day_idx

            if row_idx == -1:
                rec_source_row[rec_count] = last_row[symbol_id]
                rec_pred_rank[rec_count] = np.nan
                rec_size_rank[rec_count] = np.nan
                rec_ret[rec_count] = 0.0
                rec_tradable[rec_count] = False
            else:
                last_row[symbol_id] = row_idx
                rec_source_row[rec_count] = row_idx
                rec_size_rank[rec_count] = float(size_rank[row_idx])
                rec_ret[rec_count] = ret[row_idx]
                rec_tradable[rec_count] = tradable[row_idx]
                if tradable[row_idx] and rank_by_symbol[symbol_id] > 0:
                    rec_pred_rank[rec_count] = float(rank_by_symbol[symbol_id])
                else:
                    rec_pred_rank[rec_count] = np.nan

            day_ret_sum += rec_ret[rec_count]
            rec_count += 1

        daily_returns[day_idx] = day_ret_sum / port_size

    return (
        rec_date_idx[:rec_count],
        rec_source_row[:rec_count],
        rec_pred_rank[:rec_count],
        rec_size_rank[:rec_count],
        rec_ret[:rec_count],
        rec_tradable[:rec_count],
        close_counts,
        daily_turnover,
        daily_returns,
        held_counts,
    )


def _materialize_positions(
    pool: BacktestDataset,
    rec_date_idx: np.ndarray,
    rec_source_row: np.ndarray,
    rec_pred_rank: np.ndarray,
    rec_size_rank: np.ndarray,
    rec_ret: np.ndarray,
    rec_tradable: np.ndarray,
) -> pd.DataFrame:
    columns = [
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
        "tradable",
        "can_open",
        "pred",
        "pred_rank",
    ]
    if rec_source_row.size == 0:
        empty = pd.DataFrame(columns=["date", "symbol", *columns])
        return empty.set_index(["date", "symbol"])

    records = pl.DataFrame(
        {
            "row_idx": rec_source_row,
            "snapshot_date": pool.dates[rec_date_idx],
            "ret": rec_ret,
            "size_rank": rec_size_rank,
            "tradable": rec_tradable,
            "pred_rank": rec_pred_rank,
        }
    )

    base_columns = [
        "row_idx",
        "symbol",
        "turnover",
        "log_size",
        "industry",
        "index",
        "is_limit_up",
        "is_limit_down",
        "listed_Satisfied",
        "is_ST",
        "normal_days",
        "can_open",
        "pred",
    ]

    positions = (
        records.join(pool.pool_frame.select(base_columns), on="row_idx", how="left")
        .rename({"snapshot_date": "date"})
        .select(
            [
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
                "tradable",
                "can_open",
                "pred",
                "pred_rank",
            ]
        )
    )

    result = positions.to_pandas()
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "symbol"]).sort_index()


# ── Public API ────────────────────────────────────────────────────────────────


def generate_portfolio(
    pool: BacktestDataset,
    port_size: int,
    thresh_out_buffer: int = 200,
    size_cut: int = 9999,
    close_on_size_drop: bool = True,
    trade_on_next_day: bool = True,
    strict_first_day_top_n: bool = False,
    is_short: bool = True,
    plot_heatmap: bool = True,
    output_dir: str = "output/",
) -> PortfolioResult:
    """
    Simulate a daily-rebalanced equal-weight portfolio.

    Each trading day:
      1. Rank eligible stocks by prediction score.
      2. Close tradable positions whose rank exceeds thresh_out.
      3. Fill vacant slots from the top-ranked tradable candidates.
    Non-tradable positions are force-held until they become tradable again.

    Parameters
    ----------
    pool              : BacktestDataset from build_pool()
    port_size         : target number of holdings  (= thresh_in)
    thresh_out_buffer : exit trigger = port_size + thresh_out_buffer
    size_cut          : market-cap rank upper bound for the eligible universe
    close_on_size_drop: also close when a stock falls outside size_cut
    trade_on_next_day : if True, day T predictions are executed on day T+1
    strict_first_day_top_n: if True, day 0 only opens names ranked within the top port_size window
    is_short          : rank ascending if True (short worst), descending if False (long best)
    plot_heatmap      : save a size-rank distribution heatmap to output_dir

    Returns
    -------
    PortfolioResult with positions, close counts, daily returns, turnover, and held counts.
    """
    thresh_out = port_size + thresh_out_buffer
    sorted_rows, sorted_offsets = _build_day_orders(pool, ascending=is_short)
    can_open_exec = (pool.can_trade_sell if is_short else pool.can_trade_buy) & pool.can_open_base
    can_close_exec = pool.can_trade_buy if is_short else pool.can_trade_sell
    (
        rec_date_idx,
        rec_source_row,
        rec_pred_rank,
        rec_size_rank,
        rec_ret,
        rec_tradable,
        close_counts_arr,
        turnover_arr,
        port_ret_arr,
        held_counts_arr,
    ) = _simulate_portfolio_core(
        day_offsets=pool.day_offsets,
        sorted_rows=sorted_rows,
        sorted_offsets=sorted_offsets,
        row_symbol_ids=pool.row_symbol_ids,
        ret=pool.ret,
        size_rank=pool.size_rank,
        tradable=pool.tradable,
        can_close=can_close_exec,
        can_open_base=pool.can_open_base,
        can_open=can_open_exec,
        n_symbols=len(pool.symbols),
        port_size=port_size,
        thresh_out=thresh_out,
        size_cut=size_cut,
        close_on_size_drop=close_on_size_drop,
        strict_first_day_top_n=strict_first_day_top_n,
        trade_on_next_day=trade_on_next_day,
    )

    positions = _materialize_positions(
        pool=pool,
        rec_date_idx=rec_date_idx,
        rec_source_row=rec_source_row,
        rec_pred_rank=rec_pred_rank,
        rec_size_rank=rec_size_rank,
        rec_ret=rec_ret,
        rec_tradable=rec_tradable,
    )

    dates = pd.to_datetime(pool.dates)
    close_counts = pd.DataFrame({"n_closed": close_counts_arr}, index=dates)
    portfolio_returns = pd.Series(port_ret_arr, index=dates, name="portfolio_return")
    turnover = pd.Series(turnover_arr, index=dates, name="turnover")
    held_counts = pd.Series(held_counts_arr, index=dates, name="held_count")

    if plot_heatmap:
        plot_position_heatmap(positions, port_num=port_size, is_short=is_short, output_path=output_dir)

    return PortfolioResult(
        positions=positions,
        close_counts=close_counts,
        portfolio_returns=portfolio_returns,
        turnover=turnover,
        held_counts=held_counts,
    )
