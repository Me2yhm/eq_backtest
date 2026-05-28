"""
15-minute portfolio simulation engine.

Public API
----------
generate_portfolio_15min(pool, port_size, ...)  ->  PortfolioResult15Min
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from numba import njit

from data_loader_15min import BacktestDataset15Min
from plotting import plot_position_heatmap


@dataclass(slots=True)
class PortfolioResult15Min:
    positions: pd.DataFrame
    close_counts: pd.DataFrame
    portfolio_returns: pd.Series
    cost_turnover: pd.Series
    turnover: pd.Series
    held_counts: pd.Series


def _build_bar_orders(pool: BacktestDataset15Min, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
    cached = pool.sort_cache.get(ascending)
    if cached is not None:
        return cached

    counts = np.empty(len(pool.bars), dtype=np.int64)
    ordered_bars: list[np.ndarray] = []
    pred = pool.pred

    for bar_idx in range(len(pool.bars)):
        start = int(pool.bar_offsets[bar_idx])
        end = int(pool.bar_offsets[bar_idx + 1])
        bar_rows = np.arange(start, end, dtype=np.int32)
        mask = np.isfinite(pred[start:end])
        ranked_rows = bar_rows[mask]
        if ranked_rows.size:
            local_order = np.arange(ranked_rows.size, dtype=np.int32)
            bar_pred = pred[ranked_rows]
            sorter = np.lexsort((local_order, bar_pred if ascending else -bar_pred))
            ranked_rows = ranked_rows[sorter]
        ordered_bars.append(ranked_rows)
        counts[bar_idx] = ranked_rows.size

    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    sorted_rows = np.concatenate(ordered_bars) if ordered_bars else np.empty(0, dtype=np.int32)
    pool.sort_cache[ascending] = (sorted_rows, offsets)
    return sorted_rows, offsets


@njit(cache=True)
def _simulate_portfolio_core_15min(
    bar_offsets: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_offsets: np.ndarray,
    row_symbol_ids: np.ndarray,
    vwap_ret: np.ndarray,
    pred: np.ndarray,
    size_rank: np.ndarray,
    tradable: np.ndarray,
    can_open_base: np.ndarray,
    can_open: np.ndarray,
    can_close: np.ndarray,
    n_symbols: int,
    port_size: int,
    thresh_out: int,
    size_cut: int,
    close_on_size_drop: bool,
    strict_first_bar_top_n: bool,
    trade_on_next_bar: bool,
    is_short: bool,
) -> tuple:
    n_bars = len(bar_offsets) - 1
    first_execution_bar = 1 if trade_on_next_bar else 0

    held = np.zeros(n_symbols, dtype=np.bool_)
    current_row = np.full(n_symbols, -1, dtype=np.int64)
    rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    signal_in_size_pool = np.zeros(n_symbols, dtype=np.bool_)
    signal_can_open_pool = np.zeros(n_symbols, dtype=np.bool_)
    held_symbols = np.full(port_size, -1, dtype=np.int32)

    prev_weights = np.zeros(n_symbols, dtype=np.float64)
    curr_weights = np.zeros(n_symbols, dtype=np.float64)
    prev_symbols = np.full(n_symbols, -1, dtype=np.int32)
    curr_symbols = np.full(port_size, -1, dtype=np.int32)
    touched = np.zeros(n_symbols, dtype=np.bool_)
    touched_symbols = np.full(n_symbols, -1, dtype=np.int32)

    held_count = 0
    prev_count = 0
    close_counts = np.zeros(n_bars, dtype=np.int32)
    bar_turnover = np.zeros(n_bars, dtype=np.float64)
    bar_returns = np.zeros(n_bars, dtype=np.float64)
    held_counts = np.zeros(n_bars, dtype=np.int32)

    record_capacity = n_bars * port_size
    rec_bar_idx = np.empty(record_capacity, dtype=np.int32)
    rec_source_row = np.empty(record_capacity, dtype=np.int64)
    rec_pred_rank = np.empty(record_capacity, dtype=np.float64)
    rec_size_rank = np.empty(record_capacity, dtype=np.float64)
    rec_vwap_ret = np.empty(record_capacity, dtype=np.float64)
    rec_tradable = np.empty(record_capacity, dtype=np.bool_)
    rec_count = 0

    portfolio_sign = -1.0 if is_short else 1.0

    for bar_idx in range(n_bars):
        current_row[:] = -1
        rank_by_symbol[:] = 0
        signal_in_size_pool[:] = False
        signal_can_open_pool[:] = False

        bar_start = bar_offsets[bar_idx]
        bar_end = bar_offsets[bar_idx + 1]
        for row_idx in range(bar_start, bar_end):
            symbol_id = row_symbol_ids[row_idx]
            current_row[symbol_id] = row_idx

        signal_bar_idx = bar_idx - 1 if trade_on_next_bar else bar_idx
        has_signal = signal_bar_idx >= 0
        order_start = 0
        order_end = 0

        if has_signal:
            rank = 0
            order_start = sorted_offsets[signal_bar_idx]
            order_end = sorted_offsets[signal_bar_idx + 1]
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

                if row_idx != -1 and can_close[row_idx]:
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

        if has_signal:
            for pos in range(order_start, order_end):
                if held_count == port_size:
                    break

                signal_row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row_idx]
                exec_row_idx = current_row[symbol_id]

                if bar_idx == first_execution_bar and strict_first_bar_top_n and rank_by_symbol[symbol_id] > port_size:
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

        close_counts[bar_idx] = n_closed

        bar_ret = 0.0
        curr_count = 0
        if held_count > 0:
            target_weight = portfolio_sign / held_count
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                if row_idx == -1:
                    continue

                curr_weights[symbol_id] = target_weight
                curr_symbols[curr_count] = symbol_id
                curr_count += 1

                r = vwap_ret[row_idx]
                if np.isfinite(r):
                    bar_ret += target_weight * r

        touched_count = 0
        for i in range(prev_count):
            symbol_id = prev_symbols[i]
            if not touched[symbol_id]:
                touched[symbol_id] = True
                touched_symbols[touched_count] = symbol_id
                touched_count += 1
        for i in range(curr_count):
            symbol_id = curr_symbols[i]
            if not touched[symbol_id]:
                touched[symbol_id] = True
                touched_symbols[touched_count] = symbol_id
                touched_count += 1

        turnover = 0.0
        for i in range(touched_count):
            symbol_id = touched_symbols[i]
            touched[symbol_id] = False
            turnover += abs(curr_weights[symbol_id] - prev_weights[symbol_id])

        turnover *= 0.5
        bar_turnover[bar_idx] = turnover
        bar_returns[bar_idx] = bar_ret

        for i in range(prev_count):
            symbol_id = prev_symbols[i]
            if curr_weights[symbol_id] == 0.0:
                prev_weights[symbol_id] = 0.0

        next_prev_count = 0
        for i in range(curr_count):
            symbol_id = curr_symbols[i]
            prev_weights[symbol_id] = curr_weights[symbol_id]
            prev_symbols[next_prev_count] = symbol_id
            next_prev_count += 1
            curr_weights[symbol_id] = 0.0

        prev_count = next_prev_count
        held_counts[bar_idx] = held_count

        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]
            if row_idx == -1:
                continue

            rec_bar_idx[rec_count] = bar_idx
            rec_source_row[rec_count] = row_idx
            rec_size_rank[rec_count] = float(size_rank[row_idx])
            rec_vwap_ret[rec_count] = vwap_ret[row_idx]
            rec_tradable[rec_count] = tradable[row_idx]
            if tradable[row_idx] and rank_by_symbol[symbol_id] > 0:
                rec_pred_rank[rec_count] = float(rank_by_symbol[symbol_id])
            else:
                rec_pred_rank[rec_count] = np.nan
            rec_count += 1

    return (
        rec_bar_idx[:rec_count],
        rec_source_row[:rec_count],
        rec_pred_rank[:rec_count],
        rec_size_rank[:rec_count],
        rec_vwap_ret[:rec_count],
        rec_tradable[:rec_count],
        close_counts,
        bar_turnover,
        bar_returns,
        held_counts,
    )


def _materialize_positions_15min(
    pool: BacktestDataset15Min,
    rec_bar_idx: np.ndarray,
    rec_source_row: np.ndarray,
    rec_pred_rank: np.ndarray,
    rec_size_rank: np.ndarray,
    rec_vwap_ret: np.ndarray,
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
        "vwap_ret",
    ]
    if rec_source_row.size == 0:
        empty = pd.DataFrame(columns=["date", "symbol", *columns])
        return empty.set_index(["date", "symbol"])

    records = pl.DataFrame(
        {
            "row_idx": rec_source_row,
            "snapshot_bar": pool.bars[rec_bar_idx],
            "ret": rec_vwap_ret,
            "size_rank": rec_size_rank,
            "tradable": rec_tradable,
            "pred_rank": rec_pred_rank,
            "vwap_ret": rec_vwap_ret,
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
        .rename({"snapshot_bar": "date"})
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
                "vwap_ret",
            ]
        )
    )

    result = positions.to_pandas()
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "symbol"]).sort_index()


def generate_portfolio_15min(
    pool: BacktestDataset15Min,
    port_size: int,
    thresh_out_buffer: int = 200,
    size_cut: int = 9999,
    close_on_size_drop: bool = True,
    trade_on_next_bar: bool = True,
    strict_first_bar_top_n: bool = False,
    is_short: bool = True,
    plot_heatmap: bool = True,
    output_dir: str = "output/",
) -> PortfolioResult15Min:
    """Simulate a 15-minute equal-weight portfolio with daily constraints."""
    thresh_out = port_size + thresh_out_buffer
    sorted_rows, sorted_offsets = _build_bar_orders(pool, ascending=is_short)

    can_open_exec = (pool.can_trade_sell if is_short else pool.can_trade_buy) & pool.can_open_base
    can_close_exec = pool.can_trade_buy if is_short else pool.can_trade_sell

    (
        rec_bar_idx,
        rec_source_row,
        rec_pred_rank,
        rec_size_rank,
        rec_vwap_ret,
        rec_tradable,
        close_counts_arr,
        turnover_arr,
        port_ret_arr,
        held_counts_arr,
    ) = _simulate_portfolio_core_15min(
        bar_offsets=pool.bar_offsets,
        sorted_rows=sorted_rows,
        sorted_offsets=sorted_offsets,
        row_symbol_ids=pool.row_symbol_ids,
        vwap_ret=pool.vwap_ret,
        pred=pool.pred,
        size_rank=pool.size_rank,
        tradable=pool.tradable,
        can_open_base=pool.can_open_base,
        can_open=can_open_exec,
        can_close=can_close_exec,
        n_symbols=len(pool.symbols),
        port_size=port_size,
        thresh_out=thresh_out,
        size_cut=size_cut,
        close_on_size_drop=close_on_size_drop,
        strict_first_bar_top_n=strict_first_bar_top_n,
        trade_on_next_bar=trade_on_next_bar,
        is_short=is_short,
    )

    positions = _materialize_positions_15min(
        pool=pool,
        rec_bar_idx=rec_bar_idx,
        rec_source_row=rec_source_row,
        rec_pred_rank=rec_pred_rank,
        rec_size_rank=rec_size_rank,
        rec_vwap_ret=rec_vwap_ret,
        rec_tradable=rec_tradable,
    )

    bars = pd.to_datetime(pool.bars)
    close_counts = pd.DataFrame({"n_closed": close_counts_arr}, index=bars)
    portfolio_returns = pd.Series(port_ret_arr, index=bars, name="portfolio_return")
    turnover = pd.Series(turnover_arr, index=bars, name="turnover")
    held_counts = pd.Series(held_counts_arr, index=bars, name="held_count")

    if plot_heatmap:
        plot_position_heatmap(positions, port_num=port_size, is_short=is_short, output_path=output_dir)

    return PortfolioResult15Min(
        positions=positions,
        close_counts=close_counts,
        portfolio_returns=portfolio_returns,
        cost_turnover=turnover.copy().rename("cost_turnover"),
        turnover=turnover,
        held_counts=held_counts,
    )
