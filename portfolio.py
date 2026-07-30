"""Single weight-based portfolio simulator for daily and intraday bars."""

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
    cost_turnover: pd.Series
    turnover: pd.Series
    held_counts: pd.Series


def _build_bar_orders(pool: BacktestDataset, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
    cached = pool.sort_cache.get(ascending)
    if cached is not None:
        return cached
    orders: list[np.ndarray] = []
    counts = np.empty(len(pool.bars), dtype=np.int64)
    for bar_idx in range(len(pool.bars)):
        start, end = int(pool.bar_offsets[bar_idx]), int(pool.bar_offsets[bar_idx + 1])
        rows = np.arange(start, end, dtype=np.int32)
        rows = rows[np.isfinite(pool.pred[start:end])]
        if rows.size:
            local_order = np.arange(rows.size, dtype=np.int32)
            values = pool.pred[rows]
            rows = rows[np.lexsort((local_order, values if ascending else -values))]
        orders.append(rows)
        counts[bar_idx] = rows.size
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0], offsets[1:] = 0, np.cumsum(counts)
    result = (np.concatenate(orders) if orders else np.empty(0, dtype=np.int32), offsets)
    pool.sort_cache[ascending] = result
    return result


@njit(cache=True)
def _simulate_portfolio_core(
    bar_offsets: np.ndarray,
    bar_day_index: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_offsets: np.ndarray,
    row_symbol_ids: np.ndarray,
    ret: np.ndarray,
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
    """Select holdings and calculate ``sum(weight * return)`` per bar."""
    n_bars = len(bar_offsets) - 1
    held = np.zeros(n_symbols, dtype=np.bool_)
    entry_day = np.full(n_symbols, -1, dtype=np.int32)
    current_row = np.full(n_symbols, -1, dtype=np.int64)
    last_row = np.full(n_symbols, -1, dtype=np.int64)
    rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    in_size_pool = np.zeros(n_symbols, dtype=np.bool_)
    held_symbols = np.full(port_size, -1, dtype=np.int32)
    previous_weights = np.zeros(n_symbols, dtype=np.float64)
    held_count = 0

    closes = np.zeros(n_bars, dtype=np.int32)
    turnovers = np.zeros(n_bars, dtype=np.float64)
    returns = np.zeros(n_bars, dtype=np.float64)
    held_counts = np.zeros(n_bars, dtype=np.int32)
    capacity = n_bars * port_size
    rec_bar_idx = np.empty(capacity, dtype=np.int32)
    rec_symbol_id = np.empty(capacity, dtype=np.int32)
    rec_source_row = np.empty(capacity, dtype=np.int64)
    rec_weight = np.empty(capacity, dtype=np.float64)
    rec_pred_rank = np.empty(capacity, dtype=np.float64)
    rec_size_rank = np.empty(capacity, dtype=np.float64)
    rec_ret = np.empty(capacity, dtype=np.float64)
    rec_tradable = np.empty(capacity, dtype=np.bool_)
    rec_count = 0
    sign = -1.0 if is_short else 1.0

    for bar_idx in range(n_bars):
        current_row[:] = -1
        rank_by_symbol[:] = 0
        in_size_pool[:] = False
        for row_idx in range(bar_offsets[bar_idx], bar_offsets[bar_idx + 1]):
            symbol_id = row_symbol_ids[row_idx]
            current_row[symbol_id] = row_idx
            last_row[symbol_id] = row_idx

        signal_idx = bar_idx - 1 if trade_on_next_bar else bar_idx
        has_signal = signal_idx >= 0
        if has_signal:
            rank = 0
            for pos in range(sorted_offsets[signal_idx], sorted_offsets[signal_idx + 1]):
                row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[row_idx]
                if size_rank[row_idx] <= size_cut:
                    in_size_pool[symbol_id] = True
                    if can_open_base[row_idx] or held[symbol_id]:
                        rank += 1
                        rank_by_symbol[symbol_id] = rank

            compact_count = 0
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                close = False
                # T+1 applies to calendar days, so an intraday open cannot be sold today.
                if row_idx != -1 and can_close[row_idx] and entry_day[symbol_id] < bar_day_index[bar_idx]:
                    if rank_by_symbol[symbol_id] == 0 or rank_by_symbol[symbol_id] > thresh_out:
                        close = True
                    if close_on_size_drop and not in_size_pool[symbol_id]:
                        close = True
                if close:
                    held[symbol_id] = False
                    entry_day[symbol_id] = -1
                    closes[bar_idx] += 1
                else:
                    held_symbols[compact_count] = symbol_id
                    compact_count += 1
            held_count = compact_count

            for pos in range(sorted_offsets[signal_idx], sorted_offsets[signal_idx + 1]):
                if held_count == port_size:
                    break
                signal_row = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row]
                exec_row = current_row[symbol_id]
                rank = rank_by_symbol[symbol_id]
                if strict_first_bar_top_n and bar_idx == (1 if trade_on_next_bar else 0) and rank > port_size:
                    break
                if rank == 0:
                    continue
                if in_size_pool[symbol_id] and not held[symbol_id] and exec_row != -1 and can_open[exec_row]:
                    held[symbol_id] = True
                    entry_day[symbol_id] = bar_day_index[bar_idx]
                    held_symbols[held_count] = symbol_id
                    held_count += 1

        target_weight = sign / held_count if held_count else 0.0
        turnover_sum = 0.0
        # Existing holdings receive the new equal target weight; names that vanished
        # from the bar remain held but are assigned zero P&L until a tradable close.
        for symbol_id in range(n_symbols):
            new_weight = target_weight if held[symbol_id] else 0.0
            turnover_sum += abs(new_weight - previous_weights[symbol_id])
            previous_weights[symbol_id] = new_weight
        turnovers[bar_idx] = 0.5 * turnover_sum if has_signal else 0.0

        bar_return = 0.0
        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]
            if row_idx != -1:
                bar_return += target_weight * ret[row_idx]
        returns[bar_idx] = bar_return
        held_counts[bar_idx] = held_count

        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]
            rec_bar_idx[rec_count] = bar_idx
            rec_symbol_id[rec_count] = symbol_id
            rec_weight[rec_count] = target_weight
            if row_idx == -1:
                rec_source_row[rec_count] = last_row[symbol_id]
                rec_pred_rank[rec_count] = np.nan
                rec_size_rank[rec_count] = np.nan
                rec_ret[rec_count] = 0.0
                rec_tradable[rec_count] = False
            else:
                rec_source_row[rec_count] = row_idx
                rec_pred_rank[rec_count] = float(rank_by_symbol[symbol_id]) if rank_by_symbol[symbol_id] > 0 else np.nan
                rec_size_rank[rec_count] = float(size_rank[row_idx])
                rec_ret[rec_count] = ret[row_idx]
                rec_tradable[rec_count] = tradable[row_idx]
            rec_count += 1

    return (
        rec_bar_idx[:rec_count], rec_symbol_id[:rec_count], rec_source_row[:rec_count], rec_weight[:rec_count],
        rec_pred_rank[:rec_count], rec_size_rank[:rec_count], rec_ret[:rec_count], rec_tradable[:rec_count],
        closes, turnovers, returns, held_counts,
    )


def _materialize_positions(
    pool: BacktestDataset, rec_bar_idx: np.ndarray, rec_symbol_id: np.ndarray, rec_source_row: np.ndarray,
    rec_weight: np.ndarray, rec_pred_rank: np.ndarray, rec_size_rank: np.ndarray, rec_ret: np.ndarray, rec_tradable: np.ndarray,
) -> pd.DataFrame:
    columns = ["turnover", "log_size", "size_rank", "industry", "index", "ret", "weight_actual", "is_limit_up", "is_limit_down", "listed_Satisfied", "is_ST", "normal_days", "tradable", "can_open", "pred", "pred_rank", "vwap_ret"]
    if not rec_symbol_id.size:
        return pd.DataFrame(columns=["date", "symbol", *columns]).set_index(["date", "symbol"])
    records = pl.DataFrame({
        "row_idx": rec_source_row, "date": pd.to_datetime(pool.bars[rec_bar_idx]), "symbol": pool.symbols[rec_symbol_id],
        "ret": rec_ret, "weight_actual": rec_weight, "size_rank_recorded": rec_size_rank,
        "tradable_recorded": rec_tradable, "pred_rank": rec_pred_rank,
    })
    base = pool.pool_frame.select(["row_idx", "turnover", "log_size", "size_rank", "industry", "index", "is_limit_up", "is_limit_down", "listed_Satisfied", "is_ST", "normal_days", "can_open", "pred"])
    result = (
        records.join(base, on="row_idx", how="left")
        .with_columns(
            pl.coalesce("size_rank", "size_rank_recorded").alias("size_rank"),
            pl.col("turnover").fill_null(0.0), pl.col("can_open").fill_null(False),
            pl.col("is_limit_up").fill_null(False), pl.col("is_limit_down").fill_null(False),
            pl.col("tradable_recorded").alias("tradable"), pl.col("ret").alias("vwap_ret"),
        )
        .select(["date", "symbol", *columns])
        .to_pandas()
    )
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "symbol"]).sort_index()


def generate_portfolio(
    pool: BacktestDataset, port_size: int, thresh_out_buffer: int = 200, size_cut: int = 9999,
    close_on_size_drop: bool = True, trade_on_next_bar: bool = True, strict_first_bar_top_n: bool = False,
    is_short: bool = True, plot_heatmap: bool = True, output_dir: str = "output/", debug_mode: bool = False,
    debug_symbol: str | None = None, debug_datetime: str | None = None,
) -> PortfolioResult:
    """Generate an equal-weight portfolio for any configured bar frequency."""
    sorted_rows, sorted_offsets = _build_bar_orders(pool, ascending=is_short)
    bars = pd.to_datetime(pool.bars)
    bar_day_index = np.arange(len(bars), dtype=np.int32) if (bars == bars.normalize()).all() else pd.factorize(bars.normalize())[0].astype(np.int32)
    can_open_exec = (pool.can_trade_sell if is_short else pool.can_trade_buy) & pool.can_open_base
    can_close_exec = pool.can_trade_buy if is_short else pool.can_trade_sell
    output = _simulate_portfolio_core(
        pool.bar_offsets, bar_day_index, sorted_rows, sorted_offsets, pool.row_symbol_ids, pool.ret, pool.size_rank,
        pool.tradable, pool.can_open_base, can_open_exec, can_close_exec, len(pool.symbols), port_size,
        port_size + thresh_out_buffer, size_cut, close_on_size_drop, strict_first_bar_top_n, trade_on_next_bar, is_short,
    )
    rec_bar_idx, rec_symbol_id, rec_source_row, rec_weight, rec_pred_rank, rec_size_rank, rec_ret, rec_tradable, closes, turnovers, returns, held_counts = output
    positions = _materialize_positions(pool, rec_bar_idx, rec_symbol_id, rec_source_row, rec_weight, rec_pred_rank, rec_size_rank, rec_ret, rec_tradable)
    if debug_mode:
        target = (pd.Timestamp(debug_datetime), debug_symbol) if debug_symbol and debug_datetime else None
        held = bool(target and target in positions.index)
        print(f"[DEBUG] target={target}, held={held}, frequency bars={len(bars)}")
    if plot_heatmap:
        plot_position_heatmap(positions, port_num=port_size, is_short=is_short, output_path=output_dir)
    return PortfolioResult(
        positions=positions,
        close_counts=pd.DataFrame({"n_closed": closes}, index=bars),
        portfolio_returns=pd.Series(returns, index=bars, name="portfolio_return"),
        cost_turnover=pd.Series(turnovers, index=bars, name="cost_turnover"),
        turnover=pd.Series(turnovers, index=bars, name="turnover"),
        held_counts=pd.Series(held_counts, index=bars, name="held_count"),
    )
