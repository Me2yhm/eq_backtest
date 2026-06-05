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


def _resolve_debug_target_indices(
    pool: BacktestDataset15Min,
    debug_symbol: str | None,
    debug_datetime: str | None,
) -> tuple[int, int]:
    if not debug_symbol or not debug_datetime:
        return -1, -1

    target_ts = pd.Timestamp(debug_datetime)
    symbol_map = {sym: idx for idx, sym in enumerate(pool.symbols.tolist())}
    target_symbol_id = symbol_map.get(debug_symbol, -1)
    if target_symbol_id < 0:
        return -1, -1

    bars = pd.to_datetime(pool.bars)
    matches = np.where(bars == target_ts)[0]
    if len(matches) == 0:
        return target_symbol_id, -1
    return target_symbol_id, int(matches[0])


def _debug_reason_text(code: int) -> str:
    mapping = {
        101: "目标bar没有可用信号（例如next-bar模式下首bar）",
        102: "目标股票在信号bar没有有效pred（未进入排序）",
        103: "目标股票在信号bar不满足size_cut或can_open_base",
        104: "目标股票在执行bar缺失行情行（exec_row不存在）",
        105: "目标股票在执行bar不满足can_open（由15分钟tradable与日频开仓条件共同决定）",
        106: "目标股票当时已持仓（不是新开仓）",
        107: "开仓前组合已满仓（port_size已占满）",
        108: "strict_first_bar_top_n限制导致首执行bar未开仓",
    109: "信号bar无任何有效预测值（排序列表为空），本bar不执行调仓",
        200: "目标股票成功开仓",
        301: "目标股票在该bar调仓前并未持仓（无平仓动作）",
        302: "目标股票受T+0限制，当天新开后不可卖出",
        303: "目标股票在执行bar缺失行情行（无法平仓判断）",
        304: "目标股票在执行bar不满足can_close（不可卖出）",
        305: "目标股票在可平仓池排名超过阈值（rank > thresh_out）触发平仓",
        306: "目标股票跌出size池（close_on_size_drop）触发平仓",
        307: "目标股票同时满足排名阈值与size池跌出，触发平仓",
        308: "目标股票满足可卖条件，但未触发任何平仓规则（继续持有）",
        309: "信号bar无任何有效预测值（排序列表为空），本bar不执行平仓/开仓",
    }
    return mapping.get(code, "未命中明确原因（请结合下方状态位判断）")


@njit(cache=True)
def _simulate_portfolio_core_15min(
    bar_offsets: np.ndarray,
    bar_day_index: np.ndarray,
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
    debug_target_symbol_id: int,
    debug_target_bar_idx: int,
) -> tuple:
    n_bars = len(bar_offsets) - 1
    first_execution_bar = 1 if trade_on_next_bar else 0

    held = np.zeros(n_symbols, dtype=np.bool_)
    current_row = np.full(n_symbols, -1, dtype=np.int64)
    rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    close_rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    signal_in_size_pool = np.zeros(n_symbols, dtype=np.bool_)
    signal_can_open_pool = np.zeros(n_symbols, dtype=np.bool_)
    held_symbols = np.full(port_size, -1, dtype=np.int32)
    entry_day_by_symbol = np.full(n_symbols, -1, dtype=np.int32)

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

    debug_reason = -1
    debug_signal_bar_idx = -1
    debug_has_signal_row = 0
    debug_in_size_pool = 0
    debug_can_open_base = 0
    debug_signal_rank = 0
    debug_exec_row_present = 0
    debug_can_open_exec = 0
    debug_close_reason = -1
    debug_target_held_before_close = 0
    debug_can_sell_today = 0
    debug_can_close_exec = 0
    debug_close_rank = 0

    portfolio_sign = -1.0 if is_short else 1.0

    for bar_idx in range(n_bars):
        current_row[:] = -1
        rank_by_symbol[:] = 0
        close_rank_by_symbol[:] = 0
        signal_in_size_pool[:] = False
        signal_can_open_pool[:] = False

        bar_start = bar_offsets[bar_idx]
        bar_end = bar_offsets[bar_idx + 1]
        for row_idx in range(bar_start, bar_end):
            symbol_id = row_symbol_ids[row_idx]
            current_row[symbol_id] = row_idx

        signal_bar_idx = bar_idx - 1 if trade_on_next_bar else bar_idx
        has_signal = signal_bar_idx >= 0
        debug_this_bar = (debug_target_symbol_id >= 0) and (debug_target_bar_idx == bar_idx)
        target_seen_in_sorted = False
        target_seen_in_open_loop = False
        order_start = 0
        order_end = 0
        has_ranked_signal = False

        if debug_this_bar:
            debug_signal_bar_idx = signal_bar_idx
            if not has_signal:
                debug_reason = 101
            target_exec_row_idx = current_row[debug_target_symbol_id]
            if target_exec_row_idx != -1:
                debug_exec_row_present = 1
                if can_open[target_exec_row_idx]:
                    debug_can_open_exec = 1

        if has_signal:
            rank = 0
            close_rank = 0
            order_start = sorted_offsets[signal_bar_idx]
            order_end = sorted_offsets[signal_bar_idx + 1]
            has_ranked_signal = order_end > order_start
            if not has_ranked_signal and debug_this_bar:
                if debug_reason < 0:
                    debug_reason = 109
                if debug_close_reason < 0:
                    debug_close_reason = 309

            if not has_ranked_signal:
                close_rank = 0
                rank = 0
            else:
                for pos in range(order_start, order_end):
                    signal_row_idx = sorted_rows[pos]
                    symbol_id = row_symbol_ids[signal_row_idx]
                    exec_row_idx = current_row[symbol_id]
                    if debug_this_bar and symbol_id == debug_target_symbol_id:
                        target_seen_in_sorted = True
                        debug_has_signal_row = 1

                    if exec_row_idx != -1 and can_close[exec_row_idx]:
                        close_rank += 1
                        close_rank_by_symbol[symbol_id] = close_rank

                    if size_rank[signal_row_idx] < size_cut:
                        signal_in_size_pool[symbol_id] = True
                        if debug_this_bar and symbol_id == debug_target_symbol_id:
                            debug_in_size_pool = 1
                        if can_open_base[signal_row_idx]:
                            signal_can_open_pool[symbol_id] = True
                            if debug_this_bar and symbol_id == debug_target_symbol_id:
                                debug_can_open_base = 1
                    if signal_can_open_pool[symbol_id] or held[symbol_id]:
                        rank += 1
                        rank_by_symbol[symbol_id] = rank
                        if debug_this_bar and symbol_id == debug_target_symbol_id:
                            debug_signal_rank = rank

        n_closed = 0
        if has_signal and has_ranked_signal:
            new_count = 0
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                should_close = False
                can_sell_today = entry_day_by_symbol[symbol_id] >= 0 and entry_day_by_symbol[symbol_id] < bar_day_index[bar_idx]
                is_target_close = debug_this_bar and (symbol_id == debug_target_symbol_id)

                if is_target_close:
                    debug_target_held_before_close = 1
                    if can_sell_today:
                        debug_can_sell_today = 1
                    if row_idx != -1 and can_close[row_idx]:
                        debug_can_close_exec = 1
                    if close_rank_by_symbol[symbol_id] > 0:
                        debug_close_rank = close_rank_by_symbol[symbol_id]

                if can_sell_today and row_idx != -1 and can_close[row_idx]:
                    rank_rule = close_rank_by_symbol[symbol_id] == 0 or close_rank_by_symbol[symbol_id] > thresh_out
                    size_rule = close_on_size_drop and not signal_in_size_pool[symbol_id]

                    if rank_rule:
                        should_close = True
                    if size_rule:
                        should_close = True

                    if is_target_close and should_close and debug_close_reason < 0:
                        if rank_rule and size_rule:
                            debug_close_reason = 307
                        elif rank_rule:
                            debug_close_reason = 305
                        elif size_rule:
                            debug_close_reason = 306
                    elif is_target_close and not should_close and debug_close_reason < 0:
                        debug_close_reason = 308
                elif is_target_close and debug_close_reason < 0:
                    if not can_sell_today:
                        debug_close_reason = 302
                    elif row_idx == -1:
                        debug_close_reason = 303
                    else:
                        debug_close_reason = 304

                if should_close:
                    held[symbol_id] = False
                    n_closed += 1
                else:
                    held_symbols[new_count] = symbol_id
                    new_count += 1
            held_count = new_count

        if debug_this_bar and debug_close_reason < 0 and has_signal and has_ranked_signal:
            if debug_target_held_before_close == 0:
                debug_close_reason = 301

        if has_signal and has_ranked_signal:
            for pos in range(order_start, order_end):
                if held_count == port_size:
                    if debug_this_bar and debug_reason < 0:
                        debug_reason = 107
                    break

                signal_row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row_idx]
                exec_row_idx = current_row[symbol_id]
                is_target = debug_this_bar and (symbol_id == debug_target_symbol_id)
                if is_target:
                    target_seen_in_open_loop = True

                if bar_idx == first_execution_bar and strict_first_bar_top_n and rank_by_symbol[symbol_id] > port_size:
                    if is_target and debug_reason < 0:
                        debug_reason = 108
                    continue

                if (
                    signal_can_open_pool[symbol_id]
                    and exec_row_idx != -1
                    and can_open[exec_row_idx]
                    and not held[symbol_id]
                ):
                    held[symbol_id] = True
                    entry_day_by_symbol[symbol_id] = bar_day_index[bar_idx]
                    held_symbols[held_count] = symbol_id
                    held_count += 1
                    if is_target:
                        debug_reason = 200
                elif is_target and debug_reason < 0:
                    if not signal_can_open_pool[symbol_id]:
                        debug_reason = 103
                    elif exec_row_idx == -1:
                        debug_reason = 104
                    elif not can_open[exec_row_idx]:
                        debug_reason = 105
                    elif held[symbol_id]:
                        debug_reason = 106

        if debug_this_bar and debug_reason < 0:
            if not has_signal:
                debug_reason = 101
            elif not target_seen_in_sorted:
                debug_reason = 102
            elif not target_seen_in_open_loop and held_count == port_size:
                debug_reason = 107
            elif debug_has_signal_row == 1 and (debug_in_size_pool == 0 or debug_can_open_base == 0):
                debug_reason = 103

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
        debug_reason,
        debug_signal_bar_idx,
        debug_has_signal_row,
        debug_in_size_pool,
        debug_can_open_base,
        debug_signal_rank,
        debug_exec_row_present,
        debug_can_open_exec,
        debug_close_reason,
        debug_target_held_before_close,
        debug_can_sell_today,
        debug_can_close_exec,
        debug_close_rank,
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
    debug_mode: bool = False,
    debug_symbol: str | None = None,
    debug_datetime: str | None = None,
) -> PortfolioResult15Min:
    """Simulate a 15-minute equal-weight portfolio with daily constraints."""
    thresh_out = port_size + thresh_out_buffer
    sorted_rows, sorted_offsets = _build_bar_orders(pool, ascending=is_short)

    can_open_exec = (pool.can_trade_sell if is_short else pool.can_trade_buy) & pool.can_open_base
    can_close_exec = pool.can_trade_buy if is_short else pool.can_trade_sell
    bars = pd.to_datetime(pool.bars)
    bar_day_index = pd.factorize(bars.normalize())[0].astype(np.int32)
    debug_target_symbol_id, debug_target_bar_idx = (
        _resolve_debug_target_indices(pool, debug_symbol, debug_datetime) if debug_mode else (-1, -1)
    )

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
        debug_reason,
        debug_signal_bar_idx,
        debug_has_signal_row,
        debug_in_size_pool,
        debug_can_open_base,
        debug_signal_rank,
        debug_exec_row_present,
        debug_can_open_exec,
        debug_close_reason,
        debug_target_held_before_close,
        debug_can_sell_today,
        debug_can_close_exec,
        debug_close_rank,
    ) = _simulate_portfolio_core_15min(
        bar_offsets=pool.bar_offsets,
        bar_day_index=bar_day_index,
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
        debug_target_symbol_id=debug_target_symbol_id,
        debug_target_bar_idx=debug_target_bar_idx,
    )

    if debug_mode:
        print("\n[DEBUG-15MIN] ----------")
        print(f"target_symbol={debug_symbol}, target_datetime={debug_datetime}")
        print(f"resolved_symbol_id={debug_target_symbol_id}, resolved_bar_idx={debug_target_bar_idx}")
        print(f"signal_bar_idx={debug_signal_bar_idx}, signal_rank={debug_signal_rank}")
        print(
            "flags:",
            f"has_signal_row={bool(debug_has_signal_row)}",
            f"in_size_pool={bool(debug_in_size_pool)}",
            f"can_open_base={bool(debug_can_open_base)}",
            f"exec_row_present={bool(debug_exec_row_present)}",
            f"can_open_exec={bool(debug_can_open_exec)}",
            f"held_before_close={bool(debug_target_held_before_close)}",
            f"can_sell_today={bool(debug_can_sell_today)}",
            f"can_close_exec={bool(debug_can_close_exec)}",
            f"close_rank={int(debug_close_rank)}",
        )
        print(f"decision_code={debug_reason}, reason={_debug_reason_text(int(debug_reason))}")
        print(f"close_decision_code={debug_close_reason}, close_reason={_debug_reason_text(int(debug_close_reason))}")
        print("[DEBUG-15MIN] ----------\n")

    positions = _materialize_positions_15min(
        pool=pool,
        rec_bar_idx=rec_bar_idx,
        rec_source_row=rec_source_row,
        rec_pred_rank=rec_pred_rank,
        rec_size_rank=rec_size_rank,
        rec_vwap_ret=rec_vwap_ret,
        rec_tradable=rec_tradable,
    )

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
