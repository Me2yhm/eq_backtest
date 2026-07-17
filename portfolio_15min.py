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
    target_weights: pd.DataFrame | None = None


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
        110: "目标股票在买入候选中，但轮到它时现金已耗尽（或不足以成交）",
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
    close_prev: np.ndarray,
    close_curr: np.ndarray,
    vwap15: np.ndarray,
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
    cost_per_turnover: float,
    record_target: bool,
    debug_target_symbol_id: int,
    debug_target_bar_idx: int,
) -> tuple:
    n_bars = len(bar_offsets) - 1
    eps = 1e-12

    # target: ideal/rank-band layer. held: actual layer presence mirror.
    target = np.zeros(n_symbols, dtype=np.bool_)
    held = np.zeros(n_symbols, dtype=np.bool_)
    current_weights = np.zeros(n_symbols, dtype=np.float64)
    frozen_weights = np.zeros(n_symbols, dtype=np.float64)
    cash_weight = 1.0

    # share-based 状态 (对齐 backtest.py L783-784)
    portfolio_value = 1e8  # 初始资金
    current_shares = np.zeros(n_symbols, dtype=np.float64)  # 持仓股数

    current_row = np.full(n_symbols, -1, dtype=np.int64)
    rank_by_symbol = np.zeros(n_symbols, dtype=np.int32)
    signal_in_size_pool = np.zeros(n_symbols, dtype=np.bool_)
    # Actual holdings can temporarily exceed port_size because T+1 frozen names
    # may coexist with newly opened target names.
    held_symbols = np.full(n_symbols, -1, dtype=np.int32)
    target_symbols = np.full(port_size, -1, dtype=np.int32)
    sell_candidates = np.full(n_symbols, -1, dtype=np.int32)
    sell_keys = np.empty(n_symbols, dtype=np.float64)
    buy_candidates = np.full(n_symbols, -1, dtype=np.int32)
    buy_keys = np.empty(n_symbols, dtype=np.float64)

    curr_weights = np.zeros(n_symbols, dtype=np.float64)
    prev_symbols = np.full(n_symbols, -1, dtype=np.int32)
    curr_symbols = np.full(n_symbols, -1, dtype=np.int32)
    touched = np.zeros(n_symbols, dtype=np.bool_)
    touched_symbols = np.full(n_symbols, -1, dtype=np.int32)

    held_count = 0
    target_count = 0
    prev_count = 0
    close_counts = np.zeros(n_bars, dtype=np.int32)
    bar_turnover = np.zeros(n_bars, dtype=np.float64)
    bar_returns = np.zeros(n_bars, dtype=np.float64)
    held_counts = np.zeros(n_bars, dtype=np.int32)

    # Exact recording capacity for the current run. This can be larger than the
    # previous n_bars * port_size bound because progressive actual holdings can
    # exceed ideal count intraday.
    record_capacity = n_bars * n_symbols
    rec_bar_idx = np.empty(record_capacity, dtype=np.int32)
    rec_symbol_id = np.empty(record_capacity, dtype=np.int32)
    rec_source_row = np.empty(record_capacity, dtype=np.int64)
    rec_weight = np.empty(record_capacity, dtype=np.float64)
    rec_pred_rank = np.empty(record_capacity, dtype=np.float64)
    rec_size_rank = np.empty(record_capacity, dtype=np.float64)
    rec_vwap_ret = np.empty(record_capacity, dtype=np.float64)
    rec_tradable = np.empty(record_capacity, dtype=np.bool_)
    rec_count = 0

    target_record_capacity = n_bars * port_size if record_target else 0
    target_rec_bar_idx = np.empty(target_record_capacity, dtype=np.int32)
    target_rec_symbol_id = np.empty(target_record_capacity, dtype=np.int32)
    target_rec_weight = np.empty(target_record_capacity, dtype=np.float64)
    target_rec_count = 0

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
    debug_cash_after_sell = 0.0
    debug_buy_order_rank = 0
    debug_buy_cash_before = 0.0
    debug_buy_deficit = 0.0
    debug_buy_executed_weight = 0.0
    debug_frozen_weight = 0.0

    portfolio_sign = -1.0 if is_short else 1.0
    nominal_weight = 1.0 / port_size

    for bar_idx in range(n_bars):
        # T+1 release: newly bought weights are frozen for the same day.
        # Names bought today become sellable only on the next calendar day.
        if bar_idx == 0 or bar_day_index[bar_idx] != bar_day_index[bar_idx - 1]:
            for i in range(held_count):
                symbol_id = held_symbols[i]
                frozen_weights[symbol_id] = 0.0

        current_row[:] = -1
        rank_by_symbol[:] = 0
        signal_in_size_pool[:] = False

        bar_start = bar_offsets[bar_idx]
        bar_end = bar_offsets[bar_idx + 1]
        for row_idx in range(bar_start, bar_end):
            symbol_id = row_symbol_ids[row_idx]
            current_row[symbol_id] = row_idx

        # ── Share-based Step 1 (对齐 backtest.py L786-800) ──
        prev_capital = portfolio_value

        # Step 1: 上期持仓 close→vwap 收益
        prev_pnl = 0.0
        if bar_idx > 0:
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                if row_idx == -1:
                    continue
                sh = current_shares[symbol_id]
                if abs(sh) <= eps:
                    continue
                cp = close_prev[row_idx]
                v = vwap15[row_idx]
                if cp > 0 and v > 0:
                    prev_pnl += sh * (v - cp)
            portfolio_value += prev_pnl

        # Step 2: 调仓前盯市权重 (对齐 compute_external_metrics.py L345-352)
        # current_weights_np = shares × vwap / portfolio_value (portfolio_value 已含 prev_pnl)
        marktomarket_weights = np.zeros(n_symbols, dtype=np.float64)
        if abs(portfolio_value) > eps:
            for i in range(held_count):
                symbol_id = held_symbols[i]
                row_idx = current_row[symbol_id]
                if row_idx == -1:
                    continue
                sh = current_shares[symbol_id]
                if abs(sh) <= eps:
                    continue
                v = vwap15[row_idx]
                if v > 0 and np.isfinite(v):
                    w = sh * v / portfolio_value
                    if abs(w) > eps:
                        marktomarket_weights[symbol_id] = w

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
            order_start = sorted_offsets[signal_bar_idx]
            order_end = sorted_offsets[signal_bar_idx + 1]
            has_ranked_signal = order_end > order_start
            if not has_ranked_signal and debug_this_bar:
                if debug_reason < 0:
                    debug_reason = 109
                if debug_close_reason < 0:
                    debug_close_reason = 309

            if has_ranked_signal:
                for pos in range(order_start, order_end):
                    signal_row_idx = sorted_rows[pos]
                    symbol_id = row_symbol_ids[signal_row_idx]
                    if debug_this_bar and symbol_id == debug_target_symbol_id:
                        target_seen_in_sorted = True
                        debug_has_signal_row = 1

                    in_size_pool = size_rank[signal_row_idx] <= size_cut
                    if in_size_pool:
                        signal_in_size_pool[symbol_id] = True
                        if debug_this_bar and symbol_id == debug_target_symbol_id:
                            debug_can_open_base = 1
                            debug_in_size_pool = 1

                    # External eligible rank uses ideal entry holdings, not the
                    # actual execution state.
                    if signal_in_size_pool[symbol_id] or target[symbol_id]:
                        rank += 1
                        rank_by_symbol[symbol_id] = rank
                        if debug_this_bar and symbol_id == debug_target_symbol_id:
                            debug_signal_rank = rank

        n_closed = 0
        if has_signal and has_ranked_signal:
            # 1) Ideal target layer: keep previous target names while they remain
            # inside the exit buffer, then only consider current signal names
            # inside the top-N rank frontier. If some names inside that frontier
            # are not executable, we do not backfill with lower-ranked names.
            # Ideal target should ignore intraday execution blocks such as
            # zero-turnover or one-sided price-limit states. It only requires
            # the name to be present on the bar and pass the daily base
            # universe checks.
            new_target_count = 0
            for i in range(target_count):
                symbol_id = target_symbols[i]
                exec_row_idx = current_row[symbol_id]
                target_valid_listed = exec_row_idx != -1 and can_open_base[exec_row_idx]
                target_rank_rule = rank_by_symbol[symbol_id] == 0 or rank_by_symbol[symbol_id] > thresh_out
                target_size_rule = close_on_size_drop and not signal_in_size_pool[symbol_id]
                if target_rank_rule or target_size_rule or not target_valid_listed:
                    target[symbol_id] = False
                else:
                    target_symbols[new_target_count] = symbol_id
                    new_target_count += 1
            target_count = new_target_count

            for pos in range(order_start, order_end):
                if target_count == port_size:
                    break
                signal_row_idx = sorted_rows[pos]
                symbol_id = row_symbol_ids[signal_row_idx]
                exec_row_idx = current_row[symbol_id]
                signal_rank = rank_by_symbol[symbol_id]
                if debug_this_bar and symbol_id == debug_target_symbol_id:
                    target_seen_in_open_loop = True

                if signal_rank == 0:
                    continue
                if signal_rank > port_size:
                    break

                target_can_enter = exec_row_idx != -1 and can_open_base[exec_row_idx]

                if signal_in_size_pool[symbol_id] and target_can_enter and not target[symbol_id]:
                    target[symbol_id] = True
                    target_symbols[target_count] = symbol_id
                    target_count += 1

            target_weight = 0.0
            if target_count > 0:
                target_weight = nominal_weight

            if debug_this_bar:
                symbol_id = debug_target_symbol_id
                row_idx = current_row[symbol_id]
                if current_weights[symbol_id] > eps:
                    debug_target_held_before_close = 1
                if row_idx != -1 and can_close[row_idx]:
                    debug_can_close_exec = 1
                if rank_by_symbol[symbol_id] > 0:
                    debug_close_rank = rank_by_symbol[symbol_id]
                sellable_dbg = current_weights[symbol_id] - frozen_weights[symbol_id]
                if sellable_dbg > eps:
                    debug_can_sell_today = 1

            # 2) Actual sell layer: sell current > target by external priority:
            # target_weight asc, reduction desc, stock_idx asc.
            sell_count = 0
            for i in range(held_count):
                symbol_id = held_symbols[i]
                cur_w = current_weights[symbol_id]
                if cur_w <= eps:
                    continue
                tgt_w = target_weight if target[symbol_id] else 0.0
                reduction = cur_w - tgt_w
                if reduction > eps:
                    target_group = 1.0 if target[symbol_id] else 0.0
                    sell_candidates[sell_count] = symbol_id
                    sell_keys[sell_count] = target_group * 10.0 - reduction + symbol_id * 1e-12
                    sell_count += 1

            if sell_count > 0:
                sell_order = np.argsort(sell_keys[:sell_count])
                remaining_sell_budget = 1.0
                for oi in range(sell_count):
                    if remaining_sell_budget <= eps:
                        break
                    symbol_id = sell_candidates[sell_order[oi]]
                    cur_w = current_weights[symbol_id]
                    tgt_w = target_weight if target[symbol_id] else 0.0
                    reduction = cur_w - tgt_w
                    if reduction <= eps:
                        continue

                    row_idx = current_row[symbol_id]
                    if row_idx == -1 or not can_close[row_idx]:
                        if debug_this_bar and symbol_id == debug_target_symbol_id and debug_close_reason < 0:
                            debug_close_reason = 303 if row_idx == -1 else 304
                        continue

                    sellable = cur_w - frozen_weights[symbol_id]
                    if sellable < 0.0:
                        sellable = 0.0
                    if sellable <= eps:
                        if debug_this_bar and symbol_id == debug_target_symbol_id and debug_close_reason < 0:
                            debug_close_reason = 302
                        continue

                    sell_w = reduction
                    if sell_w > sellable:
                        sell_w = sellable
                    if sell_w > remaining_sell_budget:
                        sell_w = remaining_sell_budget
                    if sell_w <= eps:
                        continue

                    current_weights[symbol_id] = cur_w - sell_w
                    cash_weight += sell_w
                    remaining_sell_budget -= sell_w
                    if current_weights[symbol_id] <= eps:
                        current_weights[symbol_id] = 0.0
                        frozen_weights[symbol_id] = 0.0
                        held[symbol_id] = False
                        n_closed += 1
                    if debug_this_bar and symbol_id == debug_target_symbol_id and debug_close_reason < 0:
                        debug_close_reason = 305

            # Compact actual holdings after sells so newly opened names can append safely.
            new_held_count = 0
            for i in range(held_count):
                symbol_id = held_symbols[i]
                if current_weights[symbol_id] > eps:
                    held_symbols[new_held_count] = symbol_id
                    new_held_count += 1
                else:
                    held[symbol_id] = False
            held_count = new_held_count

            # 3) Actual buy layer: buy target deficits by prediction desc.
            buy_count = 0
            for i in range(target_count):
                symbol_id = target_symbols[i]
                row_idx = current_row[symbol_id]
                if row_idx == -1 or not can_open[row_idx]:
                    continue
                deficit = target_weight - current_weights[symbol_id]
                if deficit > eps:
                    buy_candidates[buy_count] = symbol_id
                    buy_keys[buy_count] = -pred[row_idx] + symbol_id * 1e-12
                    buy_count += 1

            if debug_this_bar:
                debug_cash_after_sell = cash_weight

            if buy_count > 0:
                buy_order = np.argsort(buy_keys[:buy_count])
                for oi in range(buy_count):
                    symbol_id = buy_candidates[buy_order[oi]]
                    deficit = target_weight - current_weights[symbol_id]
                    if debug_this_bar and symbol_id == debug_target_symbol_id:
                        debug_buy_order_rank = oi + 1
                        debug_buy_cash_before = cash_weight
                        debug_buy_deficit = deficit
                    if cash_weight <= eps:
                        if debug_this_bar:
                            continue
                        break
                    if deficit <= eps:
                        continue
                    buy_w = deficit
                    if buy_w > cash_weight:
                        buy_w = cash_weight
                    if buy_w <= eps:
                        continue

                    was_held = current_weights[symbol_id] > eps
                    current_weights[symbol_id] += buy_w
                    cash_weight -= buy_w
                    frozen_weights[symbol_id] += buy_w
                    if debug_this_bar and symbol_id == debug_target_symbol_id:
                        debug_buy_executed_weight = buy_w
                    if not was_held:
                        held[symbol_id] = True
                        held_symbols[held_count] = symbol_id
                        held_count += 1
                    if debug_this_bar and symbol_id == debug_target_symbol_id:
                        debug_reason = 200

            if debug_this_bar and debug_close_reason < 0:
                if current_weights[debug_target_symbol_id] <= eps:
                    debug_close_reason = 301
                elif target[debug_target_symbol_id]:
                    debug_close_reason = 308
                elif frozen_weights[debug_target_symbol_id] >= current_weights[debug_target_symbol_id] - eps:
                    debug_close_reason = 302
                else:
                    debug_close_reason = 304
                debug_frozen_weight = frozen_weights[debug_target_symbol_id]

        if debug_this_bar and debug_reason < 0:
            if not has_signal:
                debug_reason = 101
            elif not target_seen_in_sorted:
                debug_reason = 102
            elif not target_seen_in_open_loop and target_count == port_size:
                debug_reason = 107
            elif debug_has_signal_row == 1 and (debug_in_size_pool == 0 or debug_can_open_base == 0):
                debug_reason = 103
            elif target[debug_target_symbol_id] and current_weights[debug_target_symbol_id] > eps:
                debug_reason = 106
            elif debug_buy_order_rank > 0 and debug_buy_executed_weight <= eps:
                debug_reason = 110

        close_counts[bar_idx] = n_closed

        # ── Share-based Step 3-5 + turnover (对齐 backtest.py L802-863) ──

        # 构建 curr_weights / curr_symbols 快照（target selection 后）
        curr_count = 0
        for i in range(held_count):
            symbol_id = held_symbols[i]
            actual_w = current_weights[symbol_id]
            if actual_w <= eps:
                continue
            signed_w = portfolio_sign * actual_w
            curr_weights[symbol_id] = signed_w
            curr_symbols[curr_count] = symbol_id
            curr_count += 1

        # 计算 turnover（方案 B: 0.5 × Σ|effective_target - current_weights_np|）
        # 对齐 Reference: turnover = 0.5 × Σ|curr_weights - marktomarket_weights|
        # 关键对齐：Reference 仅在 has_signal 的 bar 计算 turnover，
        # 非信号 bar 的 turnover 为 0（价格漂移不计入 turnover）。
        # curr_weights = 调仓后实际权重 (effective_target)
        # marktomarket_weights = 调仓前盯市权重 (current_weights_np, 含股价漂移)
        # 并集遍历 prev_symbols ∪ curr_symbols，确保覆盖：
        #   - 仍持有但权重变化的股票
        #   - 已清仓的股票 (curr_weights=0, marktomarket_weights>0)
        #   - 新开仓的股票 (curr_weights>0, marktomarket_weights=0)
        if has_signal and has_ranked_signal:
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
                turnover += abs(curr_weights[symbol_id] - marktomarket_weights[symbol_id])

            turnover *= 0.5
        else:
            turnover = 0.0

        # Step 3: 成本扣除 + weights → shares 同步 (对齐 Reference L802-826)
        # Reference 仅对 tradable_mask 股票更新 shares，非 tradable 股票保持旧 shares。
        # 生产引擎对齐此行为：只有 vwap 有效且非涨跌停的股票才同步 shares，
        # 不可交易股票（涨跌停/零换手）保持旧 shares，避免强制重设导致的
        # marktomarket_weights 偏差累积。
        transaction_cost = 0.0
        if has_signal and has_ranked_signal:
            transaction_cost = portfolio_value * 2.0 * turnover * cost_per_turnover
            portfolio_value -= transaction_cost
            # 同步 weights → shares（仅可交易股票）
            if abs(portfolio_value) > eps:
                for i in range(held_count):
                    symbol_id = held_symbols[i]
                    row_idx = current_row[symbol_id]
                    if row_idx == -1:
                        continue
                    # 对齐 Reference tradable_mask: vwap 有效且可交易
                    if not tradable[row_idx]:
                        continue
                    v = vwap15[row_idx]
                    if v > 0:
                        current_shares[symbol_id] = portfolio_sign * current_weights[symbol_id] * portfolio_value / v
            # 清仓股票 shares 归零
            for i in range(prev_count):
                symbol_id = prev_symbols[i]
                if current_weights[symbol_id] <= eps:
                    current_shares[symbol_id] = 0.0

        # Step 4: 本期持仓 vwap→close 收益 (L834-838)
        current_pnl = 0.0
        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]
            if row_idx == -1:
                continue
            sh = current_shares[symbol_id]
            if abs(sh) <= eps:
                continue
            v = vwap15[row_idx]
            cc = close_curr[row_idx]
            if v > 0 and cc > 0:
                current_pnl += sh * (cc - v)
        portfolio_value += current_pnl

        # Step 5: 记录收益率 (L840-863, 分母 = prev_capital)
        if abs(prev_capital) > eps:
            bar_ret = (prev_pnl + current_pnl - transaction_cost) / prev_capital
        else:
            bar_ret = 0.0
        bar_turnover[bar_idx] = turnover
        bar_returns[bar_idx] = bar_ret

        # post-bar bookkeeping
        # 仅维护 prev_symbols/prev_count（并集遍历 turnover 需要），不再维护 prev_weights
        # （方案 B 的 turnover 基准是实时盯市权重 marktomarket_weights，不依赖名义权重复制）
        next_prev_count = 0
        for i in range(curr_count):
            symbol_id = curr_symbols[i]
            prev_symbols[next_prev_count] = symbol_id
            next_prev_count += 1
            curr_weights[symbol_id] = 0.0

        prev_count = next_prev_count
        held_counts[bar_idx] = held_count
        current_target_weight = 0.0
        if target_count > 0:
            current_target_weight = nominal_weight

        for i in range(held_count):
            symbol_id = held_symbols[i]
            row_idx = current_row[symbol_id]
            rec_bar_idx[rec_count] = bar_idx
            rec_symbol_id[rec_count] = symbol_id
            rec_weight[rec_count] = current_weights[symbol_id]
            if row_idx == -1:
                rec_source_row[rec_count] = -1
                rec_size_rank[rec_count] = np.nan
                rec_vwap_ret[rec_count] = 0.0
                rec_tradable[rec_count] = False
                rec_pred_rank[rec_count] = np.nan
            else:
                rec_source_row[rec_count] = row_idx
                rec_size_rank[rec_count] = float(size_rank[row_idx])
                rec_vwap_ret[rec_count] = vwap_ret[row_idx]
                rec_tradable[rec_count] = tradable[row_idx]
                if tradable[row_idx] and rank_by_symbol[symbol_id] > 0:
                    rec_pred_rank[rec_count] = float(rank_by_symbol[symbol_id])
                else:
                    rec_pred_rank[rec_count] = np.nan
            rec_count += 1

        if record_target:
            for i in range(target_count):
                symbol_id = target_symbols[i]
                target_rec_bar_idx[target_rec_count] = bar_idx
                target_rec_symbol_id[target_rec_count] = symbol_id
                target_rec_weight[target_rec_count] = current_target_weight
                target_rec_count += 1

    return (
        rec_bar_idx[:rec_count],
        rec_symbol_id[:rec_count],
        rec_source_row[:rec_count],
        rec_weight[:rec_count],
        rec_pred_rank[:rec_count],
        rec_size_rank[:rec_count],
        rec_vwap_ret[:rec_count],
        rec_tradable[:rec_count],
        target_rec_bar_idx[:target_rec_count],
        target_rec_symbol_id[:target_rec_count],
        target_rec_weight[:target_rec_count],
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
        debug_cash_after_sell,
        debug_buy_order_rank,
        debug_buy_cash_before,
        debug_buy_deficit,
        debug_buy_executed_weight,
        debug_frozen_weight,
    )


def _materialize_positions_15min(
    pool: BacktestDataset15Min,
    rec_bar_idx: np.ndarray,
    rec_symbol_id: np.ndarray,
    rec_source_row: np.ndarray,
    rec_weight: np.ndarray,
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
        "weight_actual",
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

    snapshot_bars = pd.to_datetime(pool.bars[rec_bar_idx])
    symbols = pool.symbols[rec_symbol_id]
    records = pl.DataFrame({
        "snapshot_bar": snapshot_bars,
        "symbol": symbols,
        "ret": rec_vwap_ret,
        "weight_actual": rec_weight,
        "size_rank_recorded": rec_size_rank,
        "tradable": rec_tradable,
        "pred_rank": rec_pred_rank,
        "vwap_ret": rec_vwap_ret,
    }).with_columns(pl.col("snapshot_bar").dt.date().alias("snapshot_date"))

    daily_columns = [
        "date",
        "symbol",
        "log_size",
        "size_rank",
        "industry",
        "index",
        "listed_Satisfied",
        "is_ST",
        "normal_days",
    ]
    current_bar_columns = [
        "datetime",
        "symbol",
        "turnover",
        "is_limit_up",
        "is_limit_down",
        "can_open",
        "pred",
    ]

    positions = (
        records
        .join(
            pool.daily_snapshot_frame.select(daily_columns).rename({
                "date": "snapshot_date",
                "size_rank": "size_rank_daily",
            }),
            on=["snapshot_date", "symbol"],
            how="left",
        )
        .join(
            pool.pool_frame.select(current_bar_columns).rename({"datetime": "snapshot_bar"}),
            on=["snapshot_bar", "symbol"],
            how="left",
        )
        .rename({"snapshot_bar": "date"})
        .with_columns([
            pl.coalesce("size_rank_daily", "size_rank_recorded").alias("size_rank"),
            pl.col("turnover").fill_null(0.0),
            pl.col("is_limit_up").fill_null(False),
            pl.col("is_limit_down").fill_null(False),
            pl.col("can_open").fill_null(False),
        ])
        .select([
            "date",
            "symbol",
            "turnover",
            "log_size",
            "size_rank",
            "industry",
            "index",
            "ret",
            "weight_actual",
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
        ])
    )

    result = positions.to_pandas()
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "symbol"]).sort_index()


def _materialize_weight_snapshots_15min(
    pool: BacktestDataset15Min,
    rec_bar_idx: np.ndarray,
    rec_symbol_id: np.ndarray,
    rec_weight: np.ndarray,
    weight_column: str,
) -> pd.DataFrame:
    if rec_bar_idx.size == 0:
        empty = pd.DataFrame(columns=["date", "symbol", weight_column])
        return empty.set_index(["date", "symbol"])

    result = pd.DataFrame({
        "date": pd.to_datetime(pool.bars[rec_bar_idx]),
        "symbol": pool.symbols[rec_symbol_id],
        weight_column: rec_weight,
    })
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
    record_target_weights: bool = False,
    debug_mode: bool = False,
    debug_symbol: str | None = None,
    debug_datetime: str | None = None,
    cost_per_turnover: float = 0.00045,
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
        rec_symbol_id,
        rec_source_row,
        rec_weight,
        rec_pred_rank,
        rec_size_rank,
        rec_vwap_ret,
        rec_tradable,
        target_rec_bar_idx,
        target_rec_symbol_id,
        target_rec_weight,
        close_counts_arr,
        turnover_arr,
        bar_returns_arr,
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
        debug_cash_after_sell,
        debug_buy_order_rank,
        debug_buy_cash_before,
        debug_buy_deficit,
        debug_buy_executed_weight,
        debug_frozen_weight,
    ) = _simulate_portfolio_core_15min(
        bar_offsets=pool.bar_offsets,
        bar_day_index=bar_day_index,
        sorted_rows=sorted_rows,
        sorted_offsets=sorted_offsets,
        row_symbol_ids=pool.row_symbol_ids,
        vwap_ret=pool.vwap_ret,
        close_prev=pool.close_prev,
        close_curr=pool.close_curr,
        vwap15=pool.vwap15,
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
        cost_per_turnover=cost_per_turnover,
        record_target=record_target_weights,
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
        print(
            "buy_debug:",
            f"cash_after_sell={debug_cash_after_sell:.12g}",
            f"buy_order_rank={int(debug_buy_order_rank)}",
            f"cash_before_candidate={debug_buy_cash_before:.12g}",
            f"buy_deficit={debug_buy_deficit:.12g}",
            f"buy_executed_weight={debug_buy_executed_weight:.12g}",
            f"frozen_weight={debug_frozen_weight:.12g}",
        )
        print(f"decision_code={debug_reason}, reason={_debug_reason_text(int(debug_reason))}")
        print(f"close_decision_code={debug_close_reason}, close_reason={_debug_reason_text(int(debug_close_reason))}")
        print("[DEBUG-15MIN] ----------\n")

    positions = _materialize_positions_15min(
        pool=pool,
        rec_bar_idx=rec_bar_idx,
        rec_symbol_id=rec_symbol_id,
        rec_source_row=rec_source_row,
        rec_weight=rec_weight,
        rec_pred_rank=rec_pred_rank,
        rec_size_rank=rec_size_rank,
        rec_vwap_ret=rec_vwap_ret,
        rec_tradable=rec_tradable,
    )

    close_counts = pd.DataFrame({"n_closed": close_counts_arr}, index=bars)
    portfolio_returns = pd.Series(bar_returns_arr, index=bars, name="portfolio_return")
    turnover = pd.Series(turnover_arr, index=bars, name="turnover")
    held_counts = pd.Series(held_counts_arr, index=bars, name="held_count")
    target_weights = None
    if record_target_weights:
        target_weights = _materialize_weight_snapshots_15min(
            pool=pool,
            rec_bar_idx=target_rec_bar_idx,
            rec_symbol_id=target_rec_symbol_id,
            rec_weight=target_rec_weight,
            weight_column="weight_target",
        )

    if plot_heatmap:
        plot_position_heatmap(positions, port_num=port_size, is_short=is_short, output_path=output_dir)

    return PortfolioResult15Min(
        positions=positions,
        close_counts=close_counts,
        portfolio_returns=portfolio_returns,
        cost_turnover=turnover.copy().rename("cost_turnover"),
        turnover=turnover,
        held_counts=held_counts,
        target_weights=target_weights,
    )


def generate_target_weights_15min(
    pool: BacktestDataset15Min,
    port_size: int,
    thresh_out_buffer: int = 200,
    size_cut: int = 9999,
    close_on_size_drop: bool = True,
    trade_on_next_bar: bool = True,
    strict_first_bar_top_n: bool = False,
    is_short: bool = True,
    cost_per_turnover: float = 0.00045,
) -> pd.DataFrame:
    """Simulate only to export the ideal target-layer equal-weight snapshots."""
    thresh_out = port_size + thresh_out_buffer
    sorted_rows, sorted_offsets = _build_bar_orders(pool, ascending=is_short)

    can_open_exec = (pool.can_trade_sell if is_short else pool.can_trade_buy) & pool.can_open_base
    can_close_exec = pool.can_trade_buy if is_short else pool.can_trade_sell
    bars = pd.to_datetime(pool.bars)
    bar_day_index = pd.factorize(bars.normalize())[0].astype(np.int32)

    (
        _rec_bar_idx,
        _rec_symbol_id,
        _rec_source_row,
        _rec_weight,
        _rec_pred_rank,
        _rec_size_rank,
        _rec_vwap_ret,
        _rec_tradable,
        target_rec_bar_idx,
        target_rec_symbol_id,
        target_rec_weight,
        _close_counts_arr,
        _turnover_arr,
        _bar_returns_arr,
        _held_counts_arr,
        _debug_reason,
        _debug_signal_bar_idx,
        _debug_has_signal_row,
        _debug_in_size_pool,
        _debug_can_open_base,
        _debug_signal_rank,
        _debug_exec_row_present,
        _debug_can_open_exec,
        _debug_close_reason,
        _debug_target_held_before_close,
        _debug_can_sell_today,
        _debug_can_close_exec,
        _debug_close_rank,
        _debug_cash_after_sell,
        _debug_buy_order_rank,
        _debug_buy_cash_before,
        _debug_buy_deficit,
        _debug_buy_executed_weight,
        _debug_frozen_weight,
    ) = _simulate_portfolio_core_15min(
        bar_offsets=pool.bar_offsets,
        bar_day_index=bar_day_index,
        sorted_rows=sorted_rows,
        sorted_offsets=sorted_offsets,
        row_symbol_ids=pool.row_symbol_ids,
        vwap_ret=pool.vwap_ret,
        close_prev=pool.close_prev,
        close_curr=pool.close_curr,
        vwap15=pool.vwap15,
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
        cost_per_turnover=cost_per_turnover,
        record_target=True,
        debug_target_symbol_id=-1,
        debug_target_bar_idx=-1,
    )

    return _materialize_weight_snapshots_15min(
        pool=pool,
        rec_bar_idx=target_rec_bar_idx,
        rec_symbol_id=target_rec_symbol_id,
        rec_weight=target_rec_weight,
        weight_column="weight_target",
    )
