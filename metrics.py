"""
Performance metrics for backtested portfolios.

Public API
----------
portfolio_metrics(daily_returns)  ->  dict of scalar metrics
holding_period_stats(positions)   ->  dict of time-series metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN_DAYS = 242  # trading days per year


def portfolio_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.0, compounding: bool = False) -> dict:
    """
    Compute annualized performance metrics from a daily return series.

    Parameters
    ----------
    daily_returns : daily return series
    risk_free_rate : annual risk-free rate
    compounding : False=算术年化(mean×242, 外部使用), True=几何年化(原逻辑)

    Returns
    -------
    dict with keys:
        Ann. Return   – annualized return
        Volatility    – annualized standard deviation
        Sharpe        – annualized Sharpe ratio
        Max Drawdown  – worst peak-to-trough decline (外部口径)
        Calmar        – annualized return / abs(max drawdown)
    """
    if compounding:
        # 几何年化（原逻辑）
        years = len(daily_returns) / ANN_DAYS
        cum_ret: float = (1 + daily_returns).prod()  # type: ignore[assignment]
        ann_ret = cum_ret ** (1 / years) - 1
    else:
        # 算术年化（对齐外部 metrics.py L41）
        ann_ret = daily_returns.mean() * ANN_DAYS

    vol = daily_returns.std() * np.sqrt(ANN_DAYS)
    sharpe = (ann_ret - risk_free_rate) / vol if vol else np.nan

    # MaxDD: 外部口径 (对齐 evaluation/metrics.py L88-99 + utils/utils.py L124-133)
    # wealth = 1.0 + cumsum() (算术累计, compounding="simple")
    # drawdown = wealth - expanding().max() (绝对回撤)
    # 取 quantile(0.001) (0.1% 分位数)
    # 旧口径: wealth=(1+x).cumprod(), drawdown=wealth/cummax-1, 取 min()
    wealth = 1.0 + daily_returns.cumsum()
    drawdown = wealth - wealth.expanding().max()
    max_dd = float(drawdown.quantile(0.001))

    calmar = ann_ret / abs(max_dd) if max_dd else np.nan

    return {
        "Ann. Return": ann_ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }


def holding_period_stats(positions: pd.DataFrame) -> dict:
    """
    Compute holding-period statistics in trading-session units.

    Intraday inputs are sampled at the final bar of each trading day. Positions
    must be ordered by timestamp and then symbol. A trading day without any
    position rows has no snapshot, so ``avg_closed`` cannot record exits on it.

    Returns
    -------
    dict of pd.Series indexed by date:
        avg_open    – average trading sessions held for currently open positions
        median_open – median trading sessions held for currently open positions
        max_open    – longest currently held position
        avg_closed  – average trading-session duration for recorded exits
    """
    idx = positions.index
    timestamps = pd.DatetimeIndex(pd.to_datetime(idx.get_level_values("date")))
    symbols = idx.get_level_values("symbol").to_numpy(dtype=object)
    if "sleeve" in positions.columns:
        sleeves = positions["sleeve"].to_numpy(dtype=object)
        symbols = np.char.add(np.char.add(sleeves.astype(str), ":"), symbols.astype(str)).astype(object)

    # Intraday positions are snapshots at every bar. Holding-period statistics
    # are reported in trading-session units, so use the final bar of each day
    # rather than treating each 5-minute snapshot as a separate day.
    session_dates = timestamps.normalize()
    if len(timestamps) and not timestamps.equals(session_dates):
        session_date_values = session_dates.to_numpy()
        session_ends = np.append(
            np.flatnonzero(session_date_values[:-1] != session_date_values[1:]) + 1,
            len(session_date_values),
        )
        timestamp_values = timestamps.to_numpy()
        # The chronological input lets searchsorted find the first row of each
        # day’s final timestamp, retaining all symbols in that final bar.
        final_bar_starts = np.searchsorted(timestamp_values, timestamp_values[session_ends - 1], side="left")
        final_bar_indices = np.concatenate([
            np.arange(start, end) for start, end in zip(final_bar_starts, session_ends, strict=True)
        ])
        session_dates = session_dates[final_bar_indices]
        symbols = symbols[final_bar_indices]
    dates = session_dates.to_numpy()

    if len(dates) == 0:
        empty = pd.Series(dtype=float)
        return {
            "avg_open": empty,
            "median_open": empty,
            "max_open": empty,
            "avg_closed": empty,
        }

    change_points = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(dates)]))
    unique_dates = dates[starts]

    entry_day: dict = {}
    prev_symbols: set = set()
    avg_open, med_open, max_open, avg_closed = {}, {}, {}, {}

    for day_n, (date, start, end) in enumerate(zip(unique_dates, starts, ends)):
        today = set(symbols[start:end])
        exited = prev_symbols - today

        closed_days = [day_n - entry_day[s] for s in exited if s in entry_day]
        if closed_days:
            avg_closed[pd.Timestamp(date)] = np.mean(closed_days)
        for s in exited:
            entry_day.pop(s, None)

        for s in today - prev_symbols:
            entry_day[s] = day_n

        open_days = [day_n - entry_day[s] + 1 for s in today if s in entry_day]
        if open_days:
            ts = pd.Timestamp(date)
            avg_open[ts] = np.mean(open_days)
            med_open[ts] = np.median(open_days)
            max_open[ts] = max(open_days)

        prev_symbols = today

    return {
        "avg_open": pd.Series(avg_open),
        "median_open": pd.Series(med_open),
        "max_open": pd.Series(max_open),
        "avg_closed": pd.Series(avg_closed),
    }
