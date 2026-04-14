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


def portfolio_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Compute annualized performance metrics from a daily return series.

    Returns
    -------
    dict with keys:
        Ann. Return   – geometric annualized return
        Volatility    – annualized standard deviation
        Sharpe        – annualized Sharpe ratio
        Max Drawdown  – worst peak-to-trough decline
        Calmar        – annualized return / abs(max drawdown)
    """
    years = len(daily_returns) / ANN_DAYS
    cum_ret: float = (1 + daily_returns).prod()  # type: ignore[assignment]
    ann_ret = cum_ret ** (1 / years) - 1

    vol = daily_returns.std() * np.sqrt(ANN_DAYS)
    sharpe = (ann_ret - risk_free_rate) / vol if vol else np.nan

    wealth = (1 + daily_returns).cumprod()
    drawdown = wealth / wealth.expanding().max() - 1
    max_dd = drawdown.min()
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
    Compute daily holding-period statistics over the portfolio's lifetime.

    Returns
    -------
    dict of pd.Series indexed by date:
        avg_open    – average days held for currently open positions
        median_open – median days held for currently open positions
        max_open    – longest currently held position
        avg_closed  – average days held for positions closed on that day
    """
    idx = positions.index
    dates = idx.get_level_values("date").to_numpy()
    symbols = idx.get_level_values("symbol").to_numpy(dtype=object)

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
