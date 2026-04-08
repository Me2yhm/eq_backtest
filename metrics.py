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
    ann_ret = (1 + cum_ret) ** (1 / years) - 1

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
    entry_day: dict = {}  # symbol -> day index when it was opened
    prev_symbols: set = set()
    avg_open, med_open, max_open, avg_closed = {}, {}, {}, {}

    dates = sorted(positions.index.get_level_values("date").unique())

    for day_n, date in enumerate(dates):
        today = set(positions.loc[date].index)
        exited = prev_symbols - today

        # Holding days for positions closed today
        closed_days = [day_n - entry_day[s] for s in exited if s in entry_day]
        if closed_days:
            avg_closed[date] = np.mean(closed_days)
        for s in exited:
            entry_day.pop(s, None)

        # Record entry day for newly opened positions
        for s in today - prev_symbols:
            entry_day[s] = day_n

        # Holding days for all currently open positions
        open_days = [day_n - entry_day[s] + 1 for s in today if s in entry_day]
        if open_days:
            avg_open[date] = np.mean(open_days)
            med_open[date] = np.median(open_days)
            max_open[date] = max(open_days)

        prev_symbols = today

    return {
        "avg_open": pd.Series(avg_open),
        "median_open": pd.Series(med_open),
        "max_open": pd.Series(max_open),
        "avg_closed": pd.Series(avg_closed),
    }
