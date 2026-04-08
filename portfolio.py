"""
Portfolio simulation engine.

Public API
----------
generate_portfolio(pool, port_size, ...)  ->  (positions DataFrame, close_counts DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting import plot_position_heatmap

# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_pred_rank(
    daily: pd.DataFrame,
    held_index,
    size_cut: int,
    ascending: bool,
) -> pd.Series:
    """
    Rank stocks within the eligible universe by their prediction score.
    Held stocks are always included even when they fall outside size_cut,
    so we can still decide whether to close them.
    """
    if held_index is None:
        in_scope = (daily["size_rank"] < size_cut) & daily["tradable"]
    else:
        in_scope = ((daily["size_rank"] < size_cut) | daily.index.isin(held_index)) & daily["tradable"]

    ranks = pd.Series(np.nan, index=daily.index)
    ranks[in_scope] = daily.loc[in_scope, "pred"].rank(ascending=ascending, method="first")
    return ranks


def _refresh_positions(pos: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Update held positions with today's market data.

    Stocks absent from today's data (halted, delisted, etc.) are force-held:
    their return is set to 0 and tradable flag to False.
    """
    available = pos.index.intersection(daily.index)
    if len(available):
        pos.loc[available] = daily.loc[available]

    missing = pos.index.difference(daily.index)
    if len(missing):
        pos.loc[missing, ["ret", "pred_rank", "size_rank"]] = [0.0, np.nan, np.nan]
        pos.loc[missing, "tradable"] = False

    return pos


def _should_close(
    pos: pd.DataFrame,
    thresh_out: int,
    size_cut: int,
    close_on_size_drop: bool,
) -> pd.Series:
    """
    Boolean Series: True for positions that should be closed.

    Close conditions (all require tradable == True):
      1. pred_rank  > thresh_out  (model signal degraded)
      2. pred_rank is NaN         (stock left the eligible pool)
      3. size_rank >= size_cut    (only when close_on_size_drop is True)
    """
    stale = (pos["pred_rank"] > thresh_out) | pos["pred_rank"].isna()
    if close_on_size_drop:
        stale |= pos["size_rank"] >= size_cut
    return stale & pos["tradable"]


def _select_new_entries(
    daily: pd.DataFrame,
    held_index,
    slots: int,
) -> pd.DataFrame:
    """Return the top `slots` candidates eligible to open today."""
    candidates = daily.loc[
        daily["pred_rank"].notna()  # ranked (→ tradable & in scope)
        & ~daily.index.isin(held_index)  # not already held
        & daily["can_open"]  # satisfies all open conditions
    ]
    return candidates.sort_values("pred_rank").head(slots)


# ── Public API ────────────────────────────────────────────────────────────────


def generate_portfolio(
    pool: pd.DataFrame,
    port_size: int,
    thresh_out_buffer: int = 200,
    size_cut: int = 9999,
    close_on_size_drop: bool = True,
    is_short: bool = True,
    plot_heatmap: bool = True,
    output_dir: str = "output/",
) -> tuple:
    """
    Simulate a daily-rebalanced equal-weight portfolio.

    Each trading day:
      1. Rank eligible stocks by prediction score.
      2. Close tradable positions whose rank exceeds thresh_out.
      3. Fill vacant slots from the top-ranked tradable candidates.
    Non-tradable positions are force-held until they become tradable again.

    Parameters
    ----------
    pool              : (date, symbol) indexed DataFrame from build_pool()
    port_size         : target number of holdings  (= thresh_in)
    thresh_out_buffer : exit trigger = port_size + thresh_out_buffer
    size_cut          : market-cap rank upper bound for the eligible universe
    close_on_size_drop: also close when a stock falls outside size_cut
    is_short          : rank ascending if True (short worst), descending if False (long best)
    plot_heatmap      : save a size-rank distribution heatmap to output_dir

    Returns
    -------
    positions    : DataFrame indexed by (date, symbol) – daily position snapshots
    close_counts : DataFrame indexed by date – number of closures per day
    """
    thresh_out = port_size + thresh_out_buffer

    pos: pd.DataFrame | None = None
    snapshots: list = []
    close_log: list = []

    for date in pool.index.get_level_values("date").unique():
        daily = pool.loc[date].copy()
        daily["pred_rank"] = _compute_pred_rank(
            daily,
            held_index=pos.index if pos is not None else None,
            size_cut=size_cut,
            ascending=is_short,
        )

        if pos is None:
            # ── First day: open the top-ranked, openable stocks ──────────────
            pos = daily.loc[(daily["pred_rank"] <= port_size) & daily["can_open"]].copy()
            n_closed = 0
        else:
            # ── Subsequent days ───────────────────────────────────────────────
            pos = _refresh_positions(pos, daily)

            # Step 1: close
            to_close = pos.index[_should_close(pos, thresh_out, size_cut, close_on_size_drop)]
            n_closed = len(to_close)
            pos = pos.drop(index=to_close)

            # Step 2: open
            slots = max(port_size - len(pos), 0)
            if slots:
                pos = pd.concat([pos, _select_new_entries(daily, pos.index, slots)])

        assert pos is not None
        snapshot = pos.copy().assign(date=pd.Timestamp(date)).reset_index().set_index(["date", "symbol"]).sort_index()
        snapshots.append(snapshot)
        close_log.append({"date": pd.Timestamp(date), "n_closed": n_closed})

    assert pos is not None, "Pool was empty – no trading days found."
    positions = pd.concat(snapshots).sort_index(level=[0, 1])
    close_counts = pd.DataFrame(close_log).set_index("date")

    if plot_heatmap:
        plot_position_heatmap(positions, port_num=port_size, is_short=is_short, output_path=output_dir)

    return positions, close_counts
