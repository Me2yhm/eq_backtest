"""
Visualization for backtest results.

Functions
---------
plot_holding_periods(stats, port_num, ...)
plot_position_heatmap(positions, ...)
plot_portfolio_results(cumrets, port_sizes, close_counts, ...)
plot_metrics_table(metrics_df, ...)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd

# ── Shared helper ──────────────────────────────────────────────────────────────


def _ensure_plot_dir(output_path: str, is_short: bool) -> str:
    subdir = "plots_short" if is_short else "plots_long"
    path = os.path.join(output_path, subdir)
    os.makedirs(path, exist_ok=True)
    return path


# ── Holding period ─────────────────────────────────────────────────────────────


def plot_holding_periods(
    stats: dict,
    port_num: int,
    is_short: bool = False,
    output_path: str = "output/",
) -> None:
    """
    Line chart of average, median (open positions) and average (closed positions)
    holding days over time.
    """
    avg_open = stats["avg_open"]
    med_open = stats["median_open"]
    avg_closed = stats["avg_closed"]

    if avg_open.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_open, linewidth=1.5, label="Avg (open)", color="steelblue")
    ax.plot(med_open, linewidth=1.5, label="Median (open)", color="seagreen", alpha=0.8)
    if not avg_closed.empty:
        ax.plot(
            avg_closed,
            linewidth=1.0,
            label="Avg (closed)",
            color="darkorange",
            alpha=0.7,
        )

    ax.axhline(
        avg_open.mean(),
        color="steelblue",
        linestyle="--",
        alpha=0.5,
        label=f"Overall avg: {avg_open.mean():.1f} d",
    )
    ax.set(title=f"Holding Days – Portfolio {port_num}", xlabel="Date", ylabel="Days")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    pdir = _ensure_plot_dir(output_path, is_short)
    fig.savefig(
        os.path.join(pdir, f"holding_days_p{port_num}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ── Position heatmap ───────────────────────────────────────────────────────────


def plot_position_heatmap(
    positions: pd.DataFrame,
    size_rank_bins: int = 20,
    port_num: int | None = None,
    is_short: bool = False,
    output_path: str = "output/",
) -> None:
    """Heatmap of position count across size-rank bins per day."""
    if positions.empty:
        return

    df = positions.reset_index()
    max_rank = df["size_rank"].max()
    bins = np.linspace(0, max_rank, size_rank_bins + 1)
    df["bin"] = pd.cut(df["size_rank"], bins=bins, labels=False, include_lowest=True)

    heatmap = df.groupby(["date", "bin"]).size().unstack(fill_value=0)
    dates_list = list(heatmap.index)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap.T, aspect="auto", cmap="viridis", interpolation="nearest")

    # x-axis: sampled date labels
    tick_idxs = np.linspace(0, len(dates_list) - 1, min(20, len(dates_list)), dtype=int)
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels([dates_list[i].strftime("%Y-%m-%d") for i in tick_idxs], rotation=45, ha="right")

    # y-axis: size-rank ranges
    bin_labels = [f"{int(bins[i]):.0f}–{int(bins[i + 1]):.0f}" for i in range(size_rank_bins)]
    ax.set_yticks(range(size_rank_bins))
    ax.set_yticklabels(bin_labels)

    # Vertical lines at year boundaries
    cur_year = dates_list[0].year
    for i, d in enumerate(dates_list):
        if d.year != cur_year:
            ax.axvline(i - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
            cur_year = d.year

    plt.colorbar(im, ax=ax, label="Number of Positions")
    ax.set(
        title="Position Distribution by Size Rank",
        xlabel="Date",
        ylabel="Size Rank Bin",
    )
    plt.tight_layout()

    fname = f"heatmap_p{port_num}.png" if port_num is not None else "heatmap.png"
    pdir = _ensure_plot_dir(output_path, is_short)
    fig.savefig(os.path.join(pdir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Portfolio returns overview ────────────────────────────────────────────────


def plot_portfolio_results(
    cumrets: pd.DataFrame,
    port_sizes: pd.DataFrame,
    close_counts: pd.DataFrame,
    is_short: bool = False,
    output_path: str = "output/",
) -> None:
    """
    Two-panel figure:
      Top    – cumulative excess returns and drawdowns.
      Bottom – daily portfolio size and number of closures.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Panel 1: cumulative returns + drawdowns ───────────────────────────────
    cumrets.plot(ax=ax1, linewidth=2, grid=True, title="Portfolio Excess Return (Cumulative)")
    ax1.set_ylabel("Cumulative Return")

    ax1_dd = ax1.twinx()
    for i, col in enumerate(cumrets.columns):
        wealth = 1 + cumrets[col]
        drawdown = wealth - wealth.expanding().max()
        ax1_dd.plot(drawdown, color=ax1.get_lines()[i].get_color(), linewidth=1, alpha=0.5)
    ax1_dd.set_ylabel("Drawdown")
    ax1.legend(loc="upper left")

    # ── Panel 2: portfolio size + closes ─────────────────────────────────────
    n_ports = len(port_sizes.columns)
    colors = [plt.cm.get_cmap("tab10")(i / max(n_ports - 1, 1)) for i in range(n_ports)]  # type: ignore[attr-defined]
    for i, col in enumerate(port_sizes.columns):
        port_sizes[col].plot(ax=ax2, color=colors[i], label=col, linewidth=2)
        if not close_counts.empty and col in close_counts.columns:
            close_counts[col].plot(
                ax=ax2,
                color=colors[i],
                linestyle="--",
                alpha=0.5,
                label=f"{col} closes",
            )

    ax2.set(title="Portfolio Size & Daily Closes", ylabel="Stocks", xlabel="Date")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    plt.tight_layout()
    pdir = _ensure_plot_dir(output_path, is_short)
    fig.savefig(os.path.join(pdir, "portfolio_results.png"), dpi=300, bbox_inches="tight")
    plt.show()


# ── Metrics table image ───────────────────────────────────────────────────────


def plot_metrics_table(
    metrics_df: pd.DataFrame,
    pool_size: int,
    bm_name: str,
    is_short: bool = False,
    output_path: str = "output/",
) -> None:
    """Save the metrics DataFrame as a formatted table image."""
    fig, ax = plt.subplots(figsize=(12, len(metrics_df) * 0.8 + 1))
    ax.axis("off")

    display = metrics_df.copy().apply(lambda col: col.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x))
    tbl = ax.table(
        cellText=display.values,
        rowLabels=display.index.astype(str).tolist(),
        colLabels=display.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)

    header_bg = "#40466e"
    for j in range(len(display.columns)):
        tbl[(0, j)].set_facecolor(header_bg)
        tbl[(0, j)].set_text_props(weight="bold", color="white")
    for i in range(len(display)):
        tbl[(i + 1, -1)].set_facecolor(header_bg)
        tbl[(i + 1, -1)].set_text_props(weight="bold", color="white")

    direction = "Short" if is_short else "Long"
    ax.set_title(
        f"Portfolio Metrics ({direction} | Pool: {pool_size} | BM: {bm_name})",
        fontsize=12,
        weight="bold",
        pad=20,
    )

    pdir = _ensure_plot_dir(output_path, is_short)
    fig.savefig(
        os.path.join(pdir, f"metrics_{pool_size}_{bm_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
