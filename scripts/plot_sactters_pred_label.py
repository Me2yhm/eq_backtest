"""
Draw prediction-label scatter plots and report mean IC / mean RankIC.

Cases
-----
a) all non-limit-up/down samples
b) top 1000 by cross-sectional prediction rank per timestamp
c) bottom 500 by cross-sectional prediction rank per timestamp

Notes
-----
- Limit filtering follows project logic: compare 15m `vwap15` with daily
  `limit_up_price` / `limit_down_price` on (symbol, date).
- IC/RankIC are computed cross-sectionally per timestamp using
    prediction vs label, then averaged over timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction-label scatter and IC stats")
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=Path("/ext/trq/predictions.parquet"),
        help="Path to predictions parquet (needs trade_date, stock_code, prediction, label)",
    )
    parser.add_argument(
        "--bar15-path",
        type=Path,
        default=Path("/home/danny/backtest_verify/eq-backtest/data_15min/15min_bar_full_with_vwap.parquet"),
        help="Path to 15min bar parquet (needs symbol, datetime, vwap15, vwap_ret)",
    )
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=Path("/home/danny/backtest_verify/eq-backtest/data/daily_with_limit.pqt"),
        help="Path to daily parquet (needs symbol, date, limit_up_price, limit_down_price)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scatter_pred_label_ic.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Max scatter points per panel for readability/performance",
    )
    return parser.parse_args()


def _assert_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def load_merged_frame(pred_path: Path, bar15_path: Path, daily_path: Path) -> pd.DataFrame:
    _assert_exists(pred_path, "predictions parquet")
    _assert_exists(bar15_path, "15min bar parquet")
    _assert_exists(daily_path, "daily parquet")

    preds = pd.read_parquet(pred_path, columns=["trade_date", "stock_code", "prediction", "label"]).rename(
        columns={"trade_date": "datetime", "stock_code": "symbol"}
    )
    preds["datetime"] = pd.to_datetime(preds["datetime"])

    bars = pd.read_parquet(bar15_path, columns=["symbol", "datetime", "vwap15"])
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["date"] = bars["datetime"].dt.normalize()

    daily = pd.read_parquet(daily_path, columns=["symbol", "date", "limit_up_price", "limit_down_price"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    bars = bars.merge(daily, on=["symbol", "date"], how="left")
    bars["is_limit_up"] = bars["vwap15"] >= bars["limit_up_price"]
    bars["is_limit_down"] = bars["vwap15"] <= bars["limit_down_price"]
    bars["is_limit"] = (bars["is_limit_up"] | bars["is_limit_down"]).fillna(False)

    bars = bars.loc[:, ["symbol", "datetime", "is_limit"]]

    merged = preds.merge(bars, on=["symbol", "datetime"], how="inner")
    merged = merged.dropna(subset=["prediction", "label"])
    return merged


def add_cross_section_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_rank_desc"] = out.groupby("datetime")["prediction"].rank(method="first", ascending=False)
    out["cs_size"] = out.groupby("datetime")["prediction"].transform("size")
    return out


def calc_ic_stats(df: pd.DataFrame) -> tuple[float, float, int]:
    grouped = df.groupby("datetime", sort=True)

    ic_series = grouped.apply(lambda x: x["prediction"].corr(x["label"]))
    rank_ic_series = grouped.apply(lambda x: x["prediction"].corr(x["label"], method="spearman"))

    ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
    rank_ic_series = rank_ic_series.replace([np.inf, -np.inf], np.nan).dropna()

    ic_mean = float(ic_series.mean()) if not ic_series.empty else float("nan")
    rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.empty else float("nan")
    n_ts = int(grouped.ngroups)
    return ic_mean, rank_ic_mean, n_ts


def maybe_sample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42)


def plot_four_cases(df_non_limit_ranked: pd.DataFrame, df_all: pd.DataFrame, output_path: Path, max_points: int) -> None:
    all_non_limit_case = df_non_limit_ranked
    top_case = df_non_limit_ranked[df_non_limit_ranked["pred_rank_desc"] <= 1000]
    bottom_case = df_non_limit_ranked[
        df_non_limit_ranked["pred_rank_desc"] > (df_non_limit_ranked["cs_size"] - 500).clip(lower=0)
    ]
    all_with_limit_case = df_all

    cases = [
        ("All (non-limit)", all_non_limit_case),
        ("Top1000 by pred rank", top_case),
        ("Bottom500 by pred rank", bottom_case),
        ("All (with limit)", all_with_limit_case),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, case_df) in zip(axes, cases):
        ic_mean, rank_ic_mean, n_ts = calc_ic_stats(case_df)
        draw_df = maybe_sample(case_df, max_points=max_points)

        ax.scatter(
            draw_df["prediction"],
            draw_df["label"],
            s=4,
            alpha=0.15,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("prediction")
        ax.set_ylabel("label")
        ax.grid(alpha=0.25)

        info = f"IC mean: {ic_mean:.6f}\nRankIC mean: {rank_ic_mean:.6f}\nTimestamps: {n_ts}\nPoints: {len(case_df):,}"
        ax.text(
            0.02,
            0.98,
            info,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    merged_all = load_merged_frame(args.pred_path, args.bar15_path, args.daily_path)
    merged_non_limit = merged_all.loc[~merged_all["is_limit"]].copy()
    ranked_non_limit = add_cross_section_rank(merged_non_limit)
    plot_four_cases(ranked_non_limit, merged_all, args.output, args.max_points)

    print(f"Saved figure to: {args.output}")
    print(f"Total merged rows (with limit): {len(merged_all):,}")
    print(f"Total merged rows (non-limit): {len(ranked_non_limit):,}")


if __name__ == "__main__":
    main()
