"""
Data loading utilities.

Public API
----------
build_pool(...)  ->  (pool DataFrame, benchmark Series)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Benchmark ─────────────────────────────────────────────────────────────────


def load_benchmark(path: Path) -> pd.Series:
    """Load a CSV of daily benchmark returns into a Series."""
    result = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    return pd.Series(result)


# ── Predictions ───────────────────────────────────────────────────────────────


def _horizon_suffix(path: Path) -> str:
    """Extract the horizon tag after the last underscore (e.g. '3d' from 'model_3d.parquet')."""
    return path.stem.rsplit("_", 1)[-1] if "_" in path.stem else ""


def load_predictions(preds_dir: Path, horizons: list, start: str) -> pd.Series:
    """
    Load parquet prediction files matching the given horizon suffixes and average them.

    Each parquet file must be a wide DataFrame (index=date, columns=symbol).
    Returns a Series indexed by (date, symbol).
    """
    files = [f for f in preds_dir.glob("*.parquet") if _horizon_suffix(f) in horizons]
    if not files:
        raise FileNotFoundError(f"No parquet files matching horizons {horizons} found in '{preds_dir}'")
    print(f"Predictions: {len(files)} file(s) loaded – {[f.name for f in files]}")

    stacked = []
    for f in files:
        df = pd.read_parquet(f).truncate(before=start)
        df.index.name = "date"
        s = df.stack()
        s.index.set_names(["date", "symbol"], inplace=True)
        stacked.append(s)

    preds = pd.concat(stacked, axis=1).mean(axis=1)
    preds.name = "pred"
    return preds


# ── Market data ───────────────────────────────────────────────────────────────


def load_market_data(
    path: Path,
    start: str,
    universe: list | None,
    allow_st_open: bool,
) -> pd.DataFrame:
    """
    Load daily market data parquet and attach tradability flags.

    Columns added
    -------------
    tradable  : has volume AND open price is not at a limit
    can_open  : tradable + ≥10 consecutive normal days + optionally not ST + in universe
    """
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index(["date", "symbol"]).sort_index().truncate(before=start)

    data["tradable"] = (data["turnover"] > 0) & ~data["is_limit_up"].astype(bool) & ~data["is_limit_down"].astype(bool)

    can_open = data["tradable"] & (data["normal_days"] >= 10)
    if not allow_st_open:
        can_open &= ~data["is_ST"].astype(bool)
    if universe is not None:
        can_open &= data["index"].isin(universe)
    data["can_open"] = can_open

    return data


# ── Combined entry point ───────────────────────────────────────────────────────


def build_pool(
    data_path: Path,
    preds_dir: Path,
    bm_path: Path,
    horizons: list,
    start: str,
    universe: list | None,
    allow_st_open: bool,
) -> tuple:
    """
    Assemble the full pool and benchmark return series.

    Returns
    -------
    pool   : DataFrame indexed by (date, symbol) – market data joined with predictions
    bm_ret : Series of daily benchmark returns
    """
    bm_ret = load_benchmark(bm_path)
    preds = load_predictions(preds_dir, horizons, start)
    market = load_market_data(data_path, start, universe, allow_st_open)
    pool = market.join(preds, how="left").sort_index()
    return pool, bm_ret


if __name__ == "__main__":
    import config as cfg

    bm_ret = load_benchmark(cfg.BM_PATH)
    preds = load_predictions(cfg.PREDS_DIR, cfg.HORIZONS, cfg.START)
    print(bm_ret.head())
    print(preds.head())
