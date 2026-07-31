"""Backtest configuration.

Loads overrides from ``config.yml`` (git-ignored) and falls back to defaults
defined below.  All keys are optional — anything missing keeps its default.

Usage::

    import config as cfg
    print(cfg.FREQUENCY)
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

# ═══════════════════════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    # ── Time range ────────────────────────────────────────────────────────────
    "start": "2019-01-02",
    "end": "2026-01-01",
    # ── Frequency ─────────────────────────────────────────────────────────────
    "frequency": "5min",
    "freq_config": {
        "daily": {
            "market_data": "/ext/eq_data/daily_with_limit_prevcap.pqt",
            "preds_dir": "data/preds_size",
            "horizons": ["3d", "5d", "10d"],
            "market_columns": {"execution_vwap": None, "bar_close": None},
        },
        "15min": {
            "market_data": "/ext/eq_data/15min_bar_full_left_close.parquet",
            "preds_dir": "/ext/trq",
            "horizons": ["predictions"],
            "market_columns": {"execution_vwap": "vwap15", "bar_close": "close"},
        },
        "5min": {
            "market_data": "/ext/eq_data/5min_bar_full_left_close.parquet",
            "preds_dir": "/tmp/eq_preds/output_mse/output_bs16/predictions",
            "horizons": [""],
            "market_columns": {"execution_vwap": "vwap5", "bar_close": "close"},
        },
    },
    "bm_path": "/ext/eq_data/ret_csi_1000.csv",
    "bm_name": "csi_1000",
    "pool_cache_dir": "data/.cache",
    "use_pool_cache": True,
    # ── Benchmark ─────────────────────────────────────────────────────────────
    "use_external_benchmark": True,
    "external_nav_path": "/ext/trq/nav.parquet",
    # ── Universe and strategy ─────────────────────────────────────────────────
    "universe": None,
    "is_short": False,
    "allow_st_open": False,
    "pool_size": 4400,
    "port_sizes": [800],
    "thresh_out_buffer": 600,
    "trade_on_next_bar": False,
    "strict_first_bar_top_n": False,
    "close_on_size_drop": False,
    # ── Aggregation and costs ─────────────────────────────────────────────────
    "agg_mode": "simple",
    "compounding": False,
    "cost_per_turnover": 0.00045,
    "portfolio_initial_value": 1e8,
    "exclude_period": ("2024-01-01", "2024-03-31"),
    # ── Debug ─────────────────────────────────────────────────────────────────
    "debug": False,
    "debug_symbol": "300169.XSHE",
    "debug_datetime": "2019-01-03 09:46:00",
    # ── Eligibility and output ────────────────────────────────────────────────
    "nosuspend_days": 10,
}

# Top-level keys whose string value should be converted to ``Path``.
_PATH_KEYS: frozenset[str] = frozenset({"bm_path", "pool_cache_dir", "external_nav_path"})

# Keys *inside* each ``freq_config`` sub-table that are paths.
_FREQ_PATH_KEYS: frozenset[str] = frozenset({"market_data", "preds_dir"})


# ═══════════════════════════════════════════════════════════════════════════════
#  Load & merge
# ═══════════════════════════════════════════════════════════════════════════════

_CONFIG_PATH = Path(__file__).with_suffix(".yml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _convert_paths(cfg: dict) -> dict:
    """Convert string values to ``Path`` for keys listed in ``_PATH_KEYS``."""
    for key in _PATH_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = Path(cfg[key])
    for freq_cfg in cfg.get("freq_config", {}).values():
        for pkey in _FREQ_PATH_KEYS:
            if pkey in freq_cfg and isinstance(freq_cfg[pkey], str):
                freq_cfg[pkey] = Path(freq_cfg[pkey])
    return cfg


def _load() -> dict:
    """Load config.yml (if exists) and deep-merge over defaults."""
    cfg = deepcopy(_DEFAULTS)
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, overrides)
    return _convert_paths(cfg)


_cfg = _load()


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level constants (keep original uppercase names for compat)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Time range ────────────────────────────────────────────────────────────────
START: str = _cfg["start"]
END: str = _cfg["end"]

# ── Frequency ─────────────────────────────────────────────────────────────────
FREQUENCY: str = _cfg["frequency"]
FREQ_CONFIG: dict = _cfg["freq_config"]

if FREQUENCY not in FREQ_CONFIG:
    raise ValueError(f"Unsupported FREQUENCY={FREQUENCY!r}; expected one of {tuple(FREQ_CONFIG)}")

DATA_PATH: Path = FREQ_CONFIG[FREQUENCY]["market_data"]
PREDS_DIR: Path = FREQ_CONFIG[FREQUENCY]["preds_dir"]
HORIZONS: list[str] = FREQ_CONFIG[FREQUENCY]["horizons"]

BM_PATH: Path = _cfg["bm_path"]
BM_NAME: str = _cfg["bm_name"]
POOL_CACHE_DIR: Path = _cfg["pool_cache_dir"]
USE_POOL_CACHE: bool = _cfg["use_pool_cache"]

# ── Benchmark ─────────────────────────────────────────────────────────────────
USE_EXTERNAL_BENCHMARK: bool = _cfg["use_external_benchmark"]
EXTERNAL_NAV_PATH: Path = _cfg["external_nav_path"]

# ── Universe and strategy ─────────────────────────────────────────────────────
UNIVERSE: list[str] | None = _cfg["universe"]
IS_SHORT: bool = _cfg["is_short"]
ALLOW_ST_OPEN: bool = _cfg["allow_st_open"]
POOL_SIZE: int = _cfg["pool_size"]
PORT_SIZES: list[int] = _cfg["port_sizes"]
THRESH_OUT_BUFFER: int = _cfg["thresh_out_buffer"]
TRADE_ON_NEXT_BAR: bool = _cfg["trade_on_next_bar"]
STRICT_FIRST_BAR_TOP_N: bool = _cfg["strict_first_bar_top_n"]
CLOSE_ON_SIZE_DROP: bool = _cfg["close_on_size_drop"]

# ── Aggregation and costs ─────────────────────────────────────────────────────
AGG_MODE: str = _cfg["agg_mode"]
COMPOUNDING: bool = _cfg["compounding"]
COST_PER_TURNOVER: float = _cfg["cost_per_turnover"]
PORTFOLIO_INITIAL_VALUE: float = _cfg["portfolio_initial_value"]
EXCLUDE_PERIOD: tuple[str, str] | None = (
    tuple(_cfg["exclude_period"]) if isinstance(_cfg.get("exclude_period"), list) else _cfg["exclude_period"]
)

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG: bool = _cfg["debug"]
DEBUG_SYMBOL: str = _cfg["debug_symbol"]
DEBUG_DATETIME: str = _cfg["debug_datetime"]

# ── Eligibility and output ────────────────────────────────────────────────────
NOSUSPEND_DAYS: int = _cfg["nosuspend_days"]
OUTPUT_DIR: Path = Path(f"output_{'short' if IS_SHORT else 'long'}_{POOL_SIZE}_{FREQUENCY}")
