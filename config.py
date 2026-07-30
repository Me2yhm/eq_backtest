"""Backtest configuration. Frequency-specific inputs live in ``FREQ_CONFIG``."""

from pathlib import Path

# ── Time range ────────────────────────────────────────────────────────────────
START = "2019-01-02"
END = "2026-01-01"

# ── Frequency ─────────────────────────────────────────────────────────────────
FREQUENCY = "15min"
FREQ_CONFIG = {
    "daily": {
        "market_data": Path("/ext/eq_data/daily_with_limit_prevcap.pqt"),
        "preds_dir": Path("data/preds_size"),
        "horizons": ["3d", "5d", "10d"],
        # This source exposes only bar return; its price path is derived in the loader.
        "market_columns": {"execution_vwap": None, "bar_close": None},
    },
    "15min": {
        "market_data": Path("/ext/eq_data/15min_bar_full_left_close.parquet"),
        "preds_dir": Path("/ext/trq"),
        "horizons": ["predictions"],
        "market_columns": {"execution_vwap": "vwap15", "bar_close": "close"},
    },
    "5min": {
        "market_data": Path("/ext/eq_data/5min_bar_full_left_close.parquet"),
        "preds_dir": Path("/tmp/eq_preds/output_mse/output_bs16/predictions"),
        "horizons": [""],
        "market_columns": {"execution_vwap": "vwap5", "bar_close": "close"},
    },
}
if FREQUENCY not in FREQ_CONFIG:
    raise ValueError(f"Unsupported FREQUENCY={FREQUENCY!r}; expected one of {tuple(FREQ_CONFIG)}")

DATA_PATH = FREQ_CONFIG[FREQUENCY]["market_data"]
PREDS_DIR = FREQ_CONFIG[FREQUENCY]["preds_dir"]
HORIZONS = FREQ_CONFIG[FREQUENCY]["horizons"]

BM_PATH = Path("/ext/eq_data/ret_csi_1000.csv")
BM_NAME = "csi_1000"
POOL_CACHE_DIR = Path("data/.cache")
USE_POOL_CACHE = True

# ── Benchmark ─────────────────────────────────────────────────────────────────
USE_EXTERNAL_BENCHMARK = True
EXTERNAL_NAV_PATH = Path("/ext/trq/nav.parquet")

# ── Universe and strategy ─────────────────────────────────────────────────────
UNIVERSE = None
IS_SHORT = False
ALLOW_ST_OPEN = False

if IS_SHORT:
    POOL_SIZE = 9999
    PORT_SIZES = [200, 300, 400]
else:
    POOL_SIZE = 4400
    PORT_SIZES = [800]

THRESH_OUT_BUFFER = 600
TRADE_ON_NEXT_BAR = True
STRICT_FIRST_BAR_TOP_N = False
CLOSE_ON_SIZE_DROP = False

# ── Aggregation and costs ──────────────────────────────────────────────────────
AGG_MODE = "simple"
COMPOUNDING = False
COST_PER_TURNOVER = 0.00045
PORTFOLIO_INITIAL_VALUE = 1e8
EXCLUDE_PERIOD = ("2024-01-01", "2024-03-31")

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG = False
DEBUG_SYMBOL = "300169.XSHE"
DEBUG_DATETIME = "2019-01-03 09:46:00"

# ── Eligibility and output ────────────────────────────────────────────────────
NOSUSPEND_DAYS = 10
OUTPUT_DIR = Path(f"output_{'short' if IS_SHORT else 'long'}_{POOL_SIZE}_{FREQUENCY}")
