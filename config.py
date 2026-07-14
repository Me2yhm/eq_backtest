"""
Backtest configuration.
All tunable parameters live here; no other file should define constants.
"""

from pathlib import Path

# ── Time range ────────────────────────────────────────────────────────────────
START = "2019-01-02"
# END = None  # e.g. "2026-01-01"; None means no cutoff
END = "2026-01-01"


# ── Data paths ────────────────────────────────────────────────────────────────
# DATA_PATH = Path("data/daily.pqt")
DATA_PATH = Path("/ext/eq_data/daily_with_limit_prevcap.pqt")
PREDS_DIR = Path("data/preds_size")
BM_PATH = Path("/ext/eq_data/ret_csi_1000.csv")
POOL_CACHE_DIR = Path("data/.cache")
USE_POOL_CACHE = True

# ── Prediction horizons to average ───────────────────────────────────────────
HORIZONS = ["3d", "5d", "10d"]

# ── Benchmark ─────────────────────────────────────────────────────────────────
BM_NAME = "csi_1000"

# ── Stock universe (index membership required to open a position) ─────────────
# UNIVERSE = ["000300.XSHG", "000905.XSHG", "000852.XSHG"]
UNIVERSE = None

# ── Strategy direction ────────────────────────────────────────────────────────
IS_SHORT = False  # True: short bottom-ranked stocks; False: long top-ranked
ALLOW_ST_OPEN = False  # Allow opening positions in ST-designated stocks

# ── Candidate pool and portfolio sizes ────────────────────────────────────────
if IS_SHORT:
    POOL_SIZE = 9999  # Effectively unlimited (all stocks)
    PORT_SIZES = [200, 300, 400]
else:
    POOL_SIZE = 4400
    PORT_SIZES = [800]

# thresh_out = port_size + THRESH_OUT_BUFFER  (hysteresis / exit buffer)
THRESH_OUT_BUFFER = 600
TRADE_ON_NEXT_DAY = True  # True: day T signals are executed on day T+1; False: same-day signal/same-day portfolio
STRICT_FIRST_DAY_TOP_N = False  # True: first day only opens from strict top-N ranks; False: keep scanning until full

# ── Aggregation & annualization (对齐外部 metrics.py) ─────────────────────────
# AGG_MODE: bar→日聚合方式
#   "simple"   - 日收益 = sum(bar_ret)  （外部使用，算术叠加）
#   "compound" - 日收益 = (1+x).prod()-1  （复利叠加）
AGG_MODE = "simple"

# COMPOUNDING: 年化方式
#   False - 算术年化: mean × 242  （外部使用）
#   True  - 几何年化: (1+x).prod()^(242/n) - 1
COMPOUNDING = False

# ── Transaction cost (one-way) ────────────────────────────────────────────────
COST_PER_TURNOVER = 0.00045

# ── Period whose excess returns are zeroed (e.g. anomalous market condition) ──
EXCLUDE_PERIOD = ("2024-01-01", "2024-03-31")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(f"output_{'short' if IS_SHORT else 'long'}_{POOL_SIZE}")

# ── 15-minute backtest mode ───────────────────────────────────────────────────
USE_15MIN = True
DATA_15MIN_PATH = Path("/ext/eq_data/15min_bar_full_left_close.parquet")
PREDS_15MIN_DIR = Path("/ext/trq")
# HORIZONS_15MIN = ["3b", "5b", "10b"]
HORIZONS_15MIN = ["predictions"]


# 15-minute execution timing
TRADE_ON_NEXT_BAR = False
STRICT_FIRST_BAR_TOP_N = False

# 15-minute debug tracing
DEBUG_15MIN = False
DEBUG_SYMBOL_15MIN = "300169.XSHE"
DEBUG_DATETIME_15MIN = "2019-01-03 09:46:00"

# ── Eligibility ─────────────────────────────────────────────────────────────────
NOSUSPEND_DAYS = (
    10  # Consecutive days without suspension to be eligible for universe (external: _universe_nosuspend_days=10)
)

# ── Close on size drop ─────────────────────────────────────────────────────────
CLOSE_ON_SIZE_DROP = False  # True: close positions when they drop out of size pool; False: keep holding as long as still valid_listed (matches external _drop_out_of_universe_immediate=false)
