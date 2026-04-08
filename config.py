"""
Backtest configuration.
All tunable parameters live here; no other file should define constants.
"""

from pathlib import Path

# ── Time range ────────────────────────────────────────────────────────────────
START = "2019-01-01"

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/daily.pqt")
PREDS_DIR = Path("data/preds_size")
BM_PATH = Path("data/bm_open/ret_csi_1000.csv")

# ── Prediction horizons to average ───────────────────────────────────────────
HORIZONS = ["3d", "5d", "10d"]

# ── Benchmark ─────────────────────────────────────────────────────────────────
BM_NAME = "csi_1000"

# ── Stock universe (index membership required to open a position) ─────────────
UNIVERSE = ["000300.XSHG", "000905.XSHG", "000852.XSHG"]

# ── Strategy direction ────────────────────────────────────────────────────────
IS_SHORT = False  # True: short bottom-ranked stocks; False: long top-ranked
ALLOW_ST_OPEN = True  # Allow opening positions in ST-designated stocks

# ── Candidate pool and portfolio sizes ────────────────────────────────────────
if IS_SHORT:
    POOL_SIZE = 9999  # Effectively unlimited (all stocks)
    PORT_SIZES = [200, 300, 400]
else:
    POOL_SIZE = 3800
    PORT_SIZES = [900]

# thresh_out = port_size + THRESH_OUT_BUFFER  (hysteresis / exit buffer)
THRESH_OUT_BUFFER = 500

# ── Transaction cost (one-way) ────────────────────────────────────────────────
COST_PER_TURNOVER = 0.0004

# ── Period whose excess returns are zeroed (e.g. anomalous market condition) ──
EXCLUDE_PERIOD = ("2024-01-01", "2024-03-31")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(f"output_{'short' if IS_SHORT else 'long'}_{POOL_SIZE}")
