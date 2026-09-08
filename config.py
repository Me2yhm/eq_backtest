"""Backtest configuration.

Loads overrides from ``config.yml`` (git-ignored) and falls back to defaults
defined below.  All keys are optional — anything missing keeps its default.

Usage::

    import config as cfg
    print(cfg.FREQUENCY)
"""

from __future__ import annotations

from copy import deepcopy
import os
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
            # Optional exact source list for daily production predictions.
            # Each item may be a filename or {file, start, end}; see doc/README.md.
            "prediction_sources": None,
            "market_columns": {"execution_vwap": "vwap30", "bar_close": "close_ex", "prev_close": None},
            "trade_on_next_bar": True,
        },
        "15min": {
            "market_data": "/ext/eq_data/15min_bar_full_left_close.parquet",
            "preds_dir": "/ext/trq",
            "horizons": ["predictions"],
            "prediction_sources": None,
            "market_columns": {"execution_vwap": "vwap15", "bar_close": "close"},
        },
        "5min": {
            "market_data": "/ext/eq_data/5min_bar_full_left_close.parquet",
            "preds_dir": "/tmp/eq_preds/output_mse/output_bs16/predictions",
            "horizons": [""],
            "prediction_sources": None,
            "market_columns": {"execution_vwap": "vwap5", "bar_close": "close"},
        },
    },
    # Existing direct benchmark CSV contract.  When configured, ``bm_path`` is
    # adapted to the market-cache reader without changing its ``date, ret``
    # schema.  The directory/symbol settings remain available for newer runs.
    "bm_path": None,
    "bm_name": None,
    "market_cache_dir": "cache/market_data",
    "benchmark_symbol": "000852",
    "pool_cache_dir": "data/.cache",
    "use_pool_cache": True,
    "benchmark_missing_return_policy": "error",
    "prediction_merge_mode": "concat_disjoint",
    # ── Universe and strategy ─────────────────────────────────────────────────
    "universe": None,
    "is_short": False,
    "allow_st_open": False,
    # A null value preserves legacy is_short behavior for existing run configs.
    "strategy_modes": None,
    "short_borrow_sources": [],
    "borrow_selection": "min_available_rate",
    "pool_size": 4400,
    "port_sizes": [800],
    "thresh_out_buffer": 600,
    "short_port_size": None,
    "short_exit_rank": None,
    "trade_on_next_bar": False,
    "strict_first_bar_top_n": False,
    "close_on_size_drop": False,
    "weight_mode": "equal",
    "max_weight_multiple": 2.0,
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
    # ── Optional same-host optimizer service ─────────────────────────────────
    "optimizer": {
        "enabled": False,
        "transport": "shared_memory",
        "protocol_version": "shm/1",
        "control": "unix_domain_socket",
        "socket_path": "ipc/optimizer.sock",
        "backend": "memfd",
        "zero_copy": "required",
        "schema_version": "1.1",
        "model": {"type": "equal_weight", "version": "1", "config": {}},
        "timeout_ms": 5000,
        "request_timeout_ms": 6000,
        "max_inflight_per_session": 1,
        "max_control_bytes": 1048576,
        "max_shared_bytes": 268435456,
        "failure_policy": "fail",
        "max_retries": 0,
    },
}

# Top-level keys whose string value should be converted to ``Path``.
_PATH_KEYS: frozenset[str] = frozenset({"bm_path", "pool_cache_dir", "market_cache_dir"})

# Keys *inside* each ``freq_config`` sub-table that are paths.
_FREQ_PATH_KEYS: frozenset[str] = frozenset({"market_data", "preds_dir"})


# ═══════════════════════════════════════════════════════════════════════════════
#  Load & merge
# ═══════════════════════════════════════════════════════════════════════════════

_REPO_DIR = Path(__file__).resolve().parent
_RUN_DIR_ENV = "EQ_BACKTEST_RUN_DIR"


def _resolve_run_dir() -> Path:
    """Return the configured run directory, defaulting to the repository root."""
    raw = os.environ.get(_RUN_DIR_ENV)
    if not raw:
        return _REPO_DIR
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


RUN_DIR: Path = _resolve_run_dir()
CONFIG_PATH: Path = RUN_DIR / "config.yml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the active run directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else RUN_DIR / path


def _convert_paths(cfg: dict) -> dict:
    """Convert configured paths and anchor relative values at ``RUN_DIR``."""
    for key in _PATH_KEYS:
        if key in cfg and isinstance(cfg[key], (str, Path)):
            cfg[key] = _resolve_config_path(cfg[key])
    for freq_cfg in cfg.get("freq_config", {}).values():
        for pkey in _FREQ_PATH_KEYS:
            if pkey in freq_cfg and isinstance(freq_cfg[pkey], (str, Path)):
                freq_cfg[pkey] = _resolve_config_path(freq_cfg[pkey])
    raw_borrow_sources = cfg.get("short_borrow_sources", [])
    if isinstance(raw_borrow_sources, list):
        for source in raw_borrow_sources:
            if isinstance(source, dict) and isinstance(source.get("path"), (str, Path)):
                source["path"] = _resolve_config_path(source["path"])
    optimizer = cfg.get("optimizer")
    if isinstance(optimizer, dict) and isinstance(optimizer.get("socket_path"), (str, Path)):
        optimizer["socket_path"] = _resolve_config_path(optimizer["socket_path"])
    return cfg


def _apply_legacy_benchmark_aliases(cfg: dict, overrides: dict) -> dict:
    """Map the existing ``bm_path`` contract onto the benchmark reader.

    Explicit ``market_cache_dir`` or ``benchmark_symbol`` values still win.
    This keeps old run files and their ``date, ret`` CSVs usable without a
    data migration or a second benchmark configuration.
    """
    raw_path = overrides.get("bm_path")
    if raw_path is None:
        return cfg
    if not isinstance(raw_path, (str, Path)):
        raise ValueError("bm_path must be a path or null")
    path = Path(raw_path)
    if not path.name:
        raise ValueError("bm_path must identify a benchmark CSV file")
    if "market_cache_dir" not in overrides:
        cfg["market_cache_dir"] = str(path.parent)
    if "benchmark_symbol" not in overrides:
        cfg["benchmark_symbol"] = path.stem
    return cfg


def _load() -> dict:
    """Load config.yml (if exists) and deep-merge over defaults."""
    cfg = deepcopy(_DEFAULTS)
    overrides: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, overrides)
    return _convert_paths(_apply_legacy_benchmark_aliases(cfg, overrides))


_cfg = _load()

def _strategy_modes(raw: object, is_short: bool) -> tuple[str, ...]:
    if raw is None:
        return ("short_only",) if is_short else ("long_only",)
    if not isinstance(raw, list) or not raw:
        raise ValueError("strategy_modes must be a non-empty list or null")
    modes = tuple(str(mode) for mode in raw)
    allowed = {"long_only", "short_only", "long_short"}
    unexpected = set(modes).difference(allowed)
    if unexpected:
        raise ValueError(f"Unsupported strategy_modes: {sorted(unexpected)}")
    if len(set(modes)) != len(modes):
        raise ValueError("strategy_modes must not contain duplicates")
    return modes

def _benchmark_missing_return_policy(raw: object) -> str:
    if raw not in {"error", "zero"}:
        raise ValueError("benchmark_missing_return_policy must be 'error' or 'zero'")
    return str(raw)

def _optional_positive_int(raw: object, key: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ValueError(f"{key} must be a positive integer or null")
    return raw


def _optimizer_config(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("optimizer must be an object")
    allowed = {
        "enabled", "transport", "protocol_version", "control", "socket_path", "backend",
        "zero_copy", "schema_version", "model", "timeout_ms", "request_timeout_ms",
        "max_inflight_per_session", "max_control_bytes", "max_shared_bytes",
        "failure_policy", "max_retries",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported optimizer settings: {sorted(unknown)}")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("optimizer.enabled must be boolean")
    if not isinstance(raw.get("socket_path"), Path):
        raise ValueError("optimizer.socket_path must be a path")
    for key in ("timeout_ms", "request_timeout_ms", "max_control_bytes", "max_shared_bytes"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"optimizer.{key} must be a positive integer")
    if raw["request_timeout_ms"] < raw["timeout_ms"]:
        raise ValueError("optimizer.request_timeout_ms must be >= optimizer.timeout_ms")
    if not enabled:
        return raw
    try:
        raw["socket_path"].resolve().relative_to(RUN_DIR.resolve())
    except ValueError as exc:
        raise ValueError("optimizer.socket_path must remain inside the dedicated run directory") from exc
    exact = {
        "transport": "shared_memory", "protocol_version": "shm/1",
        "control": "unix_domain_socket", "backend": "memfd", "zero_copy": "required",
        "schema_version": "1.1", "failure_policy": "fail", "max_retries": 0,
        "max_inflight_per_session": 1,
    }
    for key, value in exact.items():
        if raw.get(key) != value:
            raise ValueError(f"optimizer.{key} must be {value!r}")
    model = raw.get("model")
    if model != {"type": "equal_weight", "version": "1", "config": {}}:
        raise ValueError("optimizer.model must be equal_weight/1 with empty config")
    return raw





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

BM_PATH: Path | None = _cfg["bm_path"]
BM_NAME: str | None = _cfg["bm_name"]
MARKET_CACHE_DIR: Path = _cfg["market_cache_dir"]
BENCHMARK_SYMBOL: str = _cfg["benchmark_symbol"]
BENCHMARK_NAME: str = BM_NAME or BENCHMARK_SYMBOL
POOL_CACHE_DIR: Path = _cfg["pool_cache_dir"]
USE_POOL_CACHE: bool = _cfg["use_pool_cache"]
BENCHMARK_MISSING_RETURN_POLICY: str = _benchmark_missing_return_policy(_cfg["benchmark_missing_return_policy"])
PREDICTION_MERGE_MODE: str = _cfg["prediction_merge_mode"]

# ── Universe and strategy ─────────────────────────────────────────────────────
SHORT_PORT_SIZE: int | None = _optional_positive_int(_cfg["short_port_size"], "short_port_size")
SHORT_EXIT_RANK: int | None = _optional_positive_int(_cfg["short_exit_rank"], "short_exit_rank")
if SHORT_PORT_SIZE is not None and SHORT_EXIT_RANK is not None and SHORT_EXIT_RANK < SHORT_PORT_SIZE:
    raise ValueError("short_exit_rank must be greater than or equal to short_port_size")
UNIVERSE: list[str] | None = _cfg["universe"]
IS_SHORT: bool = _cfg["is_short"]
ALLOW_ST_OPEN: bool = _cfg["allow_st_open"]
STRATEGY_MODES: tuple[str, ...] = _strategy_modes(_cfg["strategy_modes"], IS_SHORT)
SHORT_BORROW_SOURCES: list[dict] = _cfg["short_borrow_sources"]
BORROW_SELECTION: str = _cfg["borrow_selection"]
POOL_SIZE: int = _cfg["pool_size"]
PORT_SIZES: list[int] = _cfg["port_sizes"]
THRESH_OUT_BUFFER: int = _cfg["thresh_out_buffer"]
TRADE_ON_NEXT_BAR: bool = _cfg["trade_on_next_bar"]
STRICT_FIRST_BAR_TOP_N: bool = _cfg["strict_first_bar_top_n"]
CLOSE_ON_SIZE_DROP: bool = _cfg["close_on_size_drop"]
WEIGHT_MODE: str = _cfg["weight_mode"]
MAX_WEIGHT_MULTIPLE: float = _cfg["max_weight_multiple"]

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
OPTIMIZER: dict = _optimizer_config(_cfg["optimizer"])


def output_dir_for_mode(mode: str) -> Path:
    labels = {"long_only": "long", "short_only": "short", "long_short": "long_short"}
    try:
        return RUN_DIR / f"output_{labels[mode]}_{POOL_SIZE}_{FREQUENCY}"
    except KeyError as exc:
        raise ValueError(f"Unsupported strategy mode: {mode!r}") from exc


OUTPUT_DIR: Path = output_dir_for_mode(STRATEGY_MODES[0])


def trade_on_next_bar_for(frequency: str) -> bool:
    """Return the frequency-specific execution-lag setting."""
    return bool(FREQ_CONFIG.get(frequency, {}).get("trade_on_next_bar", TRADE_ON_NEXT_BAR))


if OPTIMIZER["enabled"]:
    if FREQUENCY != "daily" or STRATEGY_MODES != ("long_only",):
        raise ValueError("optimizer is currently supported only for daily + long_only")
    if WEIGHT_MODE != "equal":
        raise ValueError("optimizer equal_weight integration requires weight_mode='equal'")
    if not trade_on_next_bar_for(FREQUENCY):
        raise ValueError("optimizer daily integration requires trade_on_next_bar=true")
