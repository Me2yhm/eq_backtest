"""Convert a complete, explicit stock benchmark into absolute NAV bounds."""

from __future__ import annotations

import math

import numpy as np

from .max_score import SnapshotDataError


def relative_to_absolute_bounds(*, benchmark_asset_ids: list[str], benchmark_weights: np.ndarray,
                                 exposure_asset_ids: list[str], exposure_values: np.ndarray,
                                 expected_benchmark_asset_ids: list[str], benchmark_complete: bool,
                                 benchmark_weight_denominator: str, exposure_metadata: dict,
                                 benchmark_exposure_metadata: dict, target_budget: float,
                                 lower_deviation: float, upper_deviation: float,
                                 deviation_denominator: str) -> dict:
    """Use every benchmark member once; never restrict it to optimizer candidates.

    ``stock_sleeve`` deviations are multiplied by B. ``total_nav`` deviations
    are already absolute NAV weights/exposure and are added after B scaling.
    The caller supplies the independently declared complete membership list;
    missing members, even outside candidates, cause a data error.
    """
    if benchmark_complete is not True or benchmark_weight_denominator != "stock_sleeve":
        raise SnapshotDataError("INVALID_BENCHMARK", "complete stock-sleeve benchmark is required")
    for name, ids in (("benchmark", benchmark_asset_ids), ("exposure", exposure_asset_ids),
                      ("expected_membership", expected_benchmark_asset_ids)):
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise SnapshotDataError("INVALID_BENCHMARK", f"invalid/duplicate {name} asset IDs")
    if set(benchmark_asset_ids) != set(expected_benchmark_asset_ids):
        raise SnapshotDataError("INCOMPLETE_BENCHMARK", "weights do not cover independently declared complete membership")
    if set(exposure_asset_ids) != set(benchmark_asset_ids):
        raise SnapshotDataError("INCOMPLETE_BENCHMARK_EXPOSURE", "exposure membership differs from full benchmark")
    if (not isinstance(benchmark_weights, np.ndarray) or benchmark_weights.dtype != np.float64
            or benchmark_weights.shape != (len(benchmark_asset_ids),) or not np.isfinite(benchmark_weights).all()
            or (benchmark_weights < 0).any() or abs(float(benchmark_weights.sum())-1) > 1e-10):
        raise SnapshotDataError("INVALID_BENCHMARK_WEIGHTS", "finite nonnegative stock weights must sum to one")
    if (not isinstance(exposure_values, np.ndarray) or exposure_values.dtype != np.float64
            or exposure_values.shape != (len(exposure_asset_ids),) or not np.isfinite(exposure_values).all()):
        raise SnapshotDataError("INVALID_BENCHMARK_EXPOSURE", "finite aligned float64 exposures are required")
    if not isinstance(exposure_metadata, dict) or not isinstance(benchmark_exposure_metadata, dict):
        raise SnapshotDataError("EXPOSURE_METADATA_MISMATCH", "source metadata is required")
    identity = ({"classification_id", "version", "level"} if "classification_id" in exposure_metadata
                else {"exposure_id", "field", "version", "unit", "transform"})
    if (not identity.issubset(exposure_metadata) or not identity.issubset(benchmark_exposure_metadata)
            or any(exposure_metadata[k] != benchmark_exposure_metadata[k] for k in identity)):
        raise SnapshotDataError("EXPOSURE_METADATA_MISMATCH", "benchmark and strategy must use the same exposure definition")
    for name, value in (("target_budget", target_budget), ("lower_deviation", lower_deviation),
                        ("upper_deviation", upper_deviation)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= target_budget <= 1 or lower_deviation < 0 or upper_deviation < 0:
        raise ValueError("invalid budget or negative deviation")
    if deviation_denominator not in {"stock_sleeve", "total_nav"}:
        raise ValueError("deviation_denominator must explicitly be stock_sleeve or total_nav")
    exposure_by_id = dict(zip(exposure_asset_ids, exposure_values))
    aligned = np.asarray([exposure_by_id[s] for s in benchmark_asset_ids], dtype=np.float64)
    benchmark_exposure = float(benchmark_weights @ aligned)
    scale = target_budget if deviation_denominator == "stock_sleeve" else 1.0
    return {"lower": target_budget*benchmark_exposure-scale*lower_deviation,
            "upper": target_budget*benchmark_exposure+scale*upper_deviation,
            "benchmark_exposure": benchmark_exposure, "benchmark_member_count": len(benchmark_asset_ids),
            "target_weight_denominator": "total_nav", "deviation_denominator": deviation_denominator}
