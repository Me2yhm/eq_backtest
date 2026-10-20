"""Strict business settings for the opt-in max_score backtest path."""

from __future__ import annotations

import math


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_max_score_settings(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("optimizer.max_score must be an object")
    allowed = {"purpose", "target_budget", "max_weight", "size_constraint", "industry_constraint",
               "turnover_constraint", "candidate_policy", "missing_valuation_policy", "no_signal_policy",
               "source_point_in_time_status", "price_convention"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown optimizer.max_score settings: {sorted(unknown)}")
    for key, allowed_values in {
        "purpose": {"engineering_acceptance", "strategy_backtest"},
        "candidate_policy": {"finite_score_eligible", "eligible"},
        "missing_valuation_policy": {"error", "carry_last"},
        "no_signal_policy": {"error", "hold"},
        "source_point_in_time_status": {"unverified", "verified", "synthetic"},
        "price_convention": {"consistent_share_prices", "engineering_unverified"},
    }.items():
        if raw.get(key) not in allowed_values:
            raise ValueError(f"optimizer.max_score.{key} must be explicitly selected from {sorted(allowed_values)}")
    if raw["purpose"] != "engineering_acceptance" and raw["price_convention"] == "engineering_unverified":
        raise ValueError("unverified execution prices are restricted to engineering_acceptance")
    budget = _number(raw.get("target_budget"), "optimizer.max_score.target_budget")
    if not 0 <= budget <= 1:
        raise ValueError("optimizer.max_score.target_budget must be in [0, 1]")
    cap = raw.get("max_weight")
    if cap is not None:
        values = cap.values() if isinstance(cap, dict) else [cap]
        if isinstance(cap, dict) and (not cap or any(not isinstance(s, str) or not s for s in cap)):
            raise ValueError("max_weight asset mapping requires non-empty symbol IDs")
        if any(not 0 <= _number(v, "max_weight") <= 1 for v in values):
            raise ValueError("max_weight must be in [0, 1]")
    size = raw.get("size_constraint")
    if size is not None:
        if not isinstance(size, dict) or set(size) != {"lower", "upper", "source"}:
            raise ValueError("size_constraint requires lower, upper, source")
        if _number(size["lower"], "size.lower") > _number(size["upper"], "size.upper"):
            raise ValueError("size lower must not exceed upper")
    industry = raw.get("industry_constraint")
    if industry is not None:
        if not isinstance(industry, dict) or set(industry) != {"bounds", "source"}:
            raise ValueError("industry_constraint requires bounds, source")
        if not isinstance(industry["bounds"], list) or not industry["bounds"]:
            raise ValueError("industry bounds must be a non-empty explicit list")
        seen = set()
        for bound in industry["bounds"]:
            if not isinstance(bound, dict) or set(bound) != {"industry_id", "lower", "upper"}:
                raise ValueError("industry bound requires industry_id, lower, upper")
            identity = bound["industry_id"]
            if not isinstance(identity, str) or identity in seen or identity in {"", "0", "0.0"}:
                raise ValueError("industry IDs must be unique valid nonzero strings")
            seen.add(identity)
            lower, upper = (_number(bound[k], f"industry.{k}") for k in ("lower", "upper"))
            if not 0 <= lower <= upper <= 1:
                raise ValueError("industry bounds must satisfy 0 <= lower <= upper <= 1")
    turnover = raw.get("turnover_constraint")
    if turnover is not None:
        if not isinstance(turnover, dict) or set(turnover) != {"definition", "limit"}:
            raise ValueError("turnover_constraint requires definition and limit")
        if turnover["definition"] not in {"stock_l1", "stock_half_l1", "cash_inclusive_half_l1"}:
            raise ValueError("unsupported explicit turnover definition")
        if _number(turnover["limit"], "turnover.limit") < 0:
            raise ValueError("turnover.limit must be nonnegative")
    return raw
