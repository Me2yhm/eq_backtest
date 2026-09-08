"""Measure shm/1 optimizer latency without loading production market data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer_client import OptimizerClient


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 800, 5000])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    results = []
    with OptimizerClient(args.socket) as client:
        client.validate_capacity(max(args.sizes))
        for size in args.sizes:
            waits: list[float] = []
            model_times: list[float] = []
            for iteration in range(args.repeats + 1):
                request = {
                    "schema_version": "1.1", "request_id": str(uuid.uuid4()),
                    "data_context": {"as_of": "2026-09-01T15:00:00+08:00",
                                     "planned_execution_at": "2026-09-02T09:30:00+08:00",
                                     "frequency": "daily"},
                    "portfolio_policy": {"long_only": True, "fully_invested": False,
                                         "allow_cash": True, "allow_leverage": False,
                                         "cash_policy": "retain", "target_budget": 1.0},
                    "model": {"type": "equal_weight", "version": "1", "config": {}},
                    "universe": {"asset_ids": [f"ASSET-{index:05d}" for index in range(size)]},
                    "constraints": [], "options": {"timeout_ms": 5000, "deterministic": True},
                }
                lease, wait_ms = client.optimize(request)
                model_ms = float(lease.response["diagnostics"]["model_time_ms"])
                lease.release()
                if iteration:
                    waits.append(wait_ms)
                    model_times.append(model_ms)
            results.append({
                "assets": size, "repeats": args.repeats,
                "wait_p50_ms": statistics.median(waits), "wait_p95_ms": _percentile(waits, 0.95),
                "model_p50_ms": statistics.median(model_times), "model_p95_ms": _percentile(model_times, 0.95),
                "payload_bytes": size * 8, "transport_copy_bytes": 0,
            })
    print(json.dumps({"warmup_excluded": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
