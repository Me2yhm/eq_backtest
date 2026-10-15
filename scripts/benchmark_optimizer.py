"""Measure optimizer backend latency without loading production market data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from optimizer_client import InProcessOptimizerClient, OptimizerClient


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def benchmark(mode: str, socket_path: Path | None, sizes: list[int], repeats: int) -> list[dict]:
    results = []
    backend = InProcessOptimizerClient() if mode == "inprocess" else OptimizerClient(socket_path)
    with backend as client:
        client.validate_capacity(max(sizes))
        before_fds = len(os.listdir("/proc/self/fd"))
        for size in sizes:
            waits: list[float] = []
            model_times: list[float] = []
            calls: list[float] = []
            mapping_times: list[float] = []
            for iteration in range(repeats + 1):
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
                started = time.perf_counter_ns()
                lease, wait_ms = client.optimize(request)
                try:
                    model_ms = float(lease.response["diagnostics"]["model_time_ms"])
                    mapping_ms = lease.mapping_validation_ms
                    shared_bytes, copy_bytes = lease.shared_bytes, lease.copy_bytes
                    if copy_bytes != 0 or shared_bytes != (size * 8 if mode == "shared_memory" else 0):
                        raise AssertionError("backend reported unexpected transport byte counters")
                    np.testing.assert_allclose(lease.weights, 1.0 / size, rtol=0, atol=1e-10)
                finally:
                    lease.release()
                call_ms = (time.perf_counter_ns() - started) / 1_000_000
                if iteration:
                    waits.append(wait_ms)
                    model_times.append(model_ms)
                    mapping_times.append(mapping_ms)
                    calls.append(call_ms)
                else:
                    warmup_ms = call_ms
            results.append({
                "mode": mode, "assets": size, "repeats": repeats,
                "wait_p50_ms": statistics.median(waits), "wait_p95_ms": _percentile(waits, 0.95),
                "model_p50_ms": statistics.median(model_times), "model_p95_ms": _percentile(model_times, 0.95),
                "call_p50_ms": statistics.median(calls), "call_p95_ms": _percentile(calls, 0.95),
                "mapping_p50_ms": statistics.median(mapping_times), "mapping_p95_ms": _percentile(mapping_times, 0.95),
                "warmup_call_ms": warmup_ms,
                "payload_bytes": size * 8, "shared_bytes": shared_bytes, "transport_copy_bytes": copy_bytes,
            })
        after_fds = len(os.listdir("/proc/self/fd"))
        if after_fds != before_fds:
            raise AssertionError(f"client FD count changed: {before_fds} -> {after_fds}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inprocess", "shared_memory", "both"), default="shared_memory")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 800, 5000])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.mode != "inprocess" and args.socket is None:
        parser.error("--socket is required for shared_memory or both")
    if args.repeats < 1 or min(args.sizes) < 1:
        parser.error("sizes and repeats must be positive")
    modes = ("inprocess", "shared_memory") if args.mode == "both" else (args.mode,)
    results = [row for mode in modes for row in benchmark(mode, args.socket, args.sizes, args.repeats)]
    print(json.dumps({
        "warmup_excluded": True, "warmup_calls_per_size": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "kernel": platform.release(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "call_scope": "optimize, map, validate weights, CONSUMED and RELEASE; excludes request construction",
        "copy_scope": "numeric output model write and consumer view; excludes control JSON and audit materialization",
        "fd_growth": 0, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
