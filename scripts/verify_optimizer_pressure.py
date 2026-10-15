"""Reproduce retained SHM outputs and verify long-horizon CLI resource bounds.

Every constrained process changes its own RLIMIT_NOFILE.  The orchestrating
process never changes the user's shell limits, and samples both real processes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import tempfile
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _snapshot(pid: int) -> dict:
    proc = Path("/proc") / str(pid)
    result: dict = {"pid": pid}
    try:
        descriptors = os.listdir(proc / "fd")
        result["fd"] = len(descriptors)
        targets = []
        for descriptor in descriptors:
            try:
                targets.append(os.readlink(proc / "fd" / descriptor))
            except OSError:
                pass  # A concurrently closing FD is normal during sampling.
        result["optimizer_fds"] = sum("optimizer-buffer" in name or "optimizer-arena" in name for name in targets)
        lines = (proc / "maps").read_text().splitlines()
        buffers = [line for line in lines if "optimizer-buffer" in line or "optimizer-arena" in line]
        result["maps"] = len(lines)
        result["optimizer_maps"] = len(buffers)
        result["optimizer_mapped_bytes"] = sum(
            int(line.split()[0].split("-")[1], 16) - int(line.split()[0].split("-")[0], 16)
            for line in buffers
        )
        status = dict(
            line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line
        )
        result["rss_kib"] = int(status["VmRSS"].split()[0]) if "VmRSS" in status else 0
    except (OSError, ValueError) as exc:
        result["sampling_error"] = str(exc)
    return result


def _merge_peak(peak: dict, sample: dict) -> None:
    for key in ("fd", "optimizer_fds", "maps", "optimizer_maps", "optimizer_mapped_bytes", "rss_kib"):
        if key in sample:
            peak[key] = max(peak.get(key, 0), sample[key])


def _limit(nofile: int) -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(nofile, hard), hard))


def _request(index: int, assets: int = 4) -> dict:
    return {
        "schema_version": "1.1", "request_id": f"pressure-{index}",
        "data_context": {"as_of": "2026-09-01T15:00:00+08:00",
                         "planned_execution_at": "2026-09-02T09:30:00+08:00", "frequency": "daily"},
        "portfolio_policy": {"long_only": True, "fully_invested": False,
                             "allow_cash": True, "allow_leverage": False,
                             "cash_policy": "retain", "target_budget": 1.0},
        "model": {"type": "equal_weight", "version": "1", "config": {}},
        "universe": {"asset_ids": [f"ASSET-{asset:05d}" for asset in range(assets)]},
        "constraints": [], "options": {"timeout_ms": 5000, "deterministic": True},
    }


def _retained_worker(socket_path: Path, requests: int, nofile: int, service_pid: int, output: Path) -> None:
    _limit(nofile)
    import optimizer
    import importlib.metadata
    from optimizer.transport.shm import ShmOptimizerClient

    client = ShmOptimizerClient(socket_path)
    leases = []
    report = {
        "requested": requests, "completed": 0, "rlimit_nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
        "optimizer_file": optimizer.__file__, "python": platform.python_version(),
        "installed_source": json.loads(importlib.metadata.distribution("optimizer").read_text("direct_url.json") or "{}"),
        "baseline_client": _snapshot(os.getpid()), "baseline_service": _snapshot(service_pid),
        "peak_client": {}, "peak_service": {}, "curve": [], "cleanup_errors": [],
    }
    try:
        client.connect()
        for index in range(requests):
            lease, _ = client.optimize(_request(index))
            leases.append(lease)
            report["completed"] = index + 1
            client_sample, service_sample = _snapshot(os.getpid()), _snapshot(service_pid)
            _merge_peak(report["peak_client"], client_sample)
            _merge_peak(report["peak_service"], service_sample)
            if index + 1 in (1, 10, 100, 1000, 2000, requests):
                report["curve"].append({"outputs": index + 1, "client": client_sample, "service": service_sample})
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        report["failed_after"] = len(leases)
        report["curve"].append({"outputs": len(leases), "client": _snapshot(os.getpid()), "service": _snapshot(service_pid)})
    finally:
        for lease in leases:
            try:
                lease.release()
            except BaseException as exc:
                report["cleanup_errors"].append({"type": type(exc).__name__, "message": str(exc)})
        leases.clear()
        client.close()
        report["final_client"] = _snapshot(os.getpid())
        report["final_service"] = _snapshot(service_pid)
        _write_json(output, report)


def _wait_socket(service: subprocess.Popen, socket_path: Path) -> None:
    deadline = time.monotonic() + 15
    while not socket_path.exists():
        if service.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(f"service did not create {socket_path}")
        time.sleep(0.005)


def _service_command(socket_path: Path, nofile: int, max_shared_bytes: int | None = None) -> list[str]:
    command = [sys.executable, str(SCRIPT), "_service", "--socket", str(socket_path), "--nofile", str(nofile)]
    if max_shared_bytes is not None:
        command += ["--max-shared-bytes", str(max_shared_bytes)]
    return command


def retained_baseline(work_dir: Path, *, counts: tuple[int, ...] = (10, 100, 1000, 2000), nofile: int = 256) -> dict:
    work_dir.mkdir(exist_ok=False)
    runs = []
    for count in counts:
        run_dir = work_dir / str(count)
        run_dir.mkdir()
        socket_path = run_dir / "service.sock"
        with (run_dir / "service.log").open("w") as log:
            service = subprocess.Popen(_service_command(socket_path, nofile), cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_socket(service, socket_path)
                with (run_dir / "client.log").open("w") as client_log:
                    client = subprocess.Popen([
                        sys.executable, str(SCRIPT), "_retained", "--socket", str(socket_path),
                        "--requests", str(count), "--nofile", str(nofile), "--service-pid", str(service.pid),
                        "--output", str(run_dir / "client.json"),
                    ], cwd=run_dir, stdout=client_log, stderr=subprocess.STDOUT)
                    monitor = {"client": {}, "service": {}, "sample_interval_seconds": 0.005, "samples": 0}
                    while client.poll() is None:
                        _merge_peak(monitor["client"], _snapshot(client.pid))
                        _merge_peak(monitor["service"], _snapshot(service.pid))
                        monitor["samples"] += 1
                        time.sleep(0.005)
                if client.returncode:
                    raise RuntimeError(f"pressure worker exited {client.returncode}: {run_dir / 'client.log'}")
                run_report = json.loads((run_dir / "client.json").read_text())
                run_report["external_monitor"] = monitor
                # Allow disconnect cleanup, then sample the still-live service.
                time.sleep(0.03)
                run_report["service_after_client_exit"] = _snapshot(service.pid)
            finally:
                service.terminate()
                service.wait(timeout=10)
        run_report["service_exit"] = service.returncode
        run_report["socket_residual"] = socket_path.exists()
        run_report["service_error_log"] = (run_dir / "service.log").read_text()
        runs.append(run_report)
    result = {
        "experiment": "one retained legacy SHM lease per request", "nofile": nofile,
        "environment": {"python": platform.python_version(), "kernel": platform.release()},
        "status": "resource_limit_reproduced" if any(row["status"] == "failed" for row in runs) else "completed",
        "runs": runs,
    }
    _write_json(work_dir / "comparison.json", result)
    return result


def _long_fixture(root: Path, target_days: int) -> tuple[object, dict]:
    import numpy as np
    import pandas as pd
    import polars as pl

    # One extra market date executes the last decision.  The count deliberately
    # excludes NO_SIGNAL and LIQUIDATE, so 2,000 means actual optimizer requests.
    kinds, valid = [], 0
    while valid < target_days:
        day = len(kinds)
        kind = "no_signal" if day % 101 == 2 else "liquidate" if day % 101 == 3 else "target"
        kinds.append(kind)
        valid += kind == "target"
    dates = pd.bdate_range("2015-01-05", periods=len(kinds) + 1)
    root.mkdir()
    (root / "predictions").mkdir()
    (root / "market_cache").mkdir()
    rng = np.random.default_rng(20261015)
    prices = 10.0 * np.cumprod(1 + rng.normal(0, 0.003, (len(dates), 4)), axis=0)
    market, predictions = [], []
    for day, stamp in enumerate(dates):
        kind = kinds[day] if day < len(kinds) else "target"
        for asset in reversed(range(4)):
            symbol = f"{asset + 1:06d}.XSHE"
            market.append({
                "date": stamp.date(), "symbol": symbol, "turnover": 1e7,
                "log_size": 10.0 - asset, "size_rank": asset + 1,
                "industry": "synthetic", "index": "TEST", "ret": 0.0,
                "is_limit_up": day % 101 == 5 and asset == 0,
                "is_limit_down": day % 101 == 4 and asset == 2,
                "listed_Satisfied": True,
                "is_ST": kind == "liquidate" or (day % 101 == 1 and asset != 2),
                "normal_days": 20, "vwap30": float(prices[day, asset]),
                "close_ex": float(prices[day, asset] * (1 + 0.001 * (asset - 1))),
            })
            if kind != "no_signal":
                predictions.append({"date": stamp.date(), "symbol": symbol,
                                    "prediction": 1.0 if day % 101 == 0 else float((asset + day) % 4)})
    pl.DataFrame(market).write_parquet(root / "market.parquet")
    pl.DataFrame(predictions).write_parquet(root / "predictions/signals.parquet")
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": 100.0,
                  "pct_change": np.linspace(-0.001, 0.001, len(dates))}).to_csv(root / "market_cache/TEST.csv", index=False)
    return dates, {"target": valid, "no_signal": kinds.count("no_signal"), "liquidate": kinds.count("liquidate")}


def _long_config(run_dir: Path, mode: str, dates: object, max_shared_bytes: int | None = None) -> None:
    import yaml

    run_dir.mkdir()
    config = {
        "frequency": "daily", "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "strategy_modes": ["long_only"], "weight_mode": "equal", "port_sizes": [4],
        "pool_size": 10, "thresh_out_buffer": 0, "close_on_size_drop": True,
        "exclude_period": None, "use_pool_cache": False, "market_cache_dir": "../fixture/market_cache",
        "benchmark_symbol": "TEST", "freq_config": {"daily": {
            "market_data": "../fixture/market.parquet", "preds_dir": "../fixture/predictions",
            "prediction_sources": ["signals.parquet"], "trade_on_next_bar": True,
        }}, "optimizer": {"enabled": True, "mode": mode, "socket_path": "ipc/service.sock"},
    }
    if max_shared_bytes is not None:
        config["optimizer"]["max_shared_bytes"] = max_shared_bytes
    (run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _cli_worker(run_dir: Path, nofile: int) -> None:
    _limit(nofile)
    report = {"baseline": _snapshot(os.getpid()), "rlimit_nofile": resource.getrlimit(resource.RLIMIT_NOFILE)}
    try:
        sys.path.insert(0, str(REPO))
        sys.argv = [str(REPO / "run.py"), str(run_dir)]
        runpy.run_path(str(REPO / "run.py"), run_name="__main__")
    finally:
        gc.collect()
        report["final"] = _snapshot(os.getpid())
        _write_json(run_dir / "cli_resources.json", report)


def _run_monitored_cli(run_dir: Path, nofile: int, service: subprocess.Popen) -> dict:
    env = {**os.environ, "MPLBACKEND": "Agg", "POLARS_MAX_THREADS": "4"}
    peaks = {"client": {}, "service": {}, "samples": 0, "sample_interval_seconds": 0.005}
    started = time.monotonic()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "_cli", "--work-dir", str(run_dir),
                                  "--nofile", str(nofile)], cwd=REPO, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time.monotonic() - started > 300:
                    child.kill()
                    raise TimeoutError("long-horizon CLI exceeded 300 seconds")
                _merge_peak(peaks["client"], _snapshot(child.pid))
                _merge_peak(peaks["service"], _snapshot(service.pid))
                peaks["samples"] += 1
                time.sleep(0.005)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
    status_path = run_dir / "optimizer_run_status.json"
    resources_path = run_dir / "cli_resources.json"
    cleanup_started = time.monotonic()
    service_final = _snapshot(service.pid)
    while service_final.get("optimizer_fds", 0) or service_final.get("optimizer_maps", 0):
        if time.monotonic() - cleanup_started > 2:
            break
        time.sleep(0.005)
        service_final = _snapshot(service.pid)
    return {"returncode": child.returncode, "elapsed_seconds": time.monotonic() - started,
            "peaks": peaks, "run_status": json.loads(status_path.read_text()) if status_path.exists() else None,
            "child_resources": json.loads(resources_path.read_text()) if resources_path.exists() else None,
            "service_cleanup_wait_seconds": time.monotonic() - cleanup_started,
            "service_after_client_exit": service_final,
            "output_files": sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("output_*/*") if path.is_file()),
            "metrics_files": sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("output_*/metrics_*.csv"))}


def _compare_long(left: Path, right: Path, expected_events: dict) -> dict:
    import numpy as np
    import pandas as pd
    from verify_optimizer_modes import _calls, _normalize, TRANSPORT_FIELDS

    direct_calls, direct_ids = _calls(left, "inprocess")
    shm_calls, shm_ids = _calls(right, "shared_memory")
    observed = {event: sum(row["event"] == event for row in direct_calls) for event in expected_events}
    assert observed == expected_events, (observed, expected_events)
    assert len(direct_calls) == len(shm_calls)
    for direct, shm in zip(direct_calls, shm_calls):
        for row, ids in ((direct, direct_ids), (shm, shm_ids)):
            if row["request_id"]:
                row["request_id"] = ids[row["request_id"]]
        assert {key: value for key, value in direct.items() if key not in TRANSPORT_FIELDS} == {
            key: value for key, value in shm.items() if key not in TRANSPORT_FIELDS}
    output_left, output_right = (root / "output_long_10_daily" for root in (left, right))
    files = sorted(path.name for path in output_left.iterdir() if path.suffix in {".csv", ".parquet"})
    assert files == sorted(path.name for path in output_right.iterdir() if path.suffix in {".csv", ".parquet"})
    max_weight = max_return = max_cash = 0.0
    for name in files:
        read = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        direct, shm = _normalize(read(output_left / name), direct_ids), _normalize(read(output_right / name), shm_ids)
        pd.testing.assert_frame_equal(direct, shm, check_exact=False, rtol=0, atol=1e-10 if "target" in name else 1e-8)
        for column in direct.select_dtypes(include="number"):
            error = float((direct[column] - shm[column]).abs().max())
            if "weight" in column:
                max_weight = max(max_weight, error)
            if "daily_" in column or name.startswith("returns_"):
                max_return = max(max_return, error)
        if name == "positions_4.csv":
            cash_direct = 1 - direct.groupby("date")["weight_actual"].sum()
            cash_shm = 1 - shm.groupby("date")["weight_actual"].sum()
            max_cash = float((cash_direct - cash_shm).abs().max())
    targets = pd.read_parquet(output_left / "decision_targets.parquet")
    np.testing.assert_allclose(targets["weight_target"], 0.25, rtol=0, atol=1e-10)
    assert any(row["target_count"] == 1 for row in direct_calls if row["event"] == "target")
    assert any(row["target_count"] == 4 for row in direct_calls if row["event"] == "target")
    # The deliberately blocked sell survives the first liquidation execution;
    # the next rebalance has an ideal A position that limit-up prevents buying.
    positions = pd.read_csv(output_left / "positions_4.csv")
    position_dates = pd.to_datetime(positions["date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(left.parent / "fixture/market.parquet")["date"]).unique()))
    assert positions.loc[position_dates == dates[4], "symbol"].tolist() == ["000003.XSHE"]
    assert "000001.XSHE" not in positions.loc[position_dates == dates[5], "symbol"].tolist()
    target_weights = pd.read_parquet(output_left / "target_weights_4.parquet")
    target_dates = pd.to_datetime(target_weights["date"]).dt.normalize()
    assert "000001.XSHE" in target_weights.loc[target_dates == dates[5], "symbol"].tolist()
    return {"compared_files": files, "events": observed, "decision_rows": len(targets),
            "max_weight_abs_error": max_weight, "max_daily_return_abs_error": max_return,
            "max_cash_weight_abs_error": max_cash}


def verify_long_horizon(work_dir: Path, *, target_days: int = 2000, nofile: int = 256,
                        max_shared_bytes: int | None = None) -> dict:
    import importlib.metadata
    import optimizer

    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=False)
    dates, expected = _long_fixture(work_dir / "fixture", target_days)
    paths = {mode: work_dir / mode for mode in ("inprocess", "shared_memory")}
    for mode, path in paths.items():
        _long_config(path, mode, dates, max_shared_bytes if mode == "shared_memory" else None)
    socket_path = paths["shared_memory"] / "ipc/service.sock"
    result = {"experiment": "real CLI long-horizon parity", "target_days": target_days,
              "market_days": len(dates), "nofile": nofile, "expected_events": expected,
              "max_shared_bytes": max_shared_bytes,
              "environment": {"python": platform.python_version(), "kernel": platform.release(),
                              "optimizer_file": optimizer.__file__,
                              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "pandas", "polars")},
                              "installed_source": json.loads(importlib.metadata.distribution("optimizer").read_text("direct_url.json") or "{}")},
              "runs": {}, "status": "failed"}
    with (work_dir / "service.log").open("w") as log:
        service = subprocess.Popen(_service_command(socket_path, nofile, max_shared_bytes), cwd=work_dir, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_socket(service, socket_path)
            result["service_baseline"] = _snapshot(service.pid)
            for mode, path in paths.items():
                result["runs"][mode] = _run_monitored_cli(path, nofile, service)
            if all(row["returncode"] == 0 and row["run_status"]["complete"] is True for row in result["runs"].values()):
                result["comparison"] = _compare_long(paths["inprocess"], paths["shared_memory"], expected)
                for mode, row in result["runs"].items():
                    before, after = row["child_resources"]["baseline"], row["child_resources"]["final"]
                    assert row["peaks"]["client"]["fd"] - before["fd"] <= 16, (mode, row)
                    assert after["optimizer_maps"] == 0, (mode, after)
                    # Imports and Matplotlib retain ordinary runtime files;
                    # inspect optimizer handles independently of that noise.
                    assert after["optimizer_fds"] == 0, (mode, after)
                    assert row["peaks"]["service"]["fd"] - result["service_baseline"]["fd"] <= 16, row
                    assert row["service_after_client_exit"]["optimizer_maps"] == 0, row
                    assert row["service_after_client_exit"]["optimizer_fds"] == 0, row
                    assert row["peaks"]["client"]["optimizer_maps"] <= 1, row
                    assert row["peaks"]["service"]["optimizer_maps"] <= 1, row
                final_fds = [row["child_resources"]["final"]["fd"] for row in result["runs"].values()]
                assert abs(final_fds[0] - final_fds[1]) <= 1, final_fds
                result["status"] = "passed"
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        finally:
            service.terminate()
            service.wait(timeout=10)
    result["service_exit"] = service.returncode
    result["socket_residual"] = socket_path.exists()
    if result["socket_residual"]:
        result["status"] = "failed"
    _write_json(work_dir / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("retained-baseline", "long-cli", "_retained", "_service", "_cli"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--nofile", type=int, default=256)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--target-days", type=int, default=2000)
    parser.add_argument("--max-shared-bytes", type=int)
    parser.add_argument("--service-pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "_service":
        _limit(args.nofile)
        sys.argv = ["optimizer-service", "--socket", str(args.socket), "--log-level", "WARNING"]
        if args.max_shared_bytes is not None:
            sys.argv += ["--max-shared-bytes", str(args.max_shared_bytes)]
        runpy.run_module("optimizer.transport.shm.service", run_name="__main__")
    elif args.command == "_retained":
        _retained_worker(args.socket, args.requests, args.nofile, args.service_pid, args.output)
    elif args.command == "_cli":
        _cli_worker(args.work_dir, args.nofile)
    else:
        verify = retained_baseline if args.command == "retained-baseline" else verify_long_horizon
        kwargs = {"nofile": args.nofile}
        if args.command == "long-cli":
            kwargs["target_days"] = args.target_days
            kwargs["max_shared_bytes"] = args.max_shared_bytes
        if args.work_dir:
            result = verify(args.work_dir.resolve(), **kwargs)
        else:
            with tempfile.TemporaryDirectory(prefix="eq-shm-pressure-") as temporary:
                result = verify(Path(temporary) / "verification", **kwargs)
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, indent=2))
        if args.command == "long-cli" and result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
