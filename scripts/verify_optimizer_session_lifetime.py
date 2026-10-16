"""Verify real CLI slow paths, per-portfolio SHM sessions, and release ordering.

The worker runs the actual run.run() entry point. Wrappers only record calls or
delay real loading/simulation/reporting; optimizer results and service are real.
Use --real-default --hold-seconds 31 for the separate deployment-time check.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from verify_optimizer_pressure import _limit, _merge_peak, _snapshot, _wait_socket, _write_json


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SIZES = (1, 2, 4, 8)


def _nofile_limit(pid: int) -> int:
    for line in (Path("/proc") / str(pid) / "limits").read_text().splitlines():
        if line.startswith("Max open files"):
            return int(line.split()[3])
    raise AssertionError("process omitted RLIMIT_NOFILE")


def _worker(args: argparse.Namespace) -> None:
    _limit(args.nofile)
    os.environ["EQ_BACKTEST_RUN_DIR"] = str(args.work_dir)
    sys.path.insert(0, str(REPO))
    import run as runner
    from optimizer.transport.shm import ShmOptimizerClient
    from optimizer.transport.shm import client as transport

    active: set[str] = set()
    events: list[dict] = []
    current = {"port_size": None}
    baseline = _snapshot(os.getpid())
    report = {"status": "failed", "baseline": baseline, "events": events,
              "nofile_soft": _nofile_limit(os.getpid())}
    with (args.work_dir / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
        def event(kind: str, **details) -> None:
            row = {
                "event": kind, "monotonic": time.monotonic(), "unix_ns": time.time_ns(),
                "port_size": current["port_size"], "active_sessions": sorted(active),
                "client": _snapshot(os.getpid()), "service": _snapshot(args.service_pid),
                **details,
            }
            events.append(row)
            timeline.write(json.dumps(row) + "\n")
            timeline.flush()

        real_connect, real_close = ShmOptimizerClient.connect, ShmOptimizerClient.close
        real_send = transport._send_packet
        real_build, real_prepare = runner.build_pool, runner.prepare_optimizer_targets
        real_generate, real_release = runner.generate_portfolio, runner.close_optimizer_plan
        real_plot, real_write = runner.plot_holding_periods, runner._write_positions_csv

        def connect(client) -> None:
            event("connect_begin")
            real_connect(client)
            active.add(client.session_id)
            event("connected", session_id=client.session_id)

        def close(client) -> None:
            session = client.session_id
            if session:
                event("close_begin", session_id=session)
            real_close(client)
            active.discard(session)
            if session:
                event("closed", session_id=session)

        def send(sock, message, *positional, **keywords):
            event("control_send", message_type=message.get("type"),
                  session_id=message.get("session_id"), buffer_id=message.get("buffer_id")
                  or (message.get("buffer") or {}).get("buffer_id"),
                  request_id=(message.get("request") or {}).get("request_id"))
            return real_send(sock, message, *positional, **keywords)

        def build(*positional, **keywords):
            event("load_begin")
            time.sleep(args.load_seconds)
            pool = real_build(*positional, **keywords)
            event("load_end")
            return pool

        def prepare(*positional, **keywords):
            current["port_size"] = keywords["port_size"]
            event("prepare_begin")
            plan = real_prepare(*positional, **keywords)
            event("prepared", lease_count=len(plan.leases), calls=len(plan.calls),
                  buffers=[lease.descriptor for lease in plan.leases],
                  copy_bytes=sum(lease.copy_bytes for lease in plan.leases))
            return plan

        def generate(*positional, **keywords):
            current["port_size"] = keywords["port_size"]
            event("simulate_begin")
            result = real_generate(*positional, **keywords)
            event("simulate_end")
            if keywords.get("optimizer_plan") is not None and current["port_size"] == SIZES[0]:
                event("hold_begin", requested_seconds=args.hold_seconds)
                time.sleep(args.hold_seconds)
                plan = keywords["optimizer_plan"]
                weight_sum = sum(float(lease.weights.sum()) for lease in plan.leases)
                assert math.isfinite(weight_sum)
                assert all(not lease.weights.flags.writeable for lease in plan.leases)
                event("hold_end", weight_sum=weight_sum, readonly=True)
            if args.fail_last and current["port_size"] == SIZES[-1]:
                event("service_termination_begin")
                os.kill(args.service_pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while Path(runner.cfg.OPTIMIZER["socket_path"]).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("service did not stop after SIGTERM")
                    time.sleep(0.005)
                event("service_terminated")
            return result

        def release(plan, primary=None):
            event("release_begin")
            try:
                real_release(plan, primary)
            except BaseException as exc:
                event("release_failed", error_type=type(exc).__name__, error=str(exc))
                raise
            event("released")

        def write(*positional, **keywords):
            event("report_begin")
            return real_write(*positional, **keywords)

        def plot(*positional, **keywords):
            event("plot_begin")
            time.sleep(args.report_seconds)
            result = real_plot(*positional, **keywords)
            event("plot_end")
            return result

        ShmOptimizerClient.connect, ShmOptimizerClient.close = connect, close
        transport._send_packet = send
        runner.build_pool, runner.prepare_optimizer_targets = build, prepare
        runner.generate_portfolio, runner.close_optimizer_plan = generate, release
        runner.plot_holding_periods, runner._write_positions_csv = plot, write
        try:
            runner.run()
            report["status"] = "passed"
        except BaseException as exc:
            report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            gc.collect()
            event("worker_final")
            report["final"] = _snapshot(os.getpid())
            report["service_final"] = _snapshot(args.service_pid)
            _write_json(args.work_dir / "worker.json", report)


def _service(args: argparse.Namespace) -> None:
    import logging
    from optimizer.transport.shm.server import OptimizerServer

    _limit(args.nofile)
    logging.basicConfig(level=logging.INFO)
    server = OptimizerServer(args.socket, handshake_timeout_s=args.handshake_seconds)
    signal.signal(signal.SIGTERM, lambda *_: server.shutdown())
    signal.signal(signal.SIGINT, lambda *_: server.shutdown())
    server.serve_forever()


def _run_case(path: Path, args: argparse.Namespace, *, fail_last: bool = False) -> dict:
    socket_path = path / "ipc/service.sock"
    service_command = [sys.executable, "-m", "optimizer.transport.shm.service", "--socket", str(socket_path)]
    if not args.real_default:
        service_command = [sys.executable, str(SCRIPT), "_service", "--socket", str(socket_path),
                           "--handshake-seconds", str(args.handshake_seconds), "--nofile", str(args.nofile)]
    result = {"service_command": service_command, "fail_last": fail_last, "run_dir": str(path),
              "nofile_requested": args.nofile}
    with (path / "service.log").open("w", encoding="utf-8") as service_log:
        service = subprocess.Popen(service_command, cwd=path, stdout=service_log, stderr=subprocess.STDOUT,
                                   preexec_fn=lambda: _limit(args.nofile))
        try:
            _wait_socket(service, socket_path)
            result["service_baseline"] = _snapshot(service.pid)
            result["service_nofile_soft"] = _nofile_limit(service.pid)
            command = [sys.executable, str(SCRIPT), "_worker", "--work-dir", str(path),
                       "--service-pid", str(service.pid), "--nofile", str(args.nofile),
                       "--load-seconds", str(args.load_seconds), "--report-seconds", str(args.report_seconds),
                       "--hold-seconds", str(args.hold_seconds if not fail_last else 0.3)]
            if fail_last:
                command.append("--fail-last")
            peaks = {"client": {}, "service": {}, "samples": 0, "sample_interval_seconds": 0.005}
            with (path / "run.log").open("w", encoding="utf-8") as log:
                worker = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "MPLBACKEND": "Agg", "POLARS_MAX_THREADS": "4"})
                deadline = time.monotonic() + args.hold_seconds + 180
                try:
                    while worker.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError("session lifetime CLI exceeded its bounded runtime")
                        _merge_peak(peaks["client"], _snapshot(worker.pid))
                        _merge_peak(peaks["service"], _snapshot(service.pid))
                        peaks["samples"] += 1
                        time.sleep(0.005)
                finally:
                    if worker.poll() is None:
                        worker.kill()
                    worker.wait(timeout=10)
            result.update(returncode=worker.returncode, peaks=peaks,
                          worker=json.loads((path / "worker.json").read_text()),
                          run_status=json.loads((path / "optimizer_run_status.json").read_text()))
            # Observe cleanup while the successful service is still alive.
            deadline = time.monotonic() + 2
            while True:
                final = _snapshot(service.pid)
                if not final.get("optimizer_fds", 0) and not final.get("optimizer_maps", 0):
                    break
                if time.monotonic() > deadline:
                    break
                time.sleep(0.005)
            result["service_after_worker"] = final
        finally:
            if service.poll() is None:
                service.terminate()
            service.wait(timeout=10)
    result.update(service_exit=service.returncode, socket_residual=socket_path.exists())
    assert not result["socket_residual"], result
    _write_json(path / "case.json", result)
    return result


def _assert_lifecycle(case: dict, *, fail_last: bool = False) -> dict:
    worker, status = case["worker"], case["run_status"]
    assert case["service_exit"] == 0, case["service_exit"]
    assert (case["returncode"] == 0) is not fail_last, case
    assert status["complete"] is not fail_last, status
    assert status["status"] == ("failed" if fail_last else "success"), status
    assert worker["nofile_soft"] == case["service_nofile_soft"] == case["nofile_requested"]
    events = worker["events"]
    load_end = next(row for row in events if row["event"] == "load_end")
    connections = [row for row in events if row["event"] == "connected"]
    closes = [row for row in events if row["event"] == "closed"]
    assert len(connections) == len(closes) == len(SIZES), (connections, closes)
    sessions = [row["session_id"] for row in connections]
    assert len(set(sessions)) == len(SIZES), sessions
    assert load_end["monotonic"] < connections[0]["monotonic"]
    for row in events:
        if row["event"] in {"load_begin", "load_end", "report_begin", "plot_begin", "plot_end", "worker_final"}:
            assert row["active_sessions"] == [], row
    for index, size in enumerate(SIZES):
        connected, closed = connections[index], closes[index]
        if index:
            assert closes[index - 1]["monotonic"] < connected["monotonic"]
        prepared = next(row for row in events if row["event"] == "prepared" and row["port_size"] == size)
        assert connected["monotonic"] < prepared["monotonic"] < closed["monotonic"]
        assert prepared["lease_count"] == 1, prepared
        assert prepared["copy_bytes"] == 0, prepared
        failure = fail_last and index == len(SIZES) - 1
        kind = "release_failed" if failure else "released"
        released = next(row for row in events if row["event"] == kind and row["port_size"] == size)
        release_begin = next(row for row in events if row["event"] == "release_begin" and row["port_size"] == size)
        assert prepared["monotonic"] < release_begin["monotonic"] < closed["monotonic"]
        if not failure:
            assert released["monotonic"] < closed["monotonic"]
        for row in events:
            if row["event"] == "control_send" and connected["monotonic"] <= row["monotonic"] < closed["monotonic"]:
                assert row["session_id"] == connected["session_id"], row
    hold_begin = next(row for row in events if row["event"] == "hold_begin")
    hold_end = next(row for row in events if row["event"] == "hold_end")
    idle = hold_end["monotonic"] - hold_begin["monotonic"]
    assert idle >= hold_begin["requested_seconds"]
    assert hold_begin["client"]["optimizer_maps"] == 1
    assert hold_begin["service"]["optimizer_maps"] == 1
    assert hold_end["client"]["optimizer_maps"] == 1
    assert hold_end["service"]["optimizer_maps"] == 1
    assert hold_end["readonly"]
    assert not any(row["event"] == "control_send" and hold_begin["monotonic"] < row["monotonic"] < hold_end["monotonic"] for row in events)
    if fail_last:
        assert status["cleanup_errors"], status
        assert status["error_type"] == worker["error_type"], (status, worker)
        assert not list(Path(case["run_dir"]).glob("output_*/metrics_*.csv"))
    else:
        assert case["service_after_worker"]["optimizer_fds"] == 0, case
        assert case["service_after_worker"]["optimizer_maps"] == 0, case
    assert worker["final"]["optimizer_maps"] == worker["final"]["optimizer_fds"] == 0, worker["final"]
    assert case["peaks"]["client"]["optimizer_maps"] <= 1, case["peaks"]
    assert case["peaks"]["service"]["optimizer_maps"] <= 1, case["peaks"]
    assert case["peaks"]["client"]["fd"] - worker["baseline"]["fd"] <= 16, case["peaks"]
    assert case["peaks"]["service"]["fd"] - case["service_baseline"]["fd"] <= 16, case["peaks"]
    return {"sessions": sessions, "arena_idle_seconds": idle, "portfolio_sizes": list(SIZES)}


def verify(work_dir: Path, args: argparse.Namespace) -> dict:
    from verify_optimizer_modes import _fixture, _config, _run_cli, _compare
    import optimizer

    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=False)
    report = {"status": "failed", "real_default_service": args.real_default,
              "hold_seconds": args.hold_seconds, "load_seconds": args.load_seconds,
              "report_seconds": args.report_seconds, "short_handshake_seconds": args.handshake_seconds,
              "nofile": args.nofile, "environment": {
                  "python": platform.python_version(), "kernel": platform.release(),
                  "optimizer_file": optimizer.__file__,
                  "installed_source": json.loads(importlib.metadata.distribution("optimizer").read_text("direct_url.json") or "{}"),
              }}
    try:
        assert "site-packages" in Path(optimizer.__file__).parts, optimizer.__file__
        dates = _fixture(work_dir / "fixture")
        direct, shm, failure = (work_dir / name for name in ("inprocess", "shared_memory", "failed_shared_memory"))
        _config(direct, "inprocess")
        _config(shm, "shared_memory")
        _config(failure, "shared_memory")
        _run_cli(direct)
        report["success"] = _run_case(shm, args)
        report["success_assertions"] = _assert_lifecycle(report["success"])
        report["comparison"] = _compare(direct, shm, dates)
        report["failure"] = _run_case(failure, args, fail_last=True)
        report["failure_assertions"] = _assert_lifecycle(report["failure"], fail_last=True)
        assert not list(failure.glob("output_*/metrics_*.csv"))
        report["status"] = "passed"
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    _write_json(work_dir / "comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="verify", choices=("verify", "_worker", "_service"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--service-pid", type=int)
    parser.add_argument("--nofile", type=int, default=256)
    parser.add_argument("--handshake-seconds", type=float, default=0.15)
    parser.add_argument("--load-seconds", type=float, default=0.3)
    parser.add_argument("--report-seconds", type=float, default=0.3)
    parser.add_argument("--hold-seconds", type=float, default=0.3)
    parser.add_argument("--real-default", action="store_true")
    parser.add_argument("--fail-last", action="store_true")
    args = parser.parse_args()
    if min(args.load_seconds, args.report_seconds, args.hold_seconds, args.handshake_seconds) <= 0:
        parser.error("all delay/timeout values must be positive")
    if args.real_default and args.hold_seconds <= 30:
        parser.error("--real-default requires --hold-seconds greater than 30")
    if args.command == "_worker":
        _worker(args)
    elif args.command == "_service":
        _service(args)
    else:
        if args.work_dir:
            result = verify(args.work_dir, args)
        else:
            with tempfile.TemporaryDirectory(prefix="eq-session-lifetime-") as temporary:
                result = verify(Path(temporary) / "verification", args)
        print(json.dumps({key: value for key, value in result.items() if key not in {"success", "failure"}}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
