"""Run both optimizer backends through the real backtest CLI on synthetic data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import polars as pl
import yaml


REPO = Path(__file__).resolve().parents[1]
SIZES = (1, 2, 4, 8)
TRANSPORT_FIELDS = {
    "protocol_version", "session_id", "service_epoch", "buffer_id", "generation",
    "shared_bytes", "staging_bytes", "copy_bytes", "wait_ms", "prepare_ms",
    "mapping_validation_ms", "model_time_ms", "config_signature",
}


def _fixture(root: Path) -> pd.DatetimeIndex:
    root.mkdir()
    (root / "predictions").mkdir()
    (root / "market_cache").mkdir()
    dates = pd.bdate_range("2026-09-01", periods=12)
    rng = np.random.default_rng(20261015)
    prices = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.015, (len(dates), 4)), axis=0)
    market, predictions = [], []
    for day, stamp in enumerate(dates):
        # Reverse input order deliberately; canonical symbol order breaks score ties.
        for asset in reversed(range(4)):
            symbol = f"{asset + 1:06d}.XSHE"
            market.append({
                "date": stamp.date(), "symbol": symbol, "turnover": 1e7,
                "log_size": 10.0 - asset, "size_rank": asset + 1,
                "industry": "synthetic", "index": "TEST", "ret": 0.0,
                "is_limit_up": day == 5 and asset == 0, "is_limit_down": False,
                "listed_Satisfied": True, "is_ST": day == 3 or (day == 1 and asset != 2),
                "normal_days": 20, "vwap30": float(prices[day, asset]),
                "close_ex": float(prices[day, asset] * (1.0 + 0.001 * (asset - 1))),
            })
            if day not in (2, 7):
                predictions.append({
                    "date": stamp.date(), "symbol": symbol,
                    "prediction": 1.0 if day in (0, 5) else float((asset + day) % 4),
                })
    pl.DataFrame(market).write_parquet(root / "market.parquet")
    pl.DataFrame(predictions).write_parquet(root / "predictions" / "signals.parquet")
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "close": 100.0,
        "pct_change": np.linspace(-0.001, 0.001, len(dates)),
    }).to_csv(root / "market_cache" / "TEST.csv", index=False)
    return dates


def _config(run_dir: Path, mode: str, *, fail: bool = False) -> None:
    run_dir.mkdir()
    config = {
        "frequency": "daily", "start": "2026-09-01", "end": "2026-09-30",
        "strategy_modes": ["long_only"], "weight_mode": "equal",
        "port_sizes": list(SIZES), "pool_size": 10, "thresh_out_buffer": 0,
        "close_on_size_drop": True, "exclude_period": None, "use_pool_cache": False,
        "market_cache_dir": "../fixture/market_cache", "benchmark_symbol": "TEST",
        "freq_config": {"daily": {
            "market_data": "../fixture/market.parquet", "preds_dir": "../fixture/predictions",
            "prediction_sources": ["signals.parquet"], "trade_on_next_bar": True,
        }},
        "optimizer": {"enabled": True, "mode": mode, "socket_path": "ipc/service.sock"},
    }
    if fail and mode == "inprocess":
        config["optimizer"]["max_assets"] = 1
    (run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _run_cli(run_dir: Path, *, success: bool = True) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg", "POLARS_MAX_THREADS": "4"}
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, str(REPO / "run.py"), str(run_dir)], cwd=REPO,
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120,
        )
    if (result.returncode == 0) != success:
        raise AssertionError(f"unexpected exit {result.returncode}: {run_dir / 'run.log'}")
    status = json.loads((run_dir / "optimizer_run_status.json").read_text())
    assert status["complete"] is success, status
    assert status["status"] == ("success" if success else "failed"), status
    if not success:
        assert not list(run_dir.glob("output_*/metrics_*.csv"))


def _calls(run_dir: Path, mode: str) -> tuple[list[dict], dict[str, str]]:
    path = run_dir / "output_long_10_daily" / "optimizer_calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    ids = {}
    for row in calls:
        if row["event"] == "target":
            assert row["status"] == "feasible"
            assert row["copy_bytes"] == row["staging_bytes"] == 0
            expected_bytes = row["target_count"] * 8 if mode == "shared_memory" else 0
            assert row["shared_bytes"] == expected_bytes
            assert row["target_budget"] == row["target_count"] / row["portfolio_size"]
            ids[row["request_id"]] = f"{row['portfolio_id']}:{row['decision_as_of']}"
        assert pd.Timestamp(row["decision_as_of"]) < pd.Timestamp(row["planned_execution_at"])
        # Signature includes the transport protocol; check it before normalizing.
        assert isinstance(row["config_signature"], str) and len(row["config_signature"]) == 64
    return calls, ids


def _normalize(frame: pd.DataFrame, ids: dict[str, str]) -> pd.DataFrame:
    frame = frame.copy()
    if "request_id" in frame:
        nonempty = frame["request_id"].dropna()
        assert set(nonempty).issubset(ids)
        frame["request_id"] = frame["request_id"].map(ids)
    return frame


def _compare(left: Path, right: Path, dates: pd.DatetimeIndex) -> dict:
    direct_calls, direct_ids = _calls(left, "inprocess")
    shm_calls, shm_ids = _calls(right, "shared_memory")
    assert {row["event"] for row in direct_calls} == {"target", "no_signal", "liquidate"}
    assert len(direct_calls) == len(shm_calls) == (len(dates) - 1) * len(SIZES)
    for direct, shm in zip(direct_calls, shm_calls):
        for row, ids in ((direct, direct_ids), (shm, shm_ids)):
            if row["request_id"]:
                row["request_id"] = ids[row["request_id"]]
        assert {k: v for k, v in direct.items() if k not in TRANSPORT_FIELDS} == {
            k: v for k, v in shm.items() if k not in TRANSPORT_FIELDS
        }

    left_output, right_output = (path / "output_long_10_daily" for path in (left, right))
    files = sorted(path.name for path in left_output.iterdir() if path.suffix in {".csv", ".parquet"})
    assert files == sorted(path.name for path in right_output.iterdir() if path.suffix in {".csv", ".parquet"})
    max_weight_error = max_return_error = 0.0
    for name in files:
        read = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        direct = _normalize(read(left_output / name), direct_ids)
        shm = _normalize(read(right_output / name), shm_ids)
        tolerance = 1e-10 if "target" in name else 1e-8
        pd.testing.assert_frame_equal(direct, shm, check_exact=False, rtol=0, atol=tolerance)
        for column in direct.select_dtypes(include="number"):
            error = float((direct[column] - shm[column]).abs().max())
            if "weight" in column:
                max_weight_error = max(max_weight_error, error)
            if "daily_" in column or name.startswith("returns_"):
                max_return_error = max(max_return_error, error)

    targets = pd.read_parquet(left_output / "decision_targets.parquet")
    for size in SIZES:
        selected = targets[targets["portfolio_size"] == size]
        np.testing.assert_allclose(selected["weight_target"], 1 / size, rtol=0, atol=1e-10)
        first = selected[selected["decision_as_of"].dt.normalize() == dates[0].tz_localize("Asia/Shanghai")]
        assert first["symbol"].tolist() == [f"{n + 1:06d}.XSHE" for n in range(min(size, 4))]
        positions = pd.read_csv(left_output / f"positions_{size}.csv")
        assert not (pd.to_datetime(positions["date"]).dt.normalize() == dates[0]).any()
        assert not (pd.to_datetime(positions["date"]).dt.normalize() == dates[4]).any()
        # Day 2 executes a one-name target, with the rest of the 1/N budget in cash.
        partial = positions[pd.to_datetime(positions["date"]).dt.normalize() == dates[2]]
        assert partial["symbol"].tolist() == ["000003.XSHE"]
        assert partial["weight_actual"].sum() < 1.0 / size + 0.05
    assert max_weight_error <= 1e-10 and max_return_error <= 1e-8
    return {
        "compared_files": files, "business_events": len(direct_calls),
        "decision_rows": len(targets), "portfolio_sizes": list(SIZES),
        "max_weight_abs_error": max_weight_error, "max_daily_return_abs_error": max_return_error,
        "events": sorted({row["event"] for row in direct_calls}),
        "normalized_fields": sorted(TRANSPORT_FIELDS | {"request_id"}),
        "signature_rule": "SHA256 of business settings plus protocol_version; protocol differs by backend",
    }


def verify(work_dir: Path) -> dict:
    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=False)
    dates = _fixture(work_dir / "fixture")
    direct, shm = (work_dir / mode for mode in ("inprocess", "shared_memory"))
    _config(direct, "inprocess")
    _config(shm, "shared_memory")
    socket_path = shm / "ipc/service.sock"
    with (work_dir / "service.log").open("w", encoding="utf-8") as log:
        service = subprocess.Popen(
            [sys.executable, "-m", "optimizer.transport.shm.service", "--socket", str(socket_path)],
            cwd=work_dir, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while not socket_path.exists():
                if service.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError(f"service failed to start: {work_dir / 'service.log'}")
                time.sleep(0.01)
            _run_cli(direct)
            _run_cli(shm)
        finally:
            service.terminate()
            service.wait(timeout=10)
    assert not socket_path.exists(), "service left a stale socket"
    report = _compare(direct, shm, dates)
    for mode in ("inprocess", "shared_memory"):
        failed = work_dir / f"failed_{mode}"
        _config(failed, mode, fail=True)
        _run_cli(failed, success=False)
    report.update({"status": "passed", "seed": 20261015, "failed_runs_incomplete": True})
    (work_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="New directory to retain fixtures, configs, logs and report")
    args = parser.parse_args()
    if args.work_dir:
        report = verify(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eq-dual-") as tmp:
            report = verify(Path(tmp) / "verification")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
