"""Twenty-five sequential max_score decisions through both real offline CLIs.

Engineering fixtures and parameters are synthetic, fixed before either run, and
are never strategy defaults. The service is a separate local Linux process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import yaml


REPO = Path(__file__).resolve().parents[1]


def fixture(root: Path, *, days: int = 26) -> pd.DatetimeIndex:
    root.mkdir()
    (root / "predictions").mkdir()
    (root / "market_cache").mkdir()
    dates = pd.bdate_range("2026-09-01", periods=days)
    market, predictions, size_rows, industry_rows = [], [], [], []
    for day, stamp in enumerate(dates):
        for asset in range(6):
            symbol = f"{asset + 1:06d}.XSHE"
            close = 10 * (1 + 0.001 * day * (asset + 1))
            market.append({"date": stamp.date(), "symbol": symbol, "turnover": 1e6,
                           "log_size": 6.0-asset, "size_rank": asset+1, "industry": str(1+asset%2),
                           "index": "TEST", "ret": 0.0, "is_limit_up": day == 1 and asset == 5,
                           "is_limit_down": day == 5, "listed_Satisfied": True,
                           "is_ST": day == 6 and asset == 5, "normal_days": 20,
                           "vwap30": close * (1.002 if asset % 2 else 0.999), "close_ex": close})
            predictions.append({"date": stamp.date(), "symbol": symbol,
                                "prediction": float((asset + day) % 6) + asset * 0.001})
            size_rows.append({"date": stamp.date(), "symbol": symbol, "value": -1+asset*0.4})
            industry_rows.append({"date": stamp.date(), "symbol": symbol, "value": 1+asset%2})
    for name, records in (("market", market), ("size", size_rows), ("industry", industry_rows)):
        pl.DataFrame(records).write_parquet(root / f"{name}.parquet")
    pl.DataFrame(predictions).write_parquet(root / "predictions" / "signals.parquet")
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": 100.0,
                  "pct_change": np.zeros(len(dates))}).to_csv(root / "market_cache" / "TEST.csv", index=False)
    return dates


def source(name: str) -> dict:
    metadata = {"exposure_id": f"synthetic-{name}", "source": "deterministic-fixture", "version": "1",
                "units": "raw", "transformation": "identity", "semantic_status": "explicit"}
    if name == "industry":
        metadata.update(classification="synthetic-industry-v1", level="1")
    return {"path": f"../fixture/{name}.parquet", "date_column": "date", "symbol_column": "symbol",
            "value_column": "value", "available_at_column": None, "metadata": metadata}


def config(run_dir: Path, mode: str, dates: pd.DatetimeIndex, *, failure: str | None = None) -> None:
    run_dir.mkdir()
    cfg = {
        "frequency": "daily", "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "strategy_modes": ["long_only"], "weight_mode": "equal", "port_sizes": [1, 800],
        "pool_size": 6000, "thresh_out_buffer": 0, "exclude_period": None, "use_pool_cache": False,
        "portfolio_initial_value": 10000.0, "cost_per_turnover": 0.00045,
        "market_cache_dir": "../fixture/market_cache", "benchmark_symbol": "TEST",
        "freq_config": {"daily": {"market_data": "../fixture/market.parquet",
                                    "preds_dir": "../fixture/predictions", "prediction_sources": ["signals.parquet"],
                                    "trade_on_next_bar": True}},
        "optimizer": {"enabled": True, "mode": mode, "schema_version": "1.2", "max_assets": 6000,
                      "model": {"type": "max_score", "version": "1", "config": {}},
                      "socket_path": "ipc/service.sock",
                      "max_score": {
                          "purpose": "engineering_acceptance", "target_budget": 0.9, "max_weight": 0.3,
                          "size_constraint": {"lower": -0.15, "upper": 0.15, "source": source("size")},
                          "industry_constraint": {"bounds": [{"industry_id": "1", "lower": 0.35, "upper": 0.55},
                                                                {"industry_id": "2", "lower": 0.35, "upper": 0.55}],
                                                  "source": source("industry")},
                          "turnover_constraint": {"definition": "stock_half_l1", "limit": 0.46},
                          "candidate_policy": "finite_score_eligible", "missing_valuation_policy": "carry_last",
                          "no_signal_policy": "error", "source_point_in_time_status": "synthetic",
                          "price_convention": "consistent_share_prices"}},
    }
    if failure == "capacity":
        cfg["optimizer"]["max_assets"] = 1
    if failure == "turnover":
        cfg["optimizer"]["max_score"]["turnover_constraint"]["limit"] = 0.1
    (run_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_cli(run_dir: Path, *, success: bool = True) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/max-score-mpl", "POLARS_MAX_THREADS": "4"}
    with (run_dir / "run.log").open("w") as log:
        result = subprocess.run([sys.executable, str(REPO / "run.py"), str(run_dir)], cwd=REPO,
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if (result.returncode == 0) != success:
        raise AssertionError(f"unexpected exit={result.returncode}; see {run_dir / 'run.log'}")
    status = json.loads((run_dir / "optimizer_run_status.json").read_text())
    assert status["complete"] is success, status
    if not success:
        assert status["status"] == "failed"
        assert not list(run_dir.glob("output_*/metrics_*.csv"))


def normalized(frame: pd.DataFrame, ids: dict) -> pd.DataFrame:
    frame = frame.copy()
    if "request_id" in frame:
        assert set(frame["request_id"].dropna()).issubset(ids)
        frame["request_id"] = frame["request_id"].map(ids)
    return frame


def compare(left: Path, right: Path, dates: pd.DatetimeIndex) -> dict:
    outputs = [p / "output_long_6000_daily" for p in (left, right)]
    calls = [[json.loads(s) for s in (p / "optimizer_calls.jsonl").read_text().splitlines()] for p in outputs]
    ids = [{r["request_id"]: f"{r['portfolio_id']}:{r['decision_as_of']}" for r in records
            if r["event"] == "target"} for records in calls]
    targets = [[r for r in records if r["event"] == "target"] for records in calls]
    assert len(targets[0]) == len(targets[1]) == (len(dates)-1)*2
    for direct, shm in zip(*targets):
        assert direct["universe_count"] == shm["universe_count"] == 6
        assert direct["target_budget"] == shm["target_budget"] == 0.9
        assert direct["diagnostics"]["optimality_verified"] is True
        assert shm["diagnostics"]["optimality_verified"] is True
    names = sorted(p.name for p in outputs[0].iterdir() if p.suffix in {".csv", ".parquet"})
    assert names == sorted(p.name for p in outputs[1].iterdir() if p.suffix in {".csv", ".parquet"})
    for name in names:
        read = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        frames = [normalized(read(p / name), mapping) for p, mapping in zip(outputs, ids)]
        pd.testing.assert_frame_equal(*frames, check_exact=False, rtol=0, atol=1e-10)
    actual = pd.read_parquet(outputs[0] / "account_state_1.parquet")
    positions = pd.read_csv(outputs[0] / "positions_1.csv")
    positions["date"] = pd.to_datetime(positions["date"])
    values = (positions["shares"] * positions["valuation_price"]).groupby(positions["date"]).sum()
    values = values.reindex(pd.DatetimeIndex(actual["date"]), fill_value=0).to_numpy()
    np.testing.assert_allclose(actual["cash"] + values, actual["nav"], rtol=0, atol=1e-8)
    trades = pd.read_parquet(outputs[0] / "executed_trades_1.parquet")
    assert {"buy_blocked", "sell_blocked"}.issubset(set(trades["status"]))
    decisions = pd.read_parquet(outputs[0] / "decision_targets.parquet")
    # Labels intentionally span the old ranking count; neither controls U or B.
    left_book = decisions[decisions["portfolio_size"] == 1]["weight_target"].to_numpy()
    right_book = decisions[decisions["portfolio_size"] == 800]["weight_target"].to_numpy()
    np.testing.assert_array_equal(left_book, right_book)
    prior_target = decisions[decisions["portfolio_size"] == 1].groupby("decision_as_of")["weight_target"].apply(list)
    actual_by_day = decisions[decisions["portfolio_size"] == 1].groupby("decision_as_of")["weight_actual"].apply(list)
    assert any(not np.allclose(a, b) for a, b in zip(prior_target.iloc[:-1], actual_by_day.iloc[1:]))
    return {"status": "passed", "purpose": "engineering_acceptance", "source_point_in_time_status": "synthetic",
            "decisions_per_portfolio": len(dates)-1, "portfolio_labels": [1, 800], "universe_count": 6,
            "compared_files": names, "account_conservation": True, "restricted_executions": sorted(set(trades["status"])),
            "raw_score_objective": True, "actual_target_divergence": True}


def _verify(work_dir: Path) -> dict:
    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=False)
    dates = fixture(work_dir / "fixture")
    direct, shm = (work_dir / name for name in ("inprocess", "shared_memory"))
    for path, mode in ((direct, "inprocess"), (shm, "shared_memory")):
        config(path, mode, dates)
    socket_path = shm / "ipc/service.sock"
    with (work_dir / "service.log").open("w") as log:
        service = subprocess.Popen([sys.executable, "-m", "optimizer.transport.shm.service", "--socket", str(socket_path),
                                    "--max-assets", "6000"], cwd=work_dir, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+15
            while not socket_path.exists():
                if service.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError(f"service startup failed: {work_dir / 'service.log'}")
                time.sleep(0.02)
            run_cli(direct)
            run_cli(shm)
            for failure in ("turnover",):
                for mode in ("inprocess", "shared_memory"):
                    failed = work_dir / f"failed_{mode}_{failure}"
                    config(failed, mode, dates, failure=failure)
                    if mode == "shared_memory":
                        failed_socket = failed / "ipc/service.sock"
                        with (failed / "service.log").open("w") as failed_log:
                            failed_service = subprocess.Popen(
                                [sys.executable, "-m", "optimizer.transport.shm.service", "--socket", str(failed_socket),
                                 "--max-assets", "6000"], cwd=work_dir,
                                stdout=failed_log, stderr=subprocess.STDOUT)
                            try:
                                failure_deadline = time.monotonic() + 15
                                while not failed_socket.exists():
                                    if failed_service.poll() is not None or time.monotonic() >= failure_deadline:
                                        raise AssertionError("failure-test service did not start")
                                    time.sleep(0.02)
                                run_cli(failed, success=False)
                            finally:
                                failed_service.terminate()
                                failed_service.wait(timeout=10)
                    else:
                        run_cli(failed, success=False)
        finally:
            service.terminate()
            service.wait(timeout=10)
    assert not socket_path.exists()
    report = compare(direct, shm, dates)
    report["infeasible_runs_failed_without_success_summary"] = True
    (work_dir / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def verify(work_dir: Path) -> dict:
    # Linux AF_UNIX paths are short. Execute in a task-owned temporary folder,
    # preserving run-local sockets and relative configuration, then retain every
    # fixture/log/output under the requested evidence destination, even on error.
    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ms-cli-") as temporary:
        execution = Path(temporary) / "v"
        try:
            return _verify(execution)
        finally:
            if execution.exists():
                shutil.copytree(execution, work_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory for retained evidence")
    args = parser.parse_args()
    print(json.dumps(verify(args.work_dir), indent=2))
