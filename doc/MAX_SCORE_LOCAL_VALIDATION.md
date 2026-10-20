# Local source and wheel verification

Both repositories must be checked out at their reviewed local max_score branches.
The EQ lock continues to pin the last published optimizer revision; an ordinary
`uv sync` can replace the locally tested optimizer. No unpushed SHA is presented
as remotely installable. Use an explicit local install and `--no-sync`, or the
interpreter directly. These commands never connect to a trading system.

From the optimizer checkout, with a new isolated environment:

```bash
uv venv /tmp/max-score-local-env
uv pip install --python /tmp/max-score-local-env/bin/python -e '.[lp]'
cd /tmp
/tmp/max-score-local-env/bin/python /path/to/optimizer/scripts/check_max_score_install.py --lp
```

To check base dependency isolation, use another new environment, install `-e .`
without `[lp]`, and run the same script without `--lp`. It verifies equal-weight
success, no SciPy import, and the explicit `DEPENDENCY_UNAVAILABLE` LP error.
Build a wheel from the reviewed checkout using `python -m build --wheel` (the
`build` package is a development tool). Install that wheel in a third environment
and run the smoke check from outside both repositories. Test both the base wheel
and that wheel with SciPy installed. Record its SHA-256 and imported package path.

For a complete EQ environment, the existing EQ virtual environment can be used
without synchronizing its SSH pin:

```bash
# From eq-backtest; install the sibling checkout, never its remote pin.
uv pip install --python .venv/bin/python -e '../optimizer[lp]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_optimizer_modes.py --work-dir /tmp/new-equal-dual
.venv/bin/python scripts/verify_max_score_modes.py --work-dir /tmp/new-max-dual
.venv/bin/python scripts/verify_optimizer_pressure.py long-cli --work-dir /tmp/new-equal-pressure --nofile 256
```

Alternatively, use `PYTHONPATH=/path/to/local/optimizer` with the EQ interpreter
and its explicitly installed SciPy extra. The acceptance manifest records which
method was used; source-path invocation is separate from wheel installation.
Linux tests need local Unix socket and memfd permissions. Set
`RUN_OPTIMIZER_IPC_TESTS=1` to include the opt-in legacy CLI/pressure tests.

From optimizer, the bounded full-market command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark_max_score.py --output /tmp/new-max-benchmark
```

It retains 20 measured samples plus one warm-up at N=1000/3000/5000/5443/6000,
then 2000 small requests with deliberate infeasibility and 100 full-market
sequential native SHM requests. All four constraints bind in the timing fixture.
The state chain is a synthetic static-price/no-fill fixture, independently of
EQ's tested actual-state chain. No stable p99 or production SLA is claimed.
An orchestrator should enforce a 900-second outer timeout. Individual LP time
limits are cooperative; they do not make native calls hard real time.

Read-only real-source snapshots use `scripts/verify_max_score_sources.py` with
explicit `--predictions`, `--market-root`, `--barra-root`, `--selection` and a
new `--output`. The selection JSON declares `purpose=engineering_acceptance`
and dates selected before optimization. Output contains private paths and must
remain local. It does not certify historical publication, eligibility, execution
price semantics or real-account history.

Publishing a release, replacing the remote dependency pin, pushing, merging and
production parameter approval remain separate actions. None occurs in these
commands.
