# Frozen dependency and local verification

EQ declares `optimizer[lp]` at the merged optimizer Git SHA
`3ffc0e128799379aa37c61bbb7ada900eac6952f`; `uv.lock` includes SciPy through the
LP extra. Package version `0.2.0` alone does not identify this revision. The
default model remains `equal_weight/1`; `max_score/1` requires explicit config.

For integration acceptance, run from the EQ checkout with GitHub SSH access and
a new environment. Do not install additional packages into this environment or
override optimizer through `PYTHONPATH`:

```bash
TASK_ROOT=$(mktemp -d /tmp/eq-frozen.XXXXXX)
uv lock --check
PYTHONPATH= UV_PROJECT_ENVIRONMENT="$TASK_ROOT/venv" uv sync --frozen
(
    cd "$TASK_ROOT"
    "$TASK_ROOT/venv/bin/python" -I -c 'import importlib.metadata as m, optimizer, scipy; print(optimizer.__file__); print(scipy.__version__); print(m.distribution("optimizer").read_text("direct_url.json"))'
)
RUN_OPTIMIZER_IPC_TESTS=1 PYTHONPATH= "$TASK_ROOT/venv/bin/python" -m unittest discover -s tests -v
PYTHONPATH= "$TASK_ROOT/venv/bin/python" scripts/verify_optimizer_modes.py --work-dir "$TASK_ROOT/equal-dual"
PYTHONPATH= "$TASK_ROOT/venv/bin/python" scripts/verify_max_score_modes.py --work-dir "$TASK_ROOT/max-dual"
```

Check that optimizer imports from the new environment's `site-packages`, its
`direct_url.json` has both `commit_id` and `requested_revision` equal to the full
SHA above, and SciPy is installed. Record Python and dependency versions and
compare the installed optimizer package files with that commit. A lock check
establishes manifest consistency; `--frozen` itself uses the existing lock
without checking whether it matches changed project metadata. Linux native
tests require Unix socket and memfd permissions; record any skips as untested.
The dual-mode scripts retain synthetic fixtures, logs and output comparisons.
For a historical compatibility claim, additionally compare both equal-weight
runs with the archived baseline using exact numeric equality; the scripts'
cross-mode comparison alone does not establish that claim.

Source and wheel checks below are optional development checks, separate from
the frozen Git installation above. From the optimizer checkout, use another
new environment:

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

Publishing a release, pushing, merging and production parameter approval remain
separate actions. These verification commands use synthetic fixtures or explicitly
selected read-only snapshots and do not connect to a trading system.
