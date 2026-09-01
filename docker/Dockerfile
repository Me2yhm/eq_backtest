# syntax=docker/dockerfile:1
# Override UV_IMAGE with a tested digest for fully reproducible image tooling.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest
FROM ${UV_IMAGE} AS uv

FROM python:3.11-slim-bookworm

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/eq-backtest-venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PATH=/opt/eq-backtest-venv/bin:$PATH

# Keep dependency installation cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . ./

# The runner mounts a fresh writable /tmp because it runs as the host UID.
# The image is always run with an existing dedicated run directory as argv[1].
ENTRYPOINT ["uv", "run", "--no-sync", "python", "run.py"]
