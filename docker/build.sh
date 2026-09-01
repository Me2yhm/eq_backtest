#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image_tag="${EQ_BACKTEST_IMAGE:-eq-backtest:local}"
uv_image="${UV_IMAGE:-ghcr.io/astral-sh/uv:latest}"

exec docker build \
  --file "$repo_dir/docker/Dockerfile" \
  --tag "$image_tag" \
  --build-arg "UV_IMAGE=$uv_image" \
  "$repo_dir"
