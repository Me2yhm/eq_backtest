#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image_tag="${EQ_BACKTEST_IMAGE:-eq-backtest:local}"
default_run_dir="$repo_dir/runs/daily_v6_production"

if (($# > 1)); then
  echo "Usage: $0 [run-directory]" >&2
  exit 2
fi

run_dir="${1:-$default_run_dir}"
run_dir="$(cd "$run_dir" && pwd -P)"
if [[ ! -f "$run_dir/config.yml" ]]; then
  echo "Run directory must contain config.yml: $run_dir" >&2
  exit 2
fi

host_home="${HOME:?HOME must be set}"
host_home="$(cd "$host_home" && pwd -P)"
case "$run_dir" in
  "$host_home"/*) ;;
  *)
    echo "Run directory must be under HOME so config paths remain valid: $run_dir" >&2
    exit 2
    ;;
esac

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env "HOME=$host_home" \
  --tmpfs /tmp:rw,mode=1777 \
  --mount "type=bind,src=$host_home,dst=$host_home,readonly" \
  --mount "type=bind,src=$repo_dir,dst=/app,readonly" \
  --mount "type=bind,src=$repo_dir/runs,dst=$repo_dir/runs" \
  --mount "type=bind,src=$repo_dir/cache,dst=$repo_dir/cache" \
  "$image_tag" \
  "$run_dir"
