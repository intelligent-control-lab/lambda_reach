#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
spark_python="${1:?Usage: bash scripts/setup_spark_env.sh /path/to/existing/spark/python}"

if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    uv venv --system-site-packages --python "$spark_python" "$repo_root/.venv"
fi
uv pip install --python "$repo_root/.venv/bin/python" --no-deps \
    -r "$repo_root/requirements/spark-overlay.txt"
uv pip install --python "$repo_root/.venv/bin/python" --no-deps -e "$repo_root/safety_value"
"$repo_root/.venv/bin/python" -c 'import spark_pipeline, mujoco, safety_value; print("SPARK monitor environment ready")'
