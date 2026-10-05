#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
run_root="$repo_root/logs/safety_analysis/spark_29dof"
# Frozen SPARK-trained pair, qualified on independent episodes in both engines.
default_bundle="$run_root/spark_replacement_20261005_v2/monitor/bundle.json"

# Small online inference batches avoid unnecessary CPU thread overhead.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ ! -x "$python_bin" ]]; then
    echo "Create the repo environment with scripts/setup_spark_env.sh first." >&2
    exit 1
fi
# Explicit model arguments take precedence over the saved default pair.
model_args=()
has_model_args=false
for argument in "$@"; do
    case "$argument" in
        --bundle|--bundle=*|--policy|--policy=*|--config|--config=*|--value|--value=*)
            has_model_args=true ;;
    esac
done
if [[ "$has_model_args" == false ]]; then
    bundle="${SPARK_MONITOR_BUNDLE:-$default_bundle}"
    if [[ ! -f "$bundle" ]]; then
        echo "Missing trained monitor bundle: $bundle" >&2
        echo "See docs/SPARK_RETRAINING.md, or supply --bundle /path/to/bundle.json." >&2
        exit 1
    fi
    model_args=(--bundle "$bundle" --no-download)
fi
mkdir -p "$run_root/viewer"
run_dir="$(mktemp -d "$run_root/viewer/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Recording viewer run to $run_dir/trajectory.hdf5"
exec "$python_bin" "$repo_root/safety_value/scripts/2_safety_analysis/run_monitor.py" \
    --backend mujoco --mujoco-scene spark --family avoid --viewer \
    "${model_args[@]}" --episodes 4 --steps 1000 --seed 62 \
    --output "$run_dir/trajectory.hdf5" "$@"
