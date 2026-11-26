#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 GlobalForecast|TyphoonTracking train|test DATA_ROOT [arguments...]" >&2
    exit 2
fi
benchmark="$1"
action="$2"
data_root="$3"
shift 3
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$data_root" = /* ]] || data_root="$PWD/$data_root"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONNOUSERSITE=1
cd "$root"
exec "${PYTHON_BIN:-python}" tools/run_protocast.py --benchmark "$benchmark" --action "$action" --data-root "$data_root" "$@"
