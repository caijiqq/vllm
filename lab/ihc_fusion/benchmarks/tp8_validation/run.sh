#!/usr/bin/env bash
set -euo pipefail
HARNESS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HARNESS/../../../.." && pwd)
PYTHON=${HY4_TP8_PYTHON:-"$REPO/.venv/bin/python"}
test -x "$PYTHON"
ENTRY=${1:?Use inspect_environment.py, validate.py, benchmark.py or analyze_perf.py}
shift
if [[ "$ENTRY" == analyze_perf.py ]]; then
  # Analysis uses saved data only, before checkpoint checks or GPU environment setup.
  exec "$PYTHON" -S "$HARNESS/$ENTRY" "${HY4_TP8_RUN:?Set the saved run directory}/performance" "$@"
fi
: "${HY4_TP8_RUN:?Set a fresh results directory outside the checkout}"
: "${HY4_TP8_CHECKPOINT:?Set the existing three-layer checkpoint path}"
export HY4_TP8_RUN=$(realpath -m -- "$HY4_TP8_RUN")
export HY4_TP8_CHECKPOINT=$(realpath -e -- "$HY4_TP8_CHECKPOINT")
case "$HY4_TP8_RUN/" in
  "$REPO/"*) echo "Keep TP8 results outside the checkout" >&2; exit 2 ;;
esac
test -f "$HY4_TP8_CHECKPOINT/config.json"
test -f "$HY4_TP8_CHECKPOINT/model.safetensors"
case "$ENTRY" in
  inspect_environment.py) PHASE=environment ;;
  validate.py) PHASE=validation ;;
  benchmark.py) PHASE=performance ;;
  *) echo "Unknown TP8 entrypoint: $ENTRY" >&2; exit 2 ;;
esac
if [[ ! -d "$HY4_TP8_RUN" ]]; then
  mkdir "$HY4_TP8_RUN"
  printf '%s\n' "$HY4_TP8_CHECKPOINT" > "$HY4_TP8_RUN/.tp8_run"
fi
test "$(cat "$HY4_TP8_RUN/.tp8_run")" = "$HY4_TP8_CHECKPOINT"
mkdir -p "$HY4_TP8_RUN/environment" "$HY4_TP8_RUN/$PHASE/raw"
export HY4_TP8_GPUS=${HY4_TP8_GPUS:-0,1,2,3,4,5,6,7}
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
# No sitecustomize, torch import or CUDA context before the availability gate.
"$PYTHON" -S "$HARNESS/check_gpus.py"
export CUDA_VISIBLE_DEVICES="$HY4_TP8_GPUS"
export PYTHONPATH="$HARNESS/bootstrap:$REPO"
export TRITON_CACHE_DIR="$HY4_TP8_RUN/cache/triton"
export VLLM_CACHE_ROOT="$HY4_TP8_RUN/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$HY4_TP8_RUN/cache/inductor"
export CUDA_CACHE_PATH="$HY4_TP8_RUN/cache/cuda"
export XDG_CACHE_HOME="$HY4_TP8_RUN/cache/xdg"
export FLASHINFER_WORKSPACE_BASE="$HY4_TP8_RUN/cache/flashinfer"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
STAMP=$(date -u +%Y%m%dT%H%M%S.%N)
"$PYTHON" "$HARNESS/$ENTRY" "$@" \
  2>&1 | tee "$HY4_TP8_RUN/$PHASE/${ENTRY%.py}_$STAMP.log"
