#!/usr/bin/env bash
set -euo pipefail
: "${HY4_TP4_RUN:?Set a fresh TP4 results directory}"
: "${HY4_TP4_CHECKPOINT:?Set the existing checkpoint path}"
export CUDA_VISIBLE_DEVICES=1,2,3,5
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/lab/ihc_fusion/benchmarks/tp4_validation:$PWD/lab/ihc_fusion/results/runtime_integration:$PWD"
export TRITON_CACHE_DIR="$HY4_TP4_RUN/cache/triton"
export VLLM_CACHE_ROOT="$HY4_TP4_RUN/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$HY4_TP4_RUN/cache/inductor"
export CUDA_CACHE_PATH="$HY4_TP4_RUN/cache/cuda"
export XDG_CACHE_HOME="$HY4_TP4_RUN/cache/xdg"
export FLASHINFER_WORKSPACE_BASE="$HY4_TP4_RUN/cache/flashinfer"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
exec lab/ihc_fusion/results/.venv/bin/python "$@"
