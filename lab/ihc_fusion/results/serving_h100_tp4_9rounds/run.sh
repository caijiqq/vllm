#!/usr/bin/env bash
set -euo pipefail
cd /data/qiuyuhao/projects/vllm-hy4-dev
export CUDA_VISIBLE_DEVICES="${HY4_GPUS:?Set HY4_GPUS to four idle GPUs}"
export PYTHONPATH="/tmp/hy4-test-bootstrap:$PWD"
export TRITON_CACHE_DIR=/tmp/hy4-ihc-rerun-JOWKo8/triton-cache
model=/data/chenhao/Hy4-preview-FP8-3-layers
out=/tmp/hy4-ihc-rerun-JOWKo8
server_pid=''
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill -INT "$server_pid"
    wait "$server_pid" || true
    server_pid=''
  fi
}
trap stop_server EXIT
for r in {0..8}; do
  modes=(original fused)
  if (( r % 2 )); then modes=(fused original); fi
  for mode in "${modes[@]}"; do
    enabled=false
    if [[ "$mode" == fused ]]; then enabled=true; fi
    dir="$out/r$r/$mode"
    mkdir -p "$dir"
    echo "$(date -u +%FT%TZ) round=$r mode=$mode starting"
    .venv/bin/python -m vllm.entrypoints.cli.main serve "$model" \
      --tensor-parallel-size 4 --host 127.0.0.1 --port 18094 \
      --enforce-eager --no-enable-prefix-caching \
      --max-model-len 65536 --max-num-batched-tokens 65536 \
      --gpu-memory-utilization 0.90 \
      --hf-overrides "{\"enable_ihc_post_pre_fusion\": $enabled, \"enable_ihc_cross_layer_fusion\": $enabled}" \
      > "$dir/server.log" 2>&1 &
    server_pid=$!
    until curl -fsS http://127.0.0.1:18094/health >/dev/null 2>&1; do
      kill -0 "$server_pid"
      sleep 2
    done
    for n in 8192 16384 32768; do
      echo "$(date -u +%FT%TZ) round=$r mode=$mode input=$n warmup=16 measured=64"
      .venv/bin/python -m vllm.entrypoints.cli.main bench serve \
        --base-url http://127.0.0.1:18094 --model "$model" \
        --dataset-name random --input-len "$n" --output-len 512 \
        --num-prompts 64 --num-warmups 16 --max-concurrency 4 \
        --request-rate inf --seed "$r" --temperature 0 --ignore-eos \
        --metric-percentiles 50,90,99 --disable-tqdm --save-result \
        --result-dir "$dir" --result-filename "n$n.json" \
        > "$dir/n$n.log" 2>&1
      echo "$(date -u +%FT%TZ) completed round=$r mode=$mode input=$n"
    done
    stop_server
  done
done
