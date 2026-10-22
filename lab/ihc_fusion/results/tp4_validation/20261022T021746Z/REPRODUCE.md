# TP4 N8192 performance experiment

Run from repository root, on four free H100 GPUs 1,2,3,5. Existing environment only; no installation. Set `HY4_TP4_RUN` to a **fresh** directory with a `raw/` child. The checkpoint must be the existing three-layer FP8 slice used in the preceding TP4 validation. Do not download or alter weights.

Actual command (checkpoint path supplied locally, omitted here):

```bash
HY4_TP4_RUN="$PWD/lab/ihc_fusion/results/tp4_validation/20261022T021746Z" \
HY4_TP4_CHECKPOINT="$EXISTING_HY4_THREE_LAYER_CHECKPOINT" \
bash lab/ihc_fusion/benchmarks/tp4_validation/run.sh \
lab/ihc_fusion/benchmarks/tp4_validation/benchmark.py \
> lab/ihc_fusion/results/tp4_validation/20261022T021746Z/run.log 2>&1

lab/ihc_fusion/results/.venv/bin/python \
lab/ihc_fusion/benchmarks/tp4_validation/analyze_perf.py \
lab/ihc_fusion/results/tp4_validation/20261022T021746Z
```

- TP4/PP1/mp, eager, BF16 activations, MXFP8, max model length8448, max batched tokens8192, prefix cache off, memory utilization0.65, max sequences4.
- One deterministic prompt:8192 token IDs `100+i%197`; temperature0, max_tokens1, seed7.
- Same model instance for all modes. Existing all-worker RPC sets flags; every switch validates all four ranks. Original off/off; Intra on/off; Cross on/on with final-layer defer off.
- Six warmup requests per mode plus one finite-output measured diagnostic; no model A/A gate. One settling request after each subsequent mode switch. Nine balanced rotating rounds,27 measurements total.
- CUDA events enclose top-level model Forward (three layers plus embedding/HCHead; logits/sampling outside). Device-group barrier is issued immediately before start event, outside measured interval. Per-sample maximum over four rank durations is the reported TP4 Forward estimate; all four local durations retained. This is not an absolute-clock multi-device span; residual barrier/start skew is a limitation. Model-required communication remains inside interval.
- `generate_wall_ms` separately measures host generate call, excluding arm/read RPCs; not confused with Forward or full78-layer E2E.
- No explicit cache flush; repeated-request steady-state eager cache conditions. No clocks/power settings changed.
- After timing persisted, three settling requests per mode, then one CPU+CUDA PyTorch Profiler trace per rank/mode. Instance-only module hooks label launch ancestry. Profiling is excluded from timing.
- Classification gives communication priority over enclosing Attention/MLP modules. Remaining categories are exclusive. Activity sums measure GPU work, not wall time. Union and first-to-last activity span explicitly expose overlap and gaps. Correlation IDs connect GPU launches to CPU module scopes.
- Original five boundaries: three Attention Posts + three MLP Pres + first two MLP Posts + next two Attention Pres. Each Pre includes both stages. Cross expected five fused kernels, initial Pre stages, and final Post.
- Model strict numerical acceptance remains open. Finite outputs and valid execution are required; known Original A/A noise is not treated as a performance gate. No numerical tolerance is changed.
