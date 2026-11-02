# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-input BF16 layout ablation; all arms share splits, partials and merge."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_hy4_smallhead_decode import make_inputs, reference
from lab.hy4_smallhead.ablation_kernels import gluon_split, triton_split
from vllm.models.hy_v4.nvidia.smallhead_sparse_decode import _softmax_merge, _split_kv
from vllm.triton_utils import triton

# Each entry changes only the named parameter relative to its language's control.
CONFIGS = {
    # Keep the historical result name tied to the frozen Gluon baseline.
    "current": ("gluon", {}),
    "production": ("production", {"num_stages": 2}),
    "gl_control": ("gluon", {}),
    "gl_loadstore": ("gluon", {"ASYNC_KV": False}),
    "gl_no_swizzle": ("gluon", {"MAX_PHASE": 1}),
    "gl_swizzle2": ("gluon", {"PER_PHASE": 2, "MAX_PHASE": 4}),
    "gl_swizzle4": ("gluon", {"PER_PHASE": 4, "MAX_PHASE": 2}),
    "gl_warp22": ("gluon", {"WM": 2, "WN": 2}),
    "gl_warp41": ("gluon", {"WM": 4, "WN": 1}),
    "gl_copy14": ("gluon", {"CM": 1, "CN": 4}),
    "gl_chunk64": ("gluon", {"CHUNK": 64}),
    "gl_chunk256": ("gluon", {"CHUNK": 256}),
    "gl_chunk512": ("gluon", {"CHUNK": 512}),
    "gl_n32": ("gluon", {"BN": 32}),
    "gl_n128": ("gluon", {"BN": 128}),
    "gl_w8": ("gluon", {"WN": 8, "CM": 8, "num_warps": 8}),
    "tl_n32_w4_s1": ("triton", {"BN": 32}),
    "tl_n32_w4_s2": ("triton", {"BN": 32, "num_stages": 2}),
    "tl_n64_w4_s1": ("triton", {}),
    "tl_n64_w4_s2": ("triton", {"num_stages": 2}),
    "tl_n64_w4_s3": ("triton", {"num_stages": 3}),
    "tl_n64_w8_s1": ("triton", {"num_warps": 8}),
    "tl_n128_w4_s1": ("triton", {"BN": 128}),
}


def launch(name, data, splits, merge=True):
    q, kv, indices, sinks, partial, out, lse = data
    t, heads, _ = q.shape
    kind, knobs = CONFIGS[name]
    args = (
        q,
        kv,
        indices,
        None,
        partial,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        indices.stride(1),
        kv.shape[0],
        heads,
        indices.shape[1],
        0.0625,
        splits,
        False,
    )
    options = {"num_warps": 4, "num_stages": 1, **knobs}
    if kind == "triton":
        kernel = triton_split[(t, splits)](*args, **options)
    elif kind == "production":
        kernel = _split_kv[(t, splits)](*args, **options)
    else:
        kernel = gluon_split[(t, splits)](*args, True, True, **options)
    if merge:
        _softmax_merge[(t, heads)](
            partial, sinks, out, lse, heads, splits, True, num_warps=4
        )
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 128])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--splits", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-once", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.cuda.get_device_properties(0)
    rows = []
    for tokens in args.tokens:
        for requested_splits in args.splits:
            if not args.seed_once:
                torch.manual_seed(args.seed + tokens)
            splits = requested_splits or triton.next_power_of_2(
                min(32, triton.cdiv(device.multi_processor_count, tokens))
            )
            pool = max(2, triton.cdiv(32, tokens))
            inputs = []
            for _ in range(pool):
                q, kv, ids, sink = make_inputs(
                    tokens, args.heads, rows=16384, independent_requests=True
                )
                partial = torch.empty(
                    (tokens, splits, args.heads, 514),
                    device="cuda",
                    dtype=torch.float32,
                )
                out = q.new_empty((tokens, args.heads, 512))
                lse = torch.empty(
                    (tokens, args.heads), device="cuda", dtype=torch.float32
                )
                inputs.append((q, kv, ids, sink, partial, out, lse))
            accessed = sum(torch.unique(x[2]).numel() * 1152 for x in inputs)
            assert accessed > 50 * 1024**2
            expected, expected_lse = reference(*inputs[0][:3], 0.0625, inputs[0][3])
            results = {}
            for name in args.variants:
                print(f"Warmup tokens={tokens} variant={name}", flush=True)
                compiled = launch(name, inputs[0], splits)
                torch.testing.assert_close(
                    inputs[0][5].float(), expected, rtol=0.02, atol=0.01
                )
                torch.testing.assert_close(
                    inputs[0][6], expected_lse, rtol=1e-4, atol=1e-4
                )
                for data in inputs[1:]:
                    launch(name, data, splits)
                ptx = compiled.asm["ptx"]
                Path(
                    f"/tmp/hy4-ablation-{name}-h{args.heads}-t{tokens}-s{splits}.ptx"
                ).write_text(ptx)
                results[name] = dict(
                    tokens=tokens,
                    heads=args.heads,
                    splits=splits,
                    pool=pool,
                    variant=name,
                    knobs=CONFIGS[name][1],
                    regs=compiled.n_regs,
                    spills=compiled.n_spills,
                    shared_bytes=compiled.metadata.shared,
                    cp_async_instructions=ptx.count("cp.async."),
                    mma_instructions=ptx.count("mma.sync."),
                    wgmma_instructions=ptx.count("wgmma."),
                    ir_layouts=[
                        line
                        for line in compiled.asm["ttgir"].splitlines()
                        if "nvidia_mma" in line or "swizzled_shared" in line
                    ],
                    accessed_kv_bytes=accessed,
                    max_abs_error=(inputs[0][5].float() - expected).abs().max().item(),
                    total_samples_us=[],
                    split_samples_us=[],
                )
            if args.profile:
                from cuda.bindings import runtime

                torch.accelerator.synchronize()
                runtime.cudaProfilerStart()
                for name in args.variants:
                    launch(name, inputs[0], splits)
                torch.accelerator.synchronize()
                runtime.cudaProfilerStop()
            else:
                for round_id in range(args.rounds):
                    order = args.variants[round_id:] + args.variants[:round_id]
                    if round_id % 2:
                        order = order[::-1]
                    for name in order:
                        for merge, key in [
                            (True, "total_samples_us"),
                            (False, "split_samples_us"),
                        ]:

                            def run(
                                name=name, merge=merge, inputs=inputs, splits=splits
                            ):
                                for data in inputs:
                                    launch(name, data, splits, merge)

                            ms = triton.testing.do_bench_cudagraph(run, rep=args.rep)
                            results[name][key].append(ms * 1000 / pool)
                for row in results.values():
                    row["total_us"] = statistics.median(row["total_samples_us"])
                    row["split_us"] = statistics.median(row["split_samples_us"])
                    row["speedup_vs_current"] = (
                        statistics.median(results["current"]["total_samples_us"])
                        / row["total_us"]
                    )
            rows.extend(results.values())
            for row in results.values():
                print(json.dumps(row), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        gpu=device.name,
                        gpu_uuid=str(device.uuid),
                        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                        sm_count=device.multi_processor_count,
                        torch=torch.__version__,
                        triton=triton.__version__,
                        cuda=torch.version.cuda,
                        seed=args.seed,
                        input_seed_policy=(
                            "seed once over ordered shapes"
                            if args.seed_once
                            else "seed + tokens"
                        ),
                        rounds=args.rounds,
                        rep_ms=args.rep,
                        kv_dtype="bfloat16",
                        topk=2048,
                        independent_requests=True,
                        timing="cuda_graph_events_preallocated",
                        baseline_commit="daa54b83bb61f49954d8823a0a3197f448b6e480",
                        production_source_sha256=hashlib.sha256(
                            (
                                Path(__file__).resolve().parents[2]
                                / "vllm/models/hy_v4/nvidia/smallhead_sparse_decode.py"
                            ).read_bytes()
                        ).hexdigest(),
                        results=rows,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
