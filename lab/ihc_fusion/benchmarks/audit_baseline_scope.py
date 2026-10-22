# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only audit: does the official baseline time the complete Post + Pre?

Decomposes the official arm into its own two halves with the *identical* launch
configuration, proves the decomposition reproduces the full chain bit-exactly,
then times Post-only, Pre-only, serial Post+Pre, the in-tree runtime wrappers
and the fused kernel under the same timer.

Phases are separate processes so profiler overhead cannot reach the timing:

    --phase timing   cold/hot graph + allocations, no profiler
    --phase profile  kernel names, order and counts per logical call
"""

import argparse
import functools
import hashlib
import json
import statistics
from pathlib import Path

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cudagraph

from lab.ihc_fusion.baselines.official import (
    OFFICIAL,
    official_post_pre,
    official_post_pre_into,
)
from lab.ihc_fusion.benchmarks.bench_post_pre import fused_call, get_timer, time_call
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "lab/ihc_fusion"
TOKENS = (64, 128, 1024, 4096)
HIDDEN = 6144
DTYPE = torch.bfloat16
MAGNITUDE, HC_EPS, NORM_EPS = 2.0, 1e-6, 1e-5
CUDA = torch.autograd.DeviceType.CUDA
SPLITS = triton.cdiv(4 * HIDDEN, 1024)


def official_post_into(x, residual, previous, updated):
    """Exactly the Post launch that official_post_pre_into performs."""
    _, channels, hidden = residual.shape
    cp = triton.next_power_of_2(channels)
    OFFICIAL._ihc_post_kernel[
        (residual.shape[0], triton.cdiv(hidden, OFFICIAL._BLOCK_D))
    ](
        x,
        residual,
        previous,
        updated,
        HIDDEN_SIZE=hidden,
        HC_MULT=channels,
        HC_POW2=cp,
        BLOCK_D=OFFICIAL._BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )


def official_pre_into(
    updated, weight, scale, base, magnitude, hc_eps, norm_eps, output, post, partial
):
    """Exactly the two Pre launches that official_post_pre_into performs."""
    tokens, channels, hidden = updated.shape
    k = channels * hidden
    cp = triton.next_power_of_2(channels)
    splits = triton.cdiv(k, OFFICIAL._BLOCK_K)
    OFFICIAL._ihc_pre_stage1[(tokens, splits)](
        updated,
        weight,
        partial,
        K_TOTAL=k,
        HC_MULT=channels,
        HC_POW2=cp,
        NUM_SPLITS=splits,
        BLOCK_K=OFFICIAL._BLOCK_K,
        PARTIAL_STRIDE=1 + 2 * cp,
        num_warps=8,
        enable_fp_fusion=False,
    )
    OFFICIAL._ihc_pre_stage2[(tokens, triton.cdiv(hidden, OFFICIAL._BLOCK_D))](
        updated,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=hidden,
        K_TOTAL=k,
        HC_MULT=channels,
        HC_POW2=cp,
        NUM_SPLITS=splits,
        PARTIAL_STRIDE=1 + 2 * cp,
        BLOCK_D=OFFICIAL._BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )


def post_only(*args):
    official_post_into(args[0], args[1], args[2], args[9][1])


def pre_only(*args):
    official_pre_into(
        args[9][1],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9][0],
        args[9][2],
        args[10],
    )


def post_pre_serial(*args):
    official_post_pre_into(*args[:9], out=args[9], partial=args[10])


def runtime_unfused(*args):
    """What vLLM actually calls: the two public wrappers, allocations included."""
    updated = OFFICIAL.triton_ihc_post(args[0], args[1], args[2])
    output, post = OFFICIAL.triton_ihc_pre(
        updated, args[3], args[4], args[5], args[6], args[7], args[8]
    )
    return output, updated, post


def runtime_unfused_call(*args):
    return runtime_unfused(*args[:9])


def arms():
    return {
        "official_post_only": (post_only, 1),
        "official_pre_only": (pre_only, 2),
        "official_post_pre_serial": (post_pre_serial, 3),
        "runtime_unfused_wrappers": (runtime_unfused_call, 3),
        "fused": (functools.partial(fused_call, num_warps=None), 1),
    }


def build_case(tokens):
    return (*make_case(tokens, HIDDEN, DTYPE)[:6], MAGNITUDE, HC_EPS, NORM_EPS)


def hot_us(fn, args, calls=32):
    samples = bench_gpu_time_with_cudagraph(
        fn,
        input_args=args,
        cold_l2_cache=False,
        num_iters_within_graph=calls,
        dry_run_time_ms=25,
        repeat_time_ms=100,
    )
    return statistics.median(samples) * 1000


def cuda_events(prof):
    events = [
        e for e in prof.events() if e.device_type == CUDA and e.device_time_total > 0
    ]
    events.sort(key=lambda e: e.time_range.start)
    return events


def profile_call(fn, args, iters=20):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
    events = cuda_events(prof)
    per_call = {}
    for event in events:
        entry = per_call.setdefault(event.name, [0, 0.0])
        entry[0] += 1
        entry[1] += event.device_time_total
    count = sum(v[0] for v in per_call.values()) / iters
    order = [e.name for e in events[: int(round(count))]]
    return {
        "kernels_per_call": count,
        "order": order,
        "device_us_per_call": sum(v[1] for v in per_call.values()) / iters,
        "per_kernel_us": {k: round(v[1] / v[0], 4) for k, v in per_call.items()},
    }


def allocations(fn, args):
    counts = {"torch.empty": 0, "torch.empty_like": 0}
    saved = (torch.empty, torch.empty_like)

    def wrap(name, original):
        def counted(*a, **k):
            counts[name] += 1
            return original(*a, **k)

        return counted

    torch.empty = wrap("torch.empty", saved[0])
    torch.empty_like = wrap("torch.empty_like", saved[1])
    try:
        fn(*args)
        torch.cuda.synchronize()
    finally:
        torch.empty, torch.empty_like = saved
    return counts


def timing_phase(options):
    timer_name, timer = get_timer()
    rows = []
    for tokens in TOKENS:
        args = build_case(tokens)
        reference = official_post_pre(*args)
        golden = torch_post_pre(*args)
        off = tuple(torch.empty_like(t) for t in reference)
        inp = (*args, off, torch.empty((tokens, SPLITS, 9), device="cuda"))
        fused_out = tuple(torch.empty_like(t) for t in reference)
        fused_inp = (*args, fused_out, inp[10])
        post_pre_serial(*inp)
        whole = tuple(t.clone() for t in off)
        post_only(*inp)
        post_half = off[1].clone()
        pre_only(*inp)
        decomposition_exact = all(
            torch.equal(a, b) for a, b in zip(off, whole)
        ) and torch.equal(post_half, whole[1])
        candidate = {
            "official_post_only": inp,
            "official_pre_only": inp,
            "official_post_pre_serial": inp,
            "runtime_unfused_wrappers": inp,
            "fused": fused_inp,
        }
        for name, (fn, declared) in arms().items():
            inputs = candidate[name]
            for _ in range(30):
                fn(*inputs)
            torch.cuda.synchronize()
            cold = time_call(fn, inputs, options.repeat_ms, timer)
            row = {
                "tokens": tokens,
                "hidden": HIDDEN,
                "arm": name,
                "declared_launches": declared,
                "graph_cold_us": cold["median_us"],
                "graph_cold_p10_us": cold["p10_us"],
                "graph_cold_p90_us": cold["p90_us"],
                "graph_hot_us": hot_us(fn, inputs),
                "rotations": cold["rotations"],
                "calls_per_graph": cold["calls_per_graph"],
                "allocations_per_call": allocations(fn, inputs),
            }
            if name == "official_post_only":
                row["bit_exact_vs_full_chain_post"] = decomposition_exact
            if name == "official_pre_only":
                row["bit_exact_vs_full_chain_pre"] = decomposition_exact
            if name == "official_post_pre_serial":
                row["bit_exact_vs_full_chain"] = decomposition_exact
                row["vs_reference"] = check_outputs(off, reference)
                row["vs_golden"] = check_outputs(off, golden)
            if name == "runtime_unfused_wrappers":
                row["vs_reference"] = check_outputs(runtime_unfused(*args), reference)
            if name == "fused":
                row["vs_reference"] = check_outputs(fused_out, reference)
                row["vs_golden"] = check_outputs(fused_out, golden)
            rows.append(row)
            print(
                f"N={tokens:5d} {name:26s} cold={row['graph_cold_us']:8.3f} "
                f"hot={row['graph_hot_us']:8.3f} rot={row['rotations']:3d} "
                f"alloc={sum(row['allocations_per_call'].values())}",
                flush=True,
            )
        del inp, fused_inp, off, fused_out, reference, golden, whole, post_half
        torch.cuda.empty_cache()
    return {"timer": timer_name, "rows": rows}


def profile_phase(options):
    rows = []
    for tokens in TOKENS:
        args = build_case(tokens)
        reference = official_post_pre(*args)
        off = tuple(torch.empty_like(t) for t in reference)
        inp = (*args, off, torch.empty((tokens, SPLITS, 9), device="cuda"))
        fused_out = tuple(torch.empty_like(t) for t in reference)
        fused_inp = (*args, fused_out, inp[10])
        candidate = {
            "official_post_only": inp,
            "official_pre_only": inp,
            "official_post_pre_serial": inp,
            "runtime_unfused_wrappers": inp,
            "fused": fused_inp,
        }
        post_pre_serial(*inp)
        for name, (fn, declared) in arms().items():
            record = profile_call(fn, candidate[name])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    fn(*candidate[name])
            graph.replay()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            in_graph = len(cuda_events(prof))
            row = {
                "tokens": tokens,
                "arm": name,
                "declared_launches": declared,
                "eager": record,
                "kernels_in_one_replay_of_32_calls": in_graph,
                "kernels_in_one_replay_per_call": in_graph / 32,
            }
            rows.append(row)
            print(
                f"N={tokens:5d} {name:26s} k/call={record['kernels_per_call']:.0f} "
                f"dev={record['device_us_per_call']:8.3f}us "
                f"k/replay32={in_graph:3d} order={record['order']}",
                flush=True,
            )
            del graph
            torch.cuda.empty_cache()
        del inp, fused_inp, off, fused_out, reference
        torch.cuda.empty_cache()
    return {"rows": rows}


def metadata():
    files = [
        LAB / "benchmarks/audit_baseline_scope.py",
        LAB / "benchmarks/bench_post_pre.py",
        LAB / "benchmarks/bench_prefill_selection.py",
        LAB / "baselines/official.py",
        ROOT / "vllm/models/hy_v4/nvidia/triton_ihc.py",
    ]
    return {
        "gpu": torch.cuda.get_device_name(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "dtype": "bfloat16",
        "hidden": HIDDEN,
        "channels": 4,
        "scalars": {"magnitude": MAGNITUDE, "hc_eps": HC_EPS, "norm_eps": NORM_EPS},
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["timing", "profile"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-ms", type=int, default=100)
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(LAB):
        parser.error("Output must remain inside lab/ihc_fusion")
    if options.output.exists():
        parser.error("Refusing to overwrite existing audit output")
    torch.backends.cuda.matmul.allow_tf32 = False
    phase = {"timing": timing_phase, "profile": profile_phase}[options.phase]
    document = {"metadata": metadata(), "phase": options.phase, **phase(options)}
    options.output.write_text(json.dumps(document, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
