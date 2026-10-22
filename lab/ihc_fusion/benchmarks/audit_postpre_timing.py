# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only audit of per-call GPU kernel ledger and timing boundary.

Two isolated phases; run them as separate processes so profiler overhead can
never contaminate the timed numbers:

    --phase timing   no profiler: eager / graph / cold / hot / host submit
    --phase profile  torch.profiler only: kernel names, order, counts, device time

Nothing here is imported by the kernels, the baseline or the runtime. The
frozen kernels, Golden, official baseline and existing results are untouched.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cudagraph
from flashinfer.testing.utils import calculate_rotation_count

from lab.ihc_fusion.baselines.official import official_post_pre, provenance
from lab.ihc_fusion.benchmarks.bench_decode_two_stage import two_call
from lab.ihc_fusion.benchmarks.bench_post_pre import (
    baseline_call,
    fused_call,
    get_timer,
    time_call,
)
from lab.ihc_fusion.kernels.decode_two_stage import decode_two_stage
from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "lab/ihc_fusion"
TOKENS = (64, 128, 1024, 4096)
HIDDEN = 6144
DTYPE = torch.bfloat16
# Public HY4-preview scalars used by the D=6144 benchmark rows.
MAGNITUDE, HC_EPS, NORM_EPS = 2.0, 1e-6, 1e-5
CUDA = torch.autograd.DeviceType.CUDA
SPLITS = triton.cdiv(4 * HIDDEN, 1024)


def arms():
    """Benchmark arms exactly as the existing harness invokes them."""
    return {
        "official": (baseline_call, 3),
        "single": (functools.partial(fused_call, num_warps=None), 1),
        "two": (two_call, 2),
    }


def wrappers():
    """Same logical calls with the allocation that the public wrappers do."""
    return {
        "official": lambda *a: official_post_pre(*a),
        "single": lambda *a: fused_post_pre(*a),
        "two": lambda *a: decode_two_stage(*a),
    }


def build_case(tokens):
    args = (*make_case(tokens, HIDDEN, DTYPE)[:6], MAGNITUDE, HC_EPS, NORM_EPS)
    return args


def build_inputs(args, tokens, reference):
    out = tuple(torch.empty_like(t) for t in reference)
    partial = torch.empty((tokens, SPLITS, 9), device="cuda", dtype=torch.float32)
    return (*args, out, partial)


@contextlib.contextmanager
def allocation_counter():
    """Count host allocator calls; purely observational, restored on exit."""
    counts = {"torch.empty": 0, "torch.empty_like": 0, "torch.zeros": 0}
    saved = (torch.empty, torch.empty_like, torch.zeros)

    def wrap(name, original):
        def counted(*a, **k):
            counts[name] += 1
            return original(*a, **k)

        return counted

    torch.empty = wrap("torch.empty", saved[0])
    torch.empty_like = wrap("torch.empty_like", saved[1])
    torch.zeros = wrap("torch.zeros", saved[2])
    try:
        yield counts
    finally:
        torch.empty, torch.empty_like, torch.zeros = saved


def host_and_wall_us(fn, args, iters):
    """Per-call CPU submit cost and per-call wall clock (host or GPU bound)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    submit_us = (time.perf_counter() - start) / iters * 1e6
    torch.cuda.synchronize()
    total_us = (time.perf_counter() - start) / iters * 1e6
    return submit_us, total_us


def eager_event_us(fn, args, iters):
    """Eager back-to-back GPU span per call, CUDA events around the loop."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def hot_graph_us(fn, args, calls=32, repeat_ms=100):
    samples = bench_gpu_time_with_cudagraph(
        fn,
        input_args=args,
        cold_l2_cache=False,
        num_iters_within_graph=calls,
        dry_run_time_ms=25,
        repeat_time_ms=repeat_ms,
    )
    return statistics.median(samples) * 1000


def rotation_tensors(args):
    """Identical tensor set that bench_post_pre.time_call rotates."""
    return list(args[:6]) + list(args[9]) + [args[10]]


def timing_phase(options):
    timer_name, timer = get_timer()
    rows = []
    for tokens in TOKENS:
        args = build_case(tokens)
        reference = official_post_pre(*args)
        golden = torch_post_pre(*args)
        for name, (fn, declared) in arms().items():
            inp = build_inputs(args, tokens, reference)
            fn(*inp)
            torch.cuda.synchronize()
            errors = check_outputs(inp[9], reference)
            with allocation_counter() as fast_allocs:
                fn(*inp)
            with allocation_counter() as wrapper_allocs:
                wrappers()[name](*args)
            torch.cuda.synchronize()
            for _ in range(30):
                fn(*inp)
            torch.cuda.synchronize()
            submit_us, wall_us = host_and_wall_us(fn, inp, 50)
            wrapper_submit_us, wrapper_us = host_and_wall_us(wrappers()[name], args, 50)
            eager_us = eager_event_us(fn, inp, 50)
            eager_wrapper_us = eager_event_us(wrappers()[name], args, 50)
            cold = time_call(fn, inp, options.repeat_ms, timer)
            row = {
                "dtype": "bfloat16",
                "tokens": tokens,
                "hidden": HIDDEN,
                "channels": 4,
                "arm": name,
                "declared_launches": declared,
                "correct": errors,
                "allocations_per_call_preallocated": dict(fast_allocs),
                "allocations_per_call_wrapper": dict(wrapper_allocs),
                "host_submit_us_per_call": submit_us,
                "host_submit_us_per_call_wrapper": wrapper_submit_us,
                "wall_us_per_call": wall_us,
                "wall_us_per_call_wrapper": wrapper_us,
                "eager_event_us_per_call": eager_us,
                "eager_event_us_per_call_wrapper": eager_wrapper_us,
                "graph_cold_us": cold["median_us"],
                "graph_cold_p10_us": cold["p10_us"],
                "graph_cold_p90_us": cold["p90_us"],
                "graph_replays": cold["graph_replays"],
                "calls_per_graph": cold["calls_per_graph"],
                "rotations": cold["rotations"],
                "graph_hot_us": hot_graph_us(fn, inp),
                "rotation_count_from_tensor_set": calculate_rotation_count(
                    rotation_tensors(inp), torch.device("cuda")
                ),
            }
            rows.append(row)
            print(
                f"N={tokens:5d} {name:8s} cold={row['graph_cold_us']:8.3f} "
                f"hot={row['graph_hot_us']:8.3f} eager={eager_us:8.3f} "
                f"eager_alloc={eager_wrapper_us:8.3f} submit={submit_us:7.3f} "
                f"rot={row['rotations']}",
                flush=True,
            )
            del inp
            torch.cuda.empty_cache()
        del args, reference, golden
        torch.cuda.empty_cache()
    return {"timer": timer_name, "rows": rows}


def summarise(prof, iters):
    events = [
        e for e in prof.events() if e.device_type == CUDA and e.device_time_total > 0
    ]
    events.sort(key=lambda e: e.time_range.start)
    per_call = {}
    for event in events:
        entry = per_call.setdefault(event.name, {"count": 0.0, "device_us": 0.0})
        entry["count"] += 1 / iters
        entry["device_us"] += event.device_time_total / iters
    total_us = sum(v["device_us"] for v in per_call.values())
    kernels = [e.name for e in events]
    return {
        "per_call": per_call,
        "kernels_per_call": sum(v["count"] for v in per_call.values()),
        "total_device_us_per_call": total_us,
        "first_call_order": kernels[
            : int(round(sum(v["count"] for v in per_call.values())))
        ],
        "distinct_names": sorted(per_call),
    }


def profile_eager(fn, args, iters=20):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
    return summarise(prof, iters)


def graph_of(fn, args, calls):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def profile_replay(graph, replays=20, divide=1):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    return summarise(prof, replays * divide)


def replay_event_us(graph, replays=20, divide=1):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / replays / divide * 1000


def profile_phase(options):
    rows = []
    for tokens in TOKENS:
        args = build_case(tokens)
        reference = official_post_pre(*args)
        for name, (fn, declared) in arms().items():
            inp = build_inputs(args, tokens, reference)
            fn(*inp)
            torch.cuda.synchronize()
            eager = profile_eager(fn, inp)
            single = graph_of(fn, inp, 1)
            one = profile_replay(single)
            one_us = replay_event_us(single)
            multi = graph_of(fn, inp, 32)
            batch = profile_replay(multi, replays=5, divide=32)
            batch_us = replay_event_us(multi, replays=5, divide=32)
            row = {
                "tokens": tokens,
                "hidden": HIDDEN,
                "arm": name,
                "declared_launches": declared,
                "eager": eager,
                "graph_one_call": one,
                "graph_one_call_event_us": one_us,
                "graph_replay_calls": 32,
                "graph_32_calls_per_call": batch,
                "graph_32_calls_event_us_per_call": batch_us,
            }
            rows.append(row)
            print(
                f"N={tokens:5d} {name:8s} kernels/call={one['kernels_per_call']:.0f} "
                f"sum_device={one['total_device_us_per_call']:8.3f}us "
                f"replay1_event={one_us:8.3f}us "
                f"replay32_kernels={batch['kernels_per_call']:.0f} "
                f"replay32_event/call={batch_us:8.3f}us",
                flush=True,
            )
            print(f"    order: {one['first_call_order']}", flush=True)
            del inp, single, multi
            torch.cuda.empty_cache()
        del args, reference
        torch.cuda.empty_cache()
    return {"rows": rows}


def variant_phase(options):
    """Diagnostic only: does the frozen Small-N warp default explain the gap?"""
    timer_name, timer = get_timer()
    rows = []
    for tokens in (64, 128):
        args = build_case(tokens)
        reference = official_post_pre(*args)
        inp = build_inputs(args, tokens, reference)
        for warps in (4, 8, 16, 32):
            fn = functools.partial(fused_call, num_warps=warps)
            fn(*inp)
            torch.cuda.synchronize()
            for _ in range(30):
                fn(*inp)
            torch.cuda.synchronize()
            cold = time_call(fn, inp, options.repeat_ms, timer)
            kernel = profile_eager(fn, inp, iters=20)
            rows.append(
                {
                    "tokens": tokens,
                    "hidden": HIDDEN,
                    "arm": "single",
                    "num_warps": warps,
                    "graph_cold_us": cold["median_us"],
                    "graph_hot_us": hot_graph_us(fn, inp),
                    "kernel_device_us": kernel["total_device_us_per_call"],
                }
            )
            print(
                f"N={tokens:5d} warps={warps:3d} cold={cold['median_us']:8.3f} "
                f"kernel={kernel['total_device_us_per_call']:8.3f}",
                flush=True,
            )
        del inp, args, reference
        torch.cuda.empty_cache()
    return {"timer": timer_name, "rows": rows}


def metadata():
    files = [
        LAB / "benchmarks/audit_postpre_timing.py",
        LAB / "benchmarks/bench_post_pre.py",
        LAB / "benchmarks/bench_decode_two_stage.py",
        LAB / "benchmarks/bench_prefill_selection.py",
        LAB / "kernels/fused_post_pre.py",
        LAB / "kernels/decode_two_stage.py",
        LAB / "baselines/official.py",
        LAB / "reference/torch_post_pre.py",
        LAB / "tests/cases.py",
        ROOT / "vllm/models/hy_v4/nvidia/triton_ihc.py",
    ]
    return {
        "gpu": torch.cuda.get_device_name(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "official": provenance(),
        "dtype": "bfloat16",
        "hidden": HIDDEN,
        "channels": 4,
        "scalars": {"magnitude": MAGNITUDE, "hc_eps": HC_EPS, "norm_eps": NORM_EPS},
        "stage2_grid": [None, triton.cdiv(HIDDEN, 1024)],
        "stage1_splits": SPLITS,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=["timing", "profile", "variant"]
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-ms", type=int, default=100)
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(LAB):
        parser.error("Output must remain inside lab/ihc_fusion")
    if options.output.exists():
        parser.error("Refusing to overwrite existing audit output")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    phase = {
        "timing": timing_phase,
        "profile": profile_phase,
        "variant": variant_phase,
    }[options.phase]
    document = {"metadata": metadata(), "phase": options.phase, **phase(options)}
    options.output.write_text(json.dumps(document, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
