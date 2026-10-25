# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preallocated Official/Single (optional Two-Stage) kernel A/B; no Runtime Cross."""

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cudagraph, bench_gpu_time_with_cupti
from flashinfer.testing.utils import calculate_rotation_count

from lab.ihc_fusion.baselines.official import (
    official_post_pre,
    official_post_pre_into,
    provenance,
)
from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case


def baseline_call(*args):
    official_post_pre_into(*args[:9], out=args[9], partial=args[10])


def fused_call(*args, num_warps):
    fused_post_pre(*args[:9], out=args[9], num_warps=num_warps)


def two_call(*args, num_warps=8):
    from lab.ihc_fusion.kernels.decode_two_stage import decode_two_stage

    decode_two_stage(*args[:9], out=args[9], partial=args[10], num_warps=num_warps)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_timer():
    if (
        importlib.util.find_spec("cupti")
        and int((package_version("cupti-python") or "0").split(".")[0]) >= 13
    ):
        return "CUPTI + CUDA Graph, cold L2", bench_gpu_time_with_cupti
    return "CUDA Events + CUDA Graph, cold L2 rotating buffers (CUPTI absent)", None


def time_call(fn, inputs, repeat_ms, timer, *, cold=True):
    tensors = list(inputs[:6]) + list(inputs[9]) + [inputs[10]]
    rotations = calculate_rotation_count(tensors, torch.device("cuda"))
    # Cover a whole rotation cycle. Ten calls would leave decode weights in L2.
    calls = rotations * math.ceil(32 / rotations)
    if not cold:
        calls = 32
        samples = bench_gpu_time_with_cudagraph(
            fn,
            input_args=inputs,
            cold_l2_cache=False,
            num_iters_within_graph=calls,
            dry_run_time_ms=25,
            repeat_time_ms=repeat_ms,
        )
    elif timer is not None:
        samples = timer(
            fn,
            input_args=inputs,
            use_cuda_graph=True,
            cold_l2_cache=True,
            dry_run_time_ms=25,
            repeat_time_ms=repeat_ms,
        )
    else:
        samples = bench_gpu_time_with_cudagraph(
            fn,
            input_args=inputs,
            cold_l2_cache=True,
            num_iters_within_graph=calls,
            dry_run_time_ms=25,
            repeat_time_ms=repeat_ms,
        )
    return {
        "samples_us": [sample * 1000 for sample in samples],
        "median_us": statistics.median(samples) * 1000,
        "p10_us": sorted(samples)[int(0.1 * (len(samples) - 1))] * 1000,
        "p90_us": sorted(samples)[int(0.9 * (len(samples) - 1))] * 1000,
        "graph_replays": len(samples),
        "calls_per_graph": calls if timer is None or not cold else None,
        "rotations": rotations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="128,513,8192")
    parser.add_argument("--hidden", default="6144")
    parser.add_argument("--dtypes", default="bfloat16")
    parser.add_argument("--warps", default="default")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--two-warps", type=int, default=8)
    parser.add_argument("--cache", choices=("cold", "hot", "both"), default="cold")
    parser.add_argument("--magnitude", type=float, default=1.7)
    parser.add_argument("--hc-eps", type=float, default=2e-6)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    if opt.warmup < 0 or min(opt.rounds, opt.repeat_ms) < 1:
        parser.error("warmup >= 0 and positive rounds/repeat-ms required")
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    timer_name, timer = get_timer()
    doc = dict(
        metadata=dict(
            options=vars(opt),
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path("lab/ihc_fusion/kernels/fused_post_pre.py"),
                    Path("lab/ihc_fusion/kernels/decode_two_stage.py"),
                    Path("tests/models/test_hyv4_ihc_fusion.py"),
                )
            },
            command=sys.argv,
            gpu=torch.cuda.get_device_name(),
            gpu_selection=os.getenv("CUDA_VISIBLE_DEVICES"),
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            official=provenance(),
            timer=timer_name,
            hot_timer="CUDA Events + CUDA Graph, 32 calls, hot L2",
            graph_warmup_ms=25,
            seed=20261021,
            allocation_in_timing=False,
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
        rows=[],
    )
    functions = {"official": baseline_call}
    for warp in opt.warps.split(","):
        functions["single_" + warp] = functools.partial(
            fused_call, num_warps=None if warp == "default" else int(warp)
        )
    if opt.two_stage:
        functions["two"] = functools.partial(two_call, num_warps=opt.two_warps)
    for dtype in opt.dtypes.split(","):
        for d in map(int, opt.hidden.split(",")):
            for n in map(int, opt.tokens.split(",")):
                args = (
                    *make_case(n, d, DTYPES[dtype])[:6],
                    opt.magnitude,
                    opt.hc_eps,
                    1e-5,
                )
                official, golden = official_post_pre(*args), torch_post_pre(*args)
                check_outputs(official, golden)
                inputs, errors = {}, {}
                for name, fn in functions.items():
                    out = tuple(torch.empty_like(t) for t in official)
                    partial = torch.empty(
                        (n, triton.cdiv(4 * d, 1024), 9), device="cuda"
                    )
                    inputs[name] = (*args, out, partial)
                    fn(*inputs[name])
                    if name == "official":
                        for a, b in zip(out, official):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    errors[name] = dict(
                        golden=check_outputs(out, golden),
                        official=check_outputs(out, official),
                    )
                    for _ in range(opt.warmup):
                        fn(*inputs[name])
                torch.cuda.synchronize()
                for cache in ["cold", "hot"] if opt.cache == "both" else [opt.cache]:
                    timings = {name: [] for name in functions}
                    for step in range(opt.rounds):
                        names = list(functions)
                        offset = step % len(names)
                        for name in names[offset:] + names[:offset]:
                            timings[name].append(
                                time_call(
                                    functions[name],
                                    inputs[name],
                                    opt.repeat_ms,
                                    timer,
                                    cold=cache == "cold",
                                )
                            )
                    medians = {
                        name: statistics.median(t["median_us"] for t in samples)
                        for name, samples in timings.items()
                    }
                    doc["rows"].append(
                        dict(
                            tokens=n,
                            hidden=d,
                            dtype=dtype,
                            cache=cache,
                            timings=timings,
                            errors=errors,
                            median_us=medians,
                        )
                    )
                    for name, us in medians.items():
                        print(
                            f"N={n} D={d} {dtype} {cache} {name}: {us:.3f} us; "
                            f"speedup={medians['official'] / us:.3f}x",
                            flush=True,
                        )
                    opt.output.write_text(json.dumps(doc, indent=2, default=str) + "\n")
                del inputs, args, official, golden, out, partial
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
