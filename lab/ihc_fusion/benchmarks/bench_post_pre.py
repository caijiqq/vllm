# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H100 CUDA Graph benchmark; run as a module from the repository root."""

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
from lab.ihc_fusion.kernels.fused_post_pre import _fused_post_pre, fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "lab/ihc_fusion"


def baseline_call(*args):
    official_post_pre_into(*args[:9], out=args[9], partial=args[10])


def fused_call(*args, num_warps):
    fused_post_pre(*args[:9], out=args[9], num_warps=num_warps)


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


def time_call(fn, inputs, repeat_ms, timer):
    tensors = list(inputs[:6]) + list(inputs[9]) + [inputs[10]]
    rotations = calculate_rotation_count(tensors, torch.device("cuda"))
    # Cover a whole rotation cycle. Ten calls would leave decode weights in L2.
    calls = rotations * math.ceil(32 / rotations)
    if timer is not None:
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
        "median_us": statistics.median(samples) * 1000,
        "p10_us": sorted(samples)[int(0.1 * (len(samples) - 1))] * 1000,
        "p90_us": sorted(samples)[int(0.9 * (len(samples) - 1))] * 1000,
        "graph_replays": len(samples),
        "calls_per_graph": calls,
        "rotations": rotations,
    }


def compiled_resources():
    result = []
    for cache in _fused_post_pre.device_caches.values():
        for kernel in cache[0].values():
            result.append(
                {
                    "hash": kernel.hash,
                    "registers_per_thread": kernel.n_regs,
                    "spill_count": kernel.n_spills,
                    "shared_bytes": kernel.metadata.shared,
                    "num_warps": kernel.metadata.num_warps,
                    "constants": {str(k): v for k, v in kernel.src.constants.items()},
                    "signature": {
                        str(k): str(v) for k, v in kernel.src.signature.items()
                    },
                    "ptx_local_loads": kernel.asm["ptx"].count("ld.local"),
                    "ptx_local_stores": kernel.asm["ptx"].count("st.local"),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,8,32,128,1024,4096")
    parser.add_argument("--hidden", default="4096,7168")
    parser.add_argument("--dtypes", default="float32,bfloat16,float16")
    parser.add_argument(
        "--warps",
        default="default",
        help="default, or comma-separated coarse-sweep candidates",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat-ms", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=LAB / "results/h100_benchmark.json"
    )
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(LAB):
        parser.error("Output must remain inside lab/ihc_fusion")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timer_name, timer = get_timer()
    device = torch.cuda.get_device_properties(0)
    metadata = {
        "command": " ".join(sys.argv),
        "gpu": device.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory": device.total_memory,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": package_version("flashinfer-python"),
        "python": sys.version,
        "timer": timer_name,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "official": provenance(),
        "warmup_calls": 30,
        "graph_warmup_ms": 25,
        "repeat_ms_per_round": options.repeat_ms,
        "rounds": options.rounds,
        "input_seed": 20261021,
        "activation_semantics": "FP32 is Lab direct-call extension; BF16/FP16 official",
        "allocation_in_timing": False,
        "baseline_launches": 3,
        "fused_launches": 1,
        "baseline_config": "fixed original: Post 4 warps, Pre stage1 8, stage2 4",
        "environment": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "TRITON_CACHE_DIR",
                "CUDA_MODULE_LOADING",
            ]
        },
        "source_hashes": {
            str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                LAB / "kernels/fused_post_pre.py",
                LAB / "baselines/official.py",
                LAB / "reference/torch_post_pre.py",
                LAB / "tests/cases.py",
                LAB / "tests/test_post_pre.py",
                Path(__file__),
            ]
        },
        "nvidia_smi_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,temperature.gpu,"
                "utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
    }
    rows = []
    doc = {"metadata": metadata, "rows": rows}
    print(timer_name, flush=True)
    for dtype_name in options.dtypes.split(","):
        for hidden in map(int, options.hidden.split(",")):
            for tokens in map(int, options.tokens.split(",")):
                args = make_case(tokens, hidden, DTYPES[dtype_name])
                official = official_post_pre(*args)
                golden = torch_post_pre(*args)
                baseline_error = check_outputs(official, golden)
                out_baseline = tuple(torch.empty_like(t) for t in official)
                out_fused = tuple(torch.empty_like(t) for t in official)
                partial = torch.empty(
                    (tokens, triton.cdiv(4 * hidden, 1024), 9),
                    dtype=torch.float32,
                    device="cuda",
                )
                baseline_inputs = (*args, out_baseline, partial)
                fused_inputs = (*args, out_fused, partial)
                baseline_call(*baseline_inputs)
                for actual, expected in zip(out_baseline, official):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                warps = (
                    [None]
                    if options.warps == "default"
                    else list(map(int, options.warps.split(",")))
                )
                functions = {"baseline": (baseline_call, baseline_inputs)}
                errors = {}
                for warp in warps:
                    label = "default" if warp is None else str(warp)
                    fn = functools.partial(fused_call, num_warps=warp)
                    fn(*fused_inputs)
                    errors[label] = {
                        "vs_golden": check_outputs(out_fused, golden),
                        "vs_official": check_outputs(out_fused, official),
                    }
                    functions[label] = fn, fused_inputs
                for fn, inputs in functions.values():
                    for _ in range(30):
                        fn(*inputs)
                torch.cuda.synchronize()
                timings = {label: [] for label in functions}
                for round_index in range(options.rounds):
                    order = list(functions)
                    if round_index % 2:
                        order.reverse()
                    for label in order:
                        fn, inputs = functions[label]
                        timings[label].append(
                            time_call(fn, inputs, options.repeat_ms, timer)
                        )
                baseline_us = statistics.median(
                    t["median_us"] for t in timings["baseline"]
                )
                for label in errors:
                    fused_us = statistics.median(t["median_us"] for t in timings[label])
                    # Lower-bound boundary traffic: read x, R, prev gates and
                    # next params once; write next x, R and gates once. Weight
                    # reuse across tokens means this is not actual HBM traffic.
                    boundary_bytes = (
                        2 * tokens * hidden * DTYPES[dtype_name].itemsize
                        + 2 * tokens * 4 * hidden * DTYPES[dtype_name].itemsize
                        + 2 * tokens * 4 * 4
                        + 8 * 4 * hidden * 4
                        + 10 * 4
                    )
                    row = {
                        "dtype": dtype_name,
                        "tokens": tokens,
                        "hidden": hidden,
                        "channels": 4,
                        "num_warps": label,
                        "effective_num_warps": (16 if tokens <= 128 else 4)
                        if label == "default"
                        else int(label),
                        "baseline_us": baseline_us,
                        "fused_us": fused_us,
                        "speedup": baseline_us / fused_us,
                        "boundary_min_bytes": boundary_bytes,
                        "baseline_boundary_gbps": boundary_bytes / baseline_us / 1000,
                        "fused_boundary_gbps": boundary_bytes / fused_us / 1000,
                        "errors": errors[label],
                        "baseline_vs_golden": baseline_error,
                        "baseline_rounds": timings["baseline"],
                        "fused_rounds": timings[label],
                    }
                    rows.append(row)
                    print(
                        f"{dtype_name:8s} N={tokens:4d} D={hidden} warps={label:7s} "
                        f"baseline={baseline_us:.3f}us fused={fused_us:.3f}us "
                        f"speedup={row['speedup']:.3f}",
                        flush=True,
                    )
                doc["compiled_resources"] = compiled_resources()
                options.output.write_text(json.dumps(doc, indent=2, default=str) + "\n")
                del args, official, golden, out_baseline, out_fused, partial
                del baseline_inputs, fused_inputs, functions
                torch.cuda.empty_cache()
    metadata["nvidia_smi_after"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,"
            "memory.used,clocks.sm,clocks.mem,power.draw",
            "--format=csv",
        ],
        text=True,
    )
    options.output.write_text(json.dumps(doc, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
