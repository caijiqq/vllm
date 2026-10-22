# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-arm H100 experiment using the existing cold-L2 graph timer."""

import argparse
import csv
import functools
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cudagraph

from lab.ihc_fusion.baselines.official import OFFICIAL, official_post_pre, provenance
from lab.ihc_fusion.benchmarks.bench_post_pre import (
    baseline_call,
    fused_call,
    get_timer,
    time_call,
)
from lab.ihc_fusion.kernels.decode_two_stage import _post_pre_stage1, decode_two_stage
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case

# Hidden sizes in the case matrix; the same set selects the saved PTX samples.
HIDDEN_SIZES = (4096, 7168)
# Split counts observed for those hidden sizes; 21 keeps the intermediate runs.
SAVED_SPLITS = (16, 21, 28)


def two_call(*args, num_warps=8):
    decode_two_stage(*args[:9], out=args[9], partial=args[10], num_warps=num_warps)


def resources():
    records = []
    for function in (_post_pre_stage1, OFFICIAL._ihc_pre_stage2):
        for cache in function.device_caches.values():
            for kernel in cache[0].values():
                ptx = kernel.asm["ptx"]
                records.append(
                    dict(
                        name=function.__name__,
                        hash=kernel.hash,
                        registers=kernel.n_regs,
                        spills=kernel.n_spills,
                        shared=kernel.metadata.shared,
                        warps=kernel.metadata.num_warps,
                        constants={str(k): v for k, v in kernel.src.constants.items()},
                        signature={
                            str(k): str(v) for k, v in kernel.src.signature.items()
                        },
                        local_loads=ptx.count("ld.local"),
                        local_stores=ptx.count("st.local"),
                        barriers=ptx.count("bar.sync"),
                        fma=ptx.count("fma.rn.f32"),
                    )
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="bfloat16,float32,float16")
    parser.add_argument("--tokens", default="1,8,32,128,1024,4096")
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    lab = Path(__file__).resolve().parents[1]
    assert opt.output.resolve().is_relative_to(lab) and not opt.output.exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    timer_name, timer = get_timer()
    metadata = dict(
        command=" ".join(sys.argv),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        official=provenance(),
        timer=timer_name,
        rounds=3,
        repeat_ms=100,
        warmup_calls=30,
        graph_warmup_ms=25,
        gpu_index=os.environ.get("CUDA_VISIBLE_DEVICES"),
        stage1_block=1024,
        stage1_warps=opt.warps,
        allocation_in_timing=False,
        hashes={
            str(p.relative_to(lab)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (lab / "kernels/decode_two_stage.py", Path(__file__))
        },
        smi=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,utilization.gpu,clocks.sm,clocks.mem",
                "--format=csv",
            ],
            text=True,
        ),
    )
    rows = []
    for dtype in opt.dtypes.split(","):
        for d in HIDDEN_SIZES:
            for n in map(int, opt.tokens.split(",")):
                args = make_case(n, d, DTYPES[dtype])
                expected = official_post_pre(*args)
                golden = torch_post_pre(*args)
                functions = dict(
                    baseline=baseline_call,
                    single=functools.partial(fused_call, num_warps=None),
                    two=functools.partial(two_call, num_warps=opt.warps),
                )
                inputs, errors = {}, {}
                for name, fn in functions.items():
                    out = tuple(torch.empty_like(t) for t in expected)
                    p = torch.empty((n, triton.cdiv(4 * d, 1024), 9), device="cuda")
                    inputs[name] = (*args, out, p)
                    fn(*inputs[name])
                    errors[name] = dict(
                        golden=check_outputs(out, golden),
                        official=check_outputs(out, expected),
                    )
                    for _ in range(30):
                        fn(*inputs[name])
                timings = {name: [] for name in functions}
                hot = {name: [] for name in functions}
                for round_index in range(3):
                    order = list(functions)
                    if round_index % 2:
                        order.reverse()
                    for name in order:
                        fn, inp = functions[name], inputs[name]
                        timings[name].append(time_call(fn, inp, 100, timer))
                        hot[name].append(
                            statistics.median(
                                bench_gpu_time_with_cudagraph(
                                    fn,
                                    input_args=inp,
                                    cold_l2_cache=False,
                                    num_iters_within_graph=32,
                                    dry_run_time_ms=25,
                                    repeat_time_ms=100,
                                )
                            )
                            * 1000
                        )
                # Separate hot-cache graph trace: profiling is excluded from timing.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    functions["two"](*inputs["two"])
                graph.replay()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(20):
                        graph.replay()
                    torch.cuda.synchronize()
                events = [
                    e
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
                stages = {}
                for e in events:
                    stages.setdefault(e.name, []).append(e.device_time_total)
                assert len(stages) == 2 and min(map(len, stages.values())) >= 10, {
                    k: len(v) for k, v in stages.items()
                }
                row = dict(
                    dtype=dtype,
                    tokens=n,
                    hidden=d,
                    errors=errors,
                    cold_rounds=timings,
                    hot_rounds=hot,
                    stage1_grid=[n, triton.cdiv(4 * d, 1024)],
                    stage2_grid=[n, triton.cdiv(d, 1024)],
                    hot_profile_us={k: statistics.median(v) for k, v in stages.items()},
                    profile_samples={k: len(v) for k, v in stages.items()},
                )
                for name in functions:
                    row[name + "_us"] = statistics.median(
                        t["median_us"] for t in timings[name]
                    )
                    row[name + "_hot_us"] = statistics.median(hot[name])
                row["single_speedup"] = row["baseline_us"] / row["single_us"]
                row["two_speedup"] = row["baseline_us"] / row["two_us"]
                rows.append(row)
                print(
                    dtype,
                    n,
                    d,
                    {
                        k: round(row[k], 3)
                        for k in ("baseline_us", "single_us", "two_us", "two_speedup")
                    },
                    flush=True,
                )
                opt.output.write_text(
                    json.dumps(
                        dict(metadata=metadata, rows=rows, resources=resources()),
                        indent=2,
                    )
                )
                del args, golden, expected, inputs, graph, out, p, inp
                torch.cuda.empty_cache()
    fields = [
        "dtype",
        "tokens",
        "hidden",
        "baseline_us",
        "single_us",
        "two_us",
        "single_speedup",
        "two_speedup",
        "baseline_hot_us",
        "single_hot_us",
        "two_hot_us",
    ]
    with opt.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for cache in _post_pre_stage1.device_caches.values():
        for kernel in cache[0].values():
            # Keep one representative PTX per dtype and hidden size.
            constants = kernel.src.constants
            # Kernel constants: index 6 is the hidden size, 7 the split count.
            hidden, splits = constants.get((6,)), constants.get((7,))
            if hidden in HIDDEN_SIZES and splits in SAVED_SPLITS:
                opt.output.parent.joinpath(f"stage1_{kernel.hash[:12]}.ptx").write_text(
                    kernel.asm["ptx"]
                )


if __name__ == "__main__":
    main()
