# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Targeted HY4 public-config Prefill experiments; no runtime selection."""

import argparse
import functools
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cudagraph

from lab.ihc_fusion.baselines.official import official_post_pre, provenance
from lab.ihc_fusion.benchmarks.bench_decode_two_stage import two_call
from lab.ihc_fusion.benchmarks.bench_post_pre import (
    baseline_call,
    fused_call,
    get_timer,
    time_call,
)
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="64,128,256,512,768,1024,2048,4096,8192")
    parser.add_argument("--output", required=True, type=Path)
    opt = parser.parse_args()
    lab = Path(__file__).resolve().parents[1]
    assert opt.output.resolve().is_relative_to(lab) and not opt.output.exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    timer_name, timer = get_timer()
    metadata = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        gpu=torch.cuda.get_device_name(),
        gpu_index=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        official=provenance(),
        timer=timer_name,
        warmup=30,
        graph_warmup_ms=25,
        rounds=3,
        repeat_ms=100,
        dtype="bfloat16",
        hidden=6144,
        channels=4,
        magnitude=2.0,
        hc_eps=1e-6,
        norm_eps=1e-5,
        seed=20261021,
        weights="synthetic seeded; not checkpoint weights or model trace",
        selection="none",
        two_warps=8,
        single_warps="frozen default",
        hashes={
            str(f.relative_to(lab)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                lab / "kernels/fused_post_pre.py",
                lab / "kernels/decode_two_stage.py",
                lab / "tests/cases.py",
            ]
        },
        gpu_state=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,utilization.gpu,clocks.sm,clocks.mem",
                "--format=csv",
            ],
            text=True,
        ),
    )
    rows = []
    for n in map(int, opt.tokens.split(",")):
        args = (*make_case(n, 6144, torch.bfloat16)[:6], 2.0, 1e-6, 1e-5)
        golden, official = torch_post_pre(*args), official_post_pre(*args)
        functions = dict(
            official=baseline_call,
            single=functools.partial(fused_call, num_warps=None),
            two=two_call,
        )
        inputs, errors = {}, {}
        for name, fn in functions.items():
            out = tuple(torch.empty_like(t) for t in official)
            partial = torch.empty((n, 24, 9), device="cuda")
            inputs[name] = (*args, out, partial)
            fn(*inputs[name])
            errors[name] = dict(
                golden=check_outputs(out, golden), official=check_outputs(out, official)
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(*inputs[name])
            graph.replay()
            check_outputs(out, official)
            for _ in range(30):
                fn(*inputs[name])
        cold, hot = ({k: [] for k in functions} for _ in range(2))
        for round_idx in range(3):
            # Rotate first position so each arm leads one round.
            names = list(functions)
            for name in names[round_idx:] + names[:round_idx]:
                fn, inp = functions[name], inputs[name]
                cold[name].append(time_call(fn, inp, 100, timer))
                samples = bench_gpu_time_with_cudagraph(
                    fn,
                    input_args=inp,
                    cold_l2_cache=False,
                    num_iters_within_graph=32,
                    dry_run_time_ms=25,
                    repeat_time_ms=100,
                )
                hot[name].append(statistics.median(samples) * 1000)
        row = dict(
            tokens=n, hidden=6144, dtype="bfloat16", errors=errors, cold=cold, hot=hot
        )
        for name in functions:
            row[name + "_us"] = statistics.median(t["median_us"] for t in cold[name])
            row[name + "_hot_us"] = statistics.median(hot[name])
        rows.append(row)
        print(
            n, {k: round(v, 3) for k, v in row.items() if k.endswith("_us")}, flush=True
        )
        opt.output.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2))
        del args, golden, official, inputs, out, partial, graph, inp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
