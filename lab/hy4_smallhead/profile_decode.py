# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Nsight Compute target; inputs and JIT stay outside the profile range."""

import argparse

import torch
from cuda.bindings import runtime

from benchmarks.kernels.benchmark_hy4_smallhead_decode import (
    flashmla_call,
    make_inputs,
)
from vllm.models.hy_v4.nvidia.smallhead_sparse_decode import smallhead_sparse_decode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tokens", type=int, default=128)
parser.add_argument("--splits", type=int, default=0)
parser.add_argument("--independent-requests", action="store_true")
args = parser.parse_args()
torch.manual_seed(42)
q, cache, indices, sinks = make_inputs(
    args.tokens, rows=16384, independent_requests=args.independent_requests
)
native = flashmla_call(q, cache, indices, 0.0625, sinks)


def candidate():
    return smallhead_sparse_decode(
        q, cache, indices, 0.0625, sinks, num_splits=args.splits or None
    )


candidate()
native()
torch.accelerator.synchronize()
runtime.cudaProfilerStart()
candidate()
native()
torch.accelerator.synchronize()
runtime.cudaProfilerStop()
