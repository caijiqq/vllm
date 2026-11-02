# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run existing operator cases against the tuned plain Triton ablation."""

import pytest
import torch

import vllm.models.hy_v4.nvidia.smallhead_sparse_decode as production
from lab.hy4_smallhead.ablation_kernels import triton_split
from vllm.triton_utils import triton


def decode(
    q, kv, indices, softmax_scale, sinks=None, *, topk_length=None, num_splits=None
):
    tokens, heads, _ = q.shape
    out = q.new_empty((tokens, heads, 512))
    lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    if tokens == 0:
        return out, lse
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = triton.next_power_of_2(
        num_splits
        or min(32, triton.cdiv(sms, tokens), triton.cdiv(indices.shape[1], 64))
    )
    partial = torch.empty(
        (tokens, splits, heads, 514), device=q.device, dtype=torch.float32
    )
    triton_split[(tokens, splits)](
        q,
        kv,
        indices,
        topk_length,
        partial,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        indices.stride(1),
        kv.shape[0],
        heads,
        indices.shape[1],
        softmax_scale,
        splits,
        topk_length is not None,
        num_warps=4,
        num_stages=2,
    )
    production._softmax_merge[(tokens, heads)](
        partial,
        sinks,
        out,
        lse,
        heads,
        splits,
        sinks is not None,
        num_warps=4,
    )
    return out, lse


if __name__ == "__main__":
    # Replace only this process's operator entry, then reuse the existing tests.
    production.smallhead_sparse_decode = decode
    raise SystemExit(
        pytest.main(
            [
                "tests/kernels/attention/test_flashmla_sparse.py",
                "-k",
                "test_hy4_smallhead_decode or test_hy4_smallhead_unaligned_tail",
                "-q",
                "--confcutdir=tests/kernels/attention",
                "--tb=short",
            ]
        )
    )
