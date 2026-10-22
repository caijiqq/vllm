# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode Post + Pre1 fusion followed by the pinned official Pre2."""

import torch
import triton
import triton.language as tl

from lab.ihc_fusion.baselines.official import OFFICIAL


@triton.jit
def _post_pre_stage1(
    X,
    R,
    Previous,
    W,
    Updated,
    Partial,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    k = split * BLOCK + tl.arange(0, BLOCK)
    mask = k < 4 * D
    x = tl.load(X + token * D + k % D, mask, 0).to(tl.float32)
    r = tl.load(R + token * (4 * D) + k, mask, 0).to(tl.float32)
    gate = tl.load(Previous + token * 4 + k // D, mask, 0)
    # The official Post materializes this cast before Pre1 reads its input.
    z = (gate * x + r).to(Updated.dtype.element_ty).to(tl.float32)
    tl.store(Updated + token * (4 * D) + k, z, mask)
    c = tl.arange(0, 4)
    offsets = c[:, None] * (4 * D) + k[None, :]
    pre = tl.sum(tl.load(W + offsets, mask[None, :], 0) * z[None, :], 1)
    post = tl.sum(tl.load(W + 4 * (4 * D) + offsets, mask[None, :], 0) * z[None, :], 1)
    p = Partial + (token * S + split) * 9
    tl.store(p, tl.sum(z * z, 0))
    tl.store(p + 1 + c, pre)
    tl.store(p + 5 + c, post)


def decode_two_stage(
    x,
    residual,
    previous_post,
    weight,
    scale,
    base,
    magnitude,
    hc_eps,
    norm_eps,
    *,
    out=None,
    partial=None,
    num_warps=8,
):
    """Return (next_x, rounded_residual, next_post); contiguous C=4 Lab inputs.

    Preallocated outputs and FP32 [N, ceil(4D/1024), 9] partials allow graph
    capture without allocations. Buffers must not alias inputs or each other.
    """
    n, c, d = residual.shape
    assert c == 4 and d > 0
    assert x.shape == (n, d) and previous_post.shape == (n, 4)
    assert weight.shape == (8, 4 * d) and scale.shape == (2,) and base.shape == (8,)
    assert x.dtype == residual.dtype
    assert x.dtype in (torch.float32, torch.bfloat16, torch.float16)
    assert all(
        t.is_cuda and t.is_contiguous() and t.device == x.device
        for t in (x, residual, previous_post, weight, scale, base)
    )
    assert all(t.dtype == torch.float32 for t in (previous_post, weight, scale, base))
    splits = triton.cdiv(4 * d, 1024)
    if out is None:
        out = (
            torch.empty_like(x),
            torch.empty_like(residual),
            torch.empty_like(previous_post),
        )
    if partial is None:
        partial = torch.empty((n, splits, 9), device=x.device, dtype=torch.float32)
    assert partial.shape == (n, splits, 9) and partial.dtype == torch.float32
    for tensor, expected in zip(out, (x, residual, previous_post)):
        assert tensor.shape == expected.shape and tensor.dtype == expected.dtype
        assert tensor.device == x.device and tensor.is_contiguous()
    assert partial.device == x.device and partial.is_contiguous()
    if not n:
        return out
    output, updated, post = out
    _post_pre_stage1[(n, splits)](
        x,
        residual,
        previous_post,
        weight,
        updated,
        partial,
        D=d,
        S=splits,
        BLOCK=1024,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    OFFICIAL._ihc_pre_stage2[(n, triton.cdiv(d, 1024))](
        updated,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=d,
        K_TOTAL=4 * d,
        HC_MULT=4,
        HC_POW2=4,
        NUM_SPLITS=splits,
        PARTIAL_STRIDE=9,
        BLOCK_D=1024,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
