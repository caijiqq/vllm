# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone single-launch HY4 iHC Post -> next Pre for contiguous tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_sum(values, SPLITS: tl.constexpr, PAD_SPLITS: tl.constexpr):
    partials = tl.sum(tl.reshape(values, (PAD_SPLITS, 1024)), axis=1)
    split_ids = tl.arange(0, PAD_SPLITS)
    total = tl.full((), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        total += tl.sum(tl.where(split_ids == split, partials, 0.0), axis=0)
    return total


@triton.jit
def _fused_post_pre_general(
    x_ptr,
    residual_ptr,
    previous_ptr,
    weight_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    updated_ptr,
    next_post_ptr,
    D: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CP: tl.constexpr,
    SPLITS: tl.constexpr,
    PAD_SPLITS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    HC_EPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    valid = k < K
    channel = k // D
    hidden = k % D
    x = tl.load(x_ptr + token * D + hidden, mask=valid, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + token * K + k, mask=valid, other=0).to(tl.float32)
    previous = tl.load(previous_ptr + token * C + channel, mask=valid, other=0)
    updated = (previous * x + residual).to(updated_ptr.dtype.element_ty)
    tl.store(updated_ptr + token * K + k, updated, mask=valid)
    z = updated.to(tl.float32)
    inv_rms = tl.rsqrt(_chunk_sum(z * z, SPLITS, PAD_SPLITS) / K + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    c = tl.arange(0, CP)
    pre_gates = tl.full((CP,), 0, tl.float32)
    for row in tl.static_range(C):
        pre_weight = tl.load(weight_ptr + row * K + k, mask=valid, other=0)
        pre_mix = _chunk_sum(pre_weight * z, SPLITS, PAD_SPLITS)
        pre = tl.sigmoid(pre_mix * inv_rms * pre_scale + tl.load(base_ptr + row))
        pre += HC_EPS
        pre_gates = tl.where(c == row, pre, pre_gates)
        post_weight = tl.load(weight_ptr + (C + row) * K + k, mask=valid, other=0)
        post_mix = _chunk_sum(post_weight * z, SPLITS, PAD_SPLITS)
        post = (
            MAGNITUDE
            * tl.sigmoid(post_mix * inv_rms * post_scale + tl.load(base_ptr + C + row))
            + HC_EPS
        )
        tl.store(next_post_ptr + token * C + row, post)
    d = tl.arange(0, BLOCK_D)
    output = tl.full((BLOCK_D,), 0, tl.float32)
    for row in tl.static_range(C):
        indices = tl.minimum(row * D + d, K - 1)
        values = tl.gather(z, indices, axis=0)
        gate = tl.sum(tl.where(c == row, pre_gates, 0.0), axis=0)
        output += gate * values
    tl.store(output_ptr + token * D + d, output, mask=d < D)


@triton.jit
def _fused_post_pre(
    x_ptr,
    residual_ptr,
    previous_ptr,
    weight_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    updated_ptr,
    next_post_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    HC_EPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, 1024)
    rows = tl.arange(0, 8)
    mix_lanes = tl.full((8, 1024), 0, tl.float32)
    square_lanes = tl.full((1024,), 0, tl.float32)
    for split in range(tl.cdiv(4 * D, 1024)):
        k = split * 1024 + lanes
        valid = k < 4 * D
        x = tl.load(x_ptr + token * D + k % D, mask=valid, other=0).to(tl.float32)
        previous = tl.load(previous_ptr + token * 4 + k // D, mask=valid, other=0)
        residual = tl.load(residual_ptr + token * 4 * D + k, mask=valid, other=0).to(
            tl.float32
        )
        rounded = (previous * x + residual).to(updated_ptr.dtype.element_ty)
        tl.store(updated_ptr + token * 4 * D + k, rounded, mask=valid)
        z = rounded.to(tl.float32)
        weights = tl.load(
            weight_ptr + rows[:, None] * 4 * D + k[None, :],
            mask=valid[None, :],
            other=0,
        )
        mix_lanes += weights * z[None, :]
        square_lanes += z * z
    inv_rms = tl.rsqrt(tl.sum(square_lanes, 0) / (4 * D) + NORM_EPS)
    mixes = tl.sum(mix_lanes, 1)
    scale = tl.load(scale_ptr + rows // 4)
    base = tl.load(base_ptr + rows)
    gates = tl.sigmoid(mixes * inv_rms * scale + base)
    pre, post = tl.split(tl.trans(tl.reshape(gates, (2, 4))))
    pre += HC_EPS
    post = MAGNITUDE * post + HC_EPS
    channels = tl.arange(0, 4)
    tl.store(next_post_ptr + token * 4 + channels, post)
    d = tl.arange(0, BLOCK_D)
    x_out = tl.load(x_ptr + token * D + d, mask=d < D, other=0).to(tl.float32)
    output = tl.full((BLOCK_D,), 0, tl.float32)
    for row in tl.static_range(4):
        previous = tl.load(previous_ptr + token * 4 + row)
        residual = tl.load(
            residual_ptr + token * 4 * D + row * D + d, mask=d < D, other=0
        ).to(tl.float32)
        rounded = (previous * x_out + residual).to(updated_ptr.dtype.element_ty)
        gate = tl.sum(tl.where(channels == row, pre, 0.0), 0)
        output += gate * rounded.to(tl.float32)
    tl.store(output_ptr + token * D + d, output, mask=d < D)


def fused_post_pre(
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
    num_warps=None,
):
    """Return (next_x, next_residual, next_post) using one GPU kernel.

    Previous gates belong to the completed sub-block. Weight/scale/base and scalar
    hyperparameters belong to the next sub-block. All inputs must be contiguous;
    storage-offset contiguous views are supported. Activations share dtype;
    weights, biases, scales and gates are FP32. FP32 activation is a Lab extension
    beyond the official BF16/FP16 runtime dispatch. Empty input launches nothing.
    The four-channel path accumulates 1024 FP32 lanes across flattened chunks,
    then tree-reduces lanes. Its GEMV/square-sum tree differs from the official
    tree. Post rounding and sequential output accumulation are preserved. It
    recomputes rounded Post values for output rather than reloading updated.
    Optional output buffers exclude allocation from graph microbenchmarks.
    """
    if x.ndim != 2 or residual.ndim != 3:
        raise ValueError("Expected x [N,D] and residual [N,C,D]")
    tokens, channels, hidden = residual.shape
    k = channels * hidden
    if hidden <= 0 or channels <= 0:
        raise ValueError("Hidden size and channel count must be positive")
    expected = [
        (tokens, hidden),
        (tokens, channels, hidden),
        (tokens, channels),
        (2 * channels, k),
        (2,),
        (2 * channels,),
    ]
    tensors = (x, residual, previous_post, weight, scale, base)
    if any(tuple(t.shape) != shape for t, shape in zip(tensors, expected)):
        raise ValueError("Inconsistent iHC tensor shapes")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported activation dtype")
    if residual.dtype != x.dtype or any(t.dtype != torch.float32 for t in tensors[2:]):
        raise ValueError("Residual dtype must match x; gates and parameters need FP32")
    if any(
        not t.is_cuda or t.device != x.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("All inputs must be contiguous CUDA tensors on one device")
    shapes = [(tokens, hidden), (tokens, channels, hidden), (tokens, channels)]
    dtypes = [x.dtype, x.dtype, torch.float32]
    if out is None:
        out = tuple(
            torch.empty(shape, dtype=dtype, device=x.device)
            for shape, dtype in zip(shapes, dtypes)
        )
    elif len(out) != 3 or any(
        tuple(t.shape) != shape
        or t.dtype != dtype
        or t.device != x.device
        or not t.is_contiguous()
        for t, shape, dtype in zip(out, shapes, dtypes)
    ):
        raise ValueError("Invalid output buffers")
    if tokens == 0:
        return out
    block_k = max(1024, triton.next_power_of_2(k))
    if num_warps is None:
        num_warps = 16 if channels == 4 and tokens <= 128 else 4
    if channels == 4:
        _fused_post_pre[(tokens,)](
            x,
            residual,
            previous_post,
            weight,
            scale,
            base,
            *out,
            D=hidden,
            BLOCK_D=triton.next_power_of_2(hidden),
            MAGNITUDE=magnitude,
            HC_EPS=hc_eps,
            NORM_EPS=norm_eps,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        return out
    _fused_post_pre_general[(tokens,)](
        x,
        residual,
        previous_post,
        weight,
        scale,
        base,
        *out,
        D=hidden,
        C=channels,
        K=k,
        BLOCK_K=block_k,
        BLOCK_D=triton.next_power_of_2(hidden),
        CP=triton.next_power_of_2(channels),
        SPLITS=triton.cdiv(k, 1024),
        PAD_SPLITS=block_k // 1024,
        MAGNITUDE=magnitude,
        HC_EPS=hc_eps,
        NORM_EPS=norm_eps,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out
