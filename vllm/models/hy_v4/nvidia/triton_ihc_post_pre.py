# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 C=4 Post + next Pre + RMSNorm fusion."""

import torch

from vllm.triton_utils import tl, triton

_IHC_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=_IHC_AUTOTUNE_CONFIGS,
    key=["D", "BLOCK_K"],
    cache_results=True,
)
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
    rms_weight_ptr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    HC_EPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_K)
    rows = tl.arange(0, 8)
    mix_lanes = tl.full((8, BLOCK_K), 0, tl.float32)
    square_lanes = tl.full((BLOCK_K,), 0, tl.float32)
    for split in range(tl.cdiv(4 * D, BLOCK_K)):
        k = split * BLOCK_K + lanes
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
    output_rms = tl.rsqrt(tl.sum(output * output, 0) / D + RMS_EPS)
    rms_weight = tl.load(rms_weight_ptr + d, mask=d < D, other=0).to(tl.float32)
    output = output * output_rms * rms_weight
    tl.store(output_ptr + token * D + d, output, mask=d < D)


def triton_ihc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    previous_post: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RMS-normalized next input, rounded residual and next gates.

    Previous gates belong to Attention; projection parameters belong to MLP.
    The caller provides contiguous BF16 activations and FP32 parameters.
    """
    output = torch.empty_like(x)
    updated = torch.empty_like(residual)
    post = torch.empty_like(previous_post)
    _fused_post_pre[(x.shape[0],)](
        x,
        residual,
        previous_post,
        weight,
        scale,
        base,
        output,
        updated,
        post,
        rms_weight,
        D=x.shape[1],
        BLOCK_K=2048 if x.shape[0] <= 128 else 1024,
        BLOCK_D=triton.next_power_of_2(x.shape[1]),
        MAGNITUDE=magnitude,
        HC_EPS=hc_eps,
        NORM_EPS=norm_eps,
        RMS_EPS=rms_eps,
        enable_fp_fusion=True,
    )
    return output, updated, post
