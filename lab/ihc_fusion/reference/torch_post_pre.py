# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent FP32 PyTorch semantic reference, including the Post cast."""

import torch


@torch.no_grad()
def torch_post_pre(
    x, residual, previous_post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    """Return (next_x, next_residual, next_post) for two adjacent sub-blocks."""
    updated = (previous_post[..., None] * x.float()[:, None, :] + residual.float()).to(
        x.dtype
    )
    channels = residual.shape[1]
    flat = updated.flatten(1).float()
    reciprocal_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(flat, weight) * reciprocal_rms
    pre = torch.sigmoid(mixes[:, :channels] * scale[0] + base[:channels]) + hc_eps
    post = (
        magnitude * torch.sigmoid(mixes[:, channels:] * scale[1] + base[channels:])
        + hc_eps
    )
    output = torch.zeros_like(x, dtype=torch.float32)
    for channel in range(channels):
        output = output + pre[:, channel, None] * updated[:, channel].float()
    return output.to(x.dtype), updated, post
