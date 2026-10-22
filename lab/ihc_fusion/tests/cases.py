# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seeded input cases and fixed, dtype-derived numerical acceptance rules."""

import torch

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def make_case(tokens, hidden, dtype, *, channels=4, kind="random", offset=False):
    generator = torch.Generator(device="cuda").manual_seed(20261021)

    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, generator=generator, device="cuda", dtype=dtype)

    x = randn((tokens, hidden), dtype)
    residual = randn((tokens, channels, hidden), dtype)
    previous = randn((tokens, channels)).sigmoid() * 2
    weight = randn((2 * channels, channels * hidden)) / (channels * hidden) ** 0.5
    scale = torch.tensor([0.7, -0.4], device="cuda", dtype=torch.float32)
    base = randn((2 * channels,)) * 0.2
    if kind == "zero":
        x.zero_()
        residual.zero_()
    elif kind == "tiny":
        x.mul_(1e-4)
        residual.mul_(1e-4)
    elif kind == "large":
        x.mul_(100)
        residual.mul_(100)
    elif kind == "cancel":
        residual.copy_((-previous[..., None] * x.float()[:, None]).to(dtype))
    elif kind == "saturated":
        scale.mul_(100)
        base.copy_(torch.linspace(-30, 30, 2 * channels, device="cuda"))
    elif kind != "random":
        raise ValueError(kind)
    if offset:

        def shifted(tensor):
            storage = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
            view = storage[1:].view(tensor.shape)
            view.copy_(tensor)
            return view

        x, residual, previous, weight, scale, base = map(
            shifted, (x, residual, previous, weight, scale, base)
        )
    return x, residual, previous, weight, scale, base, 1.7, 2e-6, 1e-5


def metrics(actual, expected):
    difference = actual.double() - expected.double()
    reference_norm = expected.double().norm().item()
    return {
        "max_abs": difference.abs().max().item() if actual.numel() else 0.0,
        "relative_l2": difference.norm().item() / max(reference_norm, 1e-30),
        "rms_abs": difference.square().mean().sqrt().item() if actual.numel() else 0.0,
    }


def check_outputs(actual, expected):
    """Require exact Residual, FP32 gate accuracy, and <=2 activation ulps.

    The 2*eps relative bound permits one final low precision rounding boundary
    on either side. FP32 uses 2e-5 relative + absolute for parallel GEMV/sigmoid
    differences. An 8*FP32-eps*sum(abs(channels)) absolute budget covers
    FP32 channel cancellation; relative L2 also stays below 2e-6 or dtype eps.
    Fixed before any fused implementation; never inferred from observed errors.
    """
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)
    dtype = actual[0].dtype
    if dtype == torch.float32:
        rtol, atol, l2_bound = 2e-5, 2e-5, 2e-6
    else:
        eps = torch.finfo(dtype).eps
        rtol, atol, l2_bound = 2 * eps, 2e-5, eps
    # Four channels: gamma_C plus multiplication/sigmoid rounding budget.
    cancellation_budget = (
        8 * torch.finfo(torch.float32).eps * expected[1].float().abs().sum(dim=1)
    )
    output_error = (actual[0].float() - expected[0].float()).abs()
    budget = atol + rtol * expected[0].float().abs() + cancellation_budget
    assert torch.isfinite(actual[0]).all()
    assert (output_error <= budget).all(), {
        "max_error": output_error.max().item(),
        "max_budget_ratio": (output_error / budget).max().item(),
    }
    for index, bound in [(0, l2_bound), (2, 2e-6)]:
        error = metrics(actual[index], expected[index])
        assert error["relative_l2"] <= bound, error
    return {
        name: metrics(a, b)
        for name, a, b in zip(("output", "residual", "post_gates"), actual, expected)
    }
