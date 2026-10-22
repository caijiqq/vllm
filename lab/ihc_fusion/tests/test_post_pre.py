# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the public Lab boundary, not model/runtime dispatch.

Contract: previous x/residual/gates and next parameters produce next single
stream, rounded residual, and FP32 gates. Guard against wrong ownership, removed
Post rounding, bad masks, reordered gates and channel accumulation. Direct GPU
operator comparisons are the cheapest level that exercises the Triton math.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from lab.ihc_fusion.baselines.official import (
    OFFICIAL,
    official_post_pre,
    official_post_pre_into,
    provenance,
)
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case

RECORDS = []


@pytest.fixture(scope="session", autouse=True)
def numeric_environment():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    if path := os.environ.get("IHC_CORRECTNESS_JSON"):
        Path(path).write_text(
            json.dumps({"provenance": provenance(), "cases": RECORDS}, indent=2) + "\n"
        )


def candidate(*args):
    if os.environ.get("IHC_TEST_BASELINE_ONLY") == "1":
        return official_post_pre(*args)
    from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre

    return fused_post_pre(*args)


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize(
    "tokens,hidden",
    [
        (0, 128),
        (1, 128),
        (3, 257),
        (17, 1024),
        (33, 3072),
        (1, 4096),
        (7, 7168),
        (2, 8192),
        (257, 4096),
    ],
)
def test_shapes(dtype, tokens, hidden):
    args = make_case(tokens, hidden, dtype)
    golden = torch_post_pre(*args)
    baseline = official_post_pre(*args)
    actual = candidate(*args)
    baseline_error = check_outputs(baseline, golden)
    golden_error = check_outputs(actual, golden)
    official_error = check_outputs(actual, baseline)
    RECORDS.append(
        {
            "dtype": str(dtype),
            "tokens": tokens,
            "hidden": hidden,
            "kind": "random",
            "baseline_vs_golden": baseline_error,
            "fused_vs_golden": golden_error,
            "fused_vs_official": official_error,
        }
    )


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("kind", ["zero", "tiny", "large", "cancel", "saturated"])
def test_numerical_boundaries(dtype, kind):
    args = make_case(5, 1025, dtype, kind=kind, offset=True)
    golden, baseline = torch_post_pre(*args), official_post_pre(*args)
    actual = candidate(*args)
    baseline_error = check_outputs(baseline, golden)
    golden_error = check_outputs(actual, golden)
    official_error = check_outputs(actual, baseline)
    RECORDS.append(
        {
            "dtype": str(dtype),
            "tokens": 5,
            "hidden": 1025,
            "kind": kind,
            "offset": True,
            "baseline_vs_golden": baseline_error,
            "fused_vs_golden": golden_error,
            "fused_vs_official": official_error,
        }
    )


def test_preallocated_baseline_matches_official_wrappers():
    args = make_case(7, 257, torch.bfloat16, offset=True)
    expected = official_post_pre(*args)
    out = tuple(torch.empty_like(tensor) for tensor in expected)
    partial = torch.empty((7, 2, 9), device="cuda", dtype=torch.float32)
    actual = official_post_pre_into(*args, out=out, partial=partial)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert OFFICIAL._ihc_pre_stage1.__class__.__name__ == "JITFunction"
    assert OFFICIAL._ihc_pre_stage2.__class__.__name__ == "JITFunction"
    assert OFFICIAL._ihc_post_kernel.__class__.__name__ == "JITFunction"


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("tokens,hidden", [(1024, 3072), (4096, 7168)])
def test_prefill(dtype, tokens, hidden):
    args = make_case(tokens, hidden, dtype)
    baseline, golden = official_post_pre(*args), torch_post_pre(*args)
    actual = candidate(*args)
    RECORDS.append(
        {
            "dtype": str(dtype),
            "tokens": tokens,
            "hidden": hidden,
            "kind": "prefill",
            "baseline_vs_golden": check_outputs(baseline, golden),
            "fused_vs_golden": check_outputs(actual, golden),
            "fused_vs_official": check_outputs(actual, baseline),
        }
    )


@pytest.mark.parametrize("channels,hidden", [(1, 1), (3, 63), (4, 255)])
def test_mask_and_channel_boundaries(channels, hidden):
    args = make_case(3, hidden, torch.bfloat16, channels=channels, offset=True)
    actual = candidate(*args)
    check_outputs(actual, torch_post_pre(*args))
    check_outputs(actual, official_post_pre(*args))


def test_preallocated_fused_has_one_gpu_launch():
    from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre

    args = make_case(3, 1025, torch.bfloat16)
    expected = official_post_pre(*args)
    out = tuple(torch.empty_like(t) for t in expected)
    for _ in range(3):
        fused_post_pre(*args, out=out)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fused_post_pre(*args, out=out)
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1, kernels
    assert "_fused_post_pre" in kernels[0]
    check_outputs(out, expected)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        official_post_pre(*args)
        torch.cuda.synchronize()
    baseline = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(baseline) == 3, baseline
    if path := os.environ.get("IHC_LAUNCH_JSON"):
        Path(path).write_text(
            json.dumps(
                {"fused_kernels": kernels, "official_kernels": baseline}, indent=2
            )
            + "\n"
        )
