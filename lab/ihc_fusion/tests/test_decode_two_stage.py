# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard Post rounding, partial layout, and replay for the independent C=4 path."""

import pytest
import torch
import triton

from lab.ihc_fusion.baselines.official import official_post_pre_into
from lab.ihc_fusion.kernels.decode_two_stage import decode_two_stage
from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize(
    "n,d,kind",
    [(n, d, "random") for n in (1, 8, 32, 128) for d in (4096, 7168)]
    + [(3, 257, k) for k in ("random", "zero", "tiny", "large", "cancel", "saturated")]
    + [(0, 128, "random")],
)
def test_partials_and_outputs(dtype, n, d, kind):
    args = make_case(n, d, dtype, kind=kind, offset=True)
    expected = torch_post_pre(*args)
    out = tuple(torch.empty_like(t) for t in expected)
    baseline = tuple(torch.empty_like(t) for t in expected)
    p = torch.empty((n, triton.cdiv(4 * d, 1024), 9), device="cuda")
    q = torch.empty_like(p)
    official_post_pre_into(*args, out=baseline, partial=q)
    decode_two_stage(*args, out=out, partial=p)
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    check_outputs(out, expected)
    check_outputs(out, baseline)
    check_outputs(out, fused_post_pre(*args))
    if not n:
        return
    # Replay must consume new input contents, without stale partials.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_two_stage(*args, out=out, partial=p)
    for factor in (0.5, -1.0):
        args[0].mul_(factor)
        graph.replay()
        check_outputs(out, torch_post_pre(*args))


def test_two_launches():
    args = make_case(1, 7168, torch.bfloat16)
    out = decode_two_stage(*args)
    partial = torch.empty((1, 28, 9), device="cuda")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        decode_two_stage(*args, out=out, partial=partial)
        torch.cuda.synchronize()
    names = [
        e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(names) == 2, names
    assert "_post_pre_stage1" in names[0] and "_ihc_pre_stage2" in names[1]
