# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production BF16/C4/D6144 math, independent references and actual GPU launches."""

import json
import os
from pathlib import Path

import pytest
import torch

from lab.ihc_fusion.baselines.official import (
    official_post_pre,
    official_post_pre_into,
    provenance,
)
from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case
from vllm.models.hy_v4.nvidia.triton_ihc_post_pre import triton_ihc_post_pre

RECORDS = []


@pytest.fixture(scope="module", autouse=True)
def numeric_environment():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous
    if path := os.getenv("IHC_CORRECTNESS_JSON"):
        Path(path).write_text(json.dumps(dict(provenance=provenance(), cases=RECORDS)))


@pytest.mark.parametrize(
    "n,kind",
    [(n, "random") for n in (128, 513, 8192)]
    + [(5, kind) for kind in ("zero", "tiny", "large", "cancel", "saturated")],
)
def test_numerical(n, kind):
    args = (
        *make_case(n, 6144, torch.bfloat16, kind=kind, offset=True)[:6],
        2.0,
        1e-6,
        1e-5,
    )
    baseline, golden = official_post_pre(*args), torch_post_pre(*args)
    actual = triton_ihc_post_pre(*args)
    RECORDS.append(
        dict(
            n=n,
            kind=kind,
            baseline_vs_golden=check_outputs(baseline, golden),
            fused_vs_golden=check_outputs(actual, golden),
            fused_vs_official=check_outputs(actual, baseline),
        )
    )
    check_outputs(fused_post_pre(*args), baseline)  # Preallocated benchmark variant.
    if n == 513:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = triton_ihc_post_pre(*args)
        args[0].neg_()
        graph.replay()
        check_outputs(captured, official_post_pre(*args))
        check_outputs(captured, torch_post_pre(*args))


def test_preallocated_baseline_matches_official_wrappers():
    args = make_case(7, 6144, torch.bfloat16, offset=True)
    expected = official_post_pre(*args)
    out = tuple(torch.empty_like(tensor) for tensor in expected)
    partial = torch.empty((7, 24, 9), device="cuda", dtype=torch.float32)
    actual = official_post_pre_into(*args, out=out, partial=partial)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_gpu_launches():
    args = make_case(513, 6144, torch.bfloat16)
    launches = {}
    for fn, count in ((triton_ihc_post_pre, 1), (official_post_pre, 3)):
        for _ in range(3):
            fn(*args)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn(*args)
            torch.cuda.synchronize()
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(names) == count, names
        launches[fn.__name__] = names
    assert "_fused_post_pre" in launches[triton_ihc_post_pre.__name__][0]
    if path := os.getenv("IHC_LAUNCH_JSON"):
        Path(path).write_text(json.dumps(launches))
