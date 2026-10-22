# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the public HY4 D=6144 contract absent from the original suite."""

import pytest
import torch

from lab.ihc_fusion.baselines.official import official_post_pre
from lab.ihc_fusion.kernels.decode_two_stage import decode_two_stage
from lab.ihc_fusion.kernels.fused_post_pre import fused_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import DTYPES, check_outputs, make_case


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_public_config_shape_and_replay(dtype, n):
    torch.backends.cuda.matmul.allow_tf32 = False
    args = (*make_case(n, 6144, dtype, offset=True)[:6], 2.0, 1e-6, 1e-5)
    baseline, golden = official_post_pre(*args), torch_post_pre(*args)
    for fn in (fused_post_pre, decode_two_stage):
        actual = fn(*args)
        check_outputs(actual, baseline)
        check_outputs(actual, golden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*args, out=actual)
        args[0].neg_()
        graph.replay()
        check_outputs(actual, official_post_pre(*args))
        check_outputs(actual, torch_post_pre(*args))
        args[0].neg_()
