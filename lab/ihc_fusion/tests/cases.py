# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the frozen input and acceptance helpers from the production regression."""

import torch

from tests.models.test_hyv4_ihc_fusion import (  # noqa: F401
    check_outputs,
    make_case,
    metrics,
)

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
