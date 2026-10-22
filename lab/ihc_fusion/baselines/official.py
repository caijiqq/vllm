# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute unchanged official Triton definitions without runtime dependencies.

The source AST is used only to omit runtime imports and the dispatch predicate.
Every selected function, decorator and constant is compiled without rewriting.
This avoids importing vLLM's unavailable compiled extensions in a standalone Lab.
"""

import ast
import hashlib
import subprocess
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl

SOURCE_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
SOURCE_SHA256 = "9765c65453dd2d1b9828e82790ece36a304c7c8f6f54229b146cbfdbd5e9094a"
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm/models/hy_v4/nvidia/triton_ihc.py"


def _load_official():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Official iHC source differs from frozen baseline")
    recorded = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE.relative_to(ROOT)}"], cwd=ROOT
    )
    if source != recorded:
        raise RuntimeError("Official source does not match pinned commit")
    tree = ast.parse(source, filename=str(SOURCE))
    names = {
        "_ihc_pre_stage1",
        "_ihc_pre_stage2",
        "_ihc_post_kernel",
        "triton_ihc_pre",
        "triton_ihc_post",
    }
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id in {"_BLOCK_K", "_BLOCK_D"}
                for target in node.targets
            )
        )
    ]
    if {n.name for n in selected if isinstance(n, ast.FunctionDef)} != names:
        raise RuntimeError("Incomplete official definitions")
    module = ModuleType("ihc_lab_pinned_official")
    module.__file__ = str(SOURCE)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        module.__dict__,
    )
    return module


OFFICIAL = _load_official()


def provenance():
    """Return the exact source and callable origin used by this baseline."""
    return {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_path": str(SOURCE),
        "pre_code_path": OFFICIAL.triton_ihc_pre.__code__.co_filename,
        "post_code_path": OFFICIAL.triton_ihc_post.__code__.co_filename,
        "loader": "unmodified source AST definitions; runtime imports omitted",
        "fallback": False,
        "launches": 3,
    }


def official_post_pre(
    x, residual, previous_post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    """Call the pinned official Post and Pre, with no model dispatch."""
    updated = OFFICIAL.triton_ihc_post(x, residual, previous_post)
    output, post = OFFICIAL.triton_ihc_pre(
        updated, weight, scale, base, magnitude, hc_eps, norm_eps
    )
    return output, updated, post


def official_post_pre_into(
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
    out,
    partial,
):
    """Launch the same official kernels into preallocated benchmark buffers.

    Grid, constants and launch options match the pinned official wrappers.
    Allocation is excluded from both benchmark arms; this still launches three
    original kernels. Correctness tests compare this path to the wrappers.
    """
    output, updated, post = out
    tokens, channels, hidden = residual.shape
    if tokens == 0:
        return out
    k = channels * hidden
    cp = triton.next_power_of_2(channels)
    splits = triton.cdiv(k, OFFICIAL._BLOCK_K)
    OFFICIAL._ihc_post_kernel[(tokens, triton.cdiv(hidden, OFFICIAL._BLOCK_D))](
        x,
        residual,
        previous_post,
        updated,
        HIDDEN_SIZE=hidden,
        HC_MULT=channels,
        HC_POW2=cp,
        BLOCK_D=OFFICIAL._BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    OFFICIAL._ihc_pre_stage1[(tokens, splits)](
        updated,
        weight,
        partial,
        K_TOTAL=k,
        HC_MULT=channels,
        HC_POW2=cp,
        NUM_SPLITS=splits,
        BLOCK_K=OFFICIAL._BLOCK_K,
        PARTIAL_STRIDE=1 + 2 * cp,
        num_warps=8,
        enable_fp_fusion=False,
    )
    OFFICIAL._ihc_pre_stage2[(tokens, triton.cdiv(hidden, OFFICIAL._BLOCK_D))](
        updated,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=hidden,
        K_TOTAL=k,
        HC_MULT=channels,
        HC_POW2=cp,
        NUM_SPLITS=splits,
        PARTIAL_STRIDE=1 + 2 * cp,
        BLOCK_D=OFFICIAL._BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
