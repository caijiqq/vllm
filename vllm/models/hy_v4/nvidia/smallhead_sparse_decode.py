# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton BF16 sparse decode adapted from minatoyukinaa/vllm add_decode_kernel.

Reference: c1252328ad, vllm/models/hy_v4/nvidia/sparse_mla_decode.py (TileLang).
Gluon supplies explicit shared-memory copies and MMA layouts in Triton.
"""

import torch

from vllm import envs
from vllm.triton_utils import gl, gluon, tl, triton

async_copy = gl.nvidia.ampere.async_copy
mma_v2 = gl.nvidia.ampere.mma_v2


@gluon.jit
def _copy_shared(dst, src, mask, ALIGNED: gl.constexpr):
    if ALIGNED:
        async_copy.async_copy_global_to_shared(dst, src, mask)
    else:
        dst.store(gl.load(src, mask, other=0))


@gluon.jit
def _split_kv(
    Q,
    KV,
    Indices,
    Lengths,
    Partial,
    QT: gl.constexpr,
    QH: gl.constexpr,
    KR: gl.constexpr,
    IT: gl.constexpr,
    IK: gl.constexpr,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    SCALE: gl.constexpr,
    SPLITS: gl.constexpr,
    HAS_LENGTHS: gl.constexpr,
    ALIGNED_Q: gl.constexpr,
    ALIGNED_KV: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    BH: gl.constexpr = max(16, HEADS)
    COPY: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]
    )
    A: gl.constexpr = gl.DotOperandLayout(0, MMA, 2)
    B: gl.constexpr = gl.DotOperandLayout(1, MMA, 2)
    SH: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    h = gl.arange(0, BH, layout=gl.SliceLayout(1, COPY))
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, COPY))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, COPY))
    qs = gl.allocate_shared_memory(gl.bfloat16, [BH, 512], SH)
    qr = gl.allocate_shared_memory(gl.bfloat16, [BH, 64], SH)
    ks = gl.allocate_shared_memory(gl.bfloat16, [64, 512], SH)
    kr = gl.allocate_shared_memory(gl.bfloat16, [64, 64], SH)
    qbase = Q + token * QT + h[:, None] * QH
    _copy_shared(qs, qbase + d[None, :], h[:, None] < HEADS, ALIGNED_Q)
    _copy_shared(qr, qbase + 512 + r[None, :], h[:, None] < HEADS, ALIGNED_Q)
    async_copy.commit_group()
    async_copy.wait_group(0)
    acc = gl.full([BH, 512], 0, gl.float32, MMA)
    m = gl.full([BH], float("-inf"), gl.float32, gl.SliceLayout(1, MMA))
    denom = gl.full([BH], 0, gl.float32, gl.SliceLayout(1, MMA))
    blocks = gl.cdiv(gl.cdiv(TOPK, 64), SPLITS)
    for block in range(
        split * blocks, gl.minimum((split + 1) * blocks, gl.cdiv(TOPK, 64))
    ):
        n = block * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, COPY))
        ids = gl.load(Indices + token * IT + n * IK, n < TOPK, other=-1)
        valid = (n < TOPK) & (ids >= 0) & (ids < ROWS)
        if HAS_LENGTHS:
            valid &= n < gl.load(Lengths + token)
        row = KV + gl.where(valid, ids, 0).to(gl.int64) * KR
        _copy_shared(ks, row[:, None] + d[None, :], valid[:, None], ALIGNED_KV)
        _copy_shared(kr, row[:, None] + 512 + r[None, :], valid[:, None], ALIGNED_KV)
        async_copy.commit_group()
        async_copy.wait_group(0)
        score = gl.full([BH, 64], 0, gl.float32, MMA)
        for chunk in gl.static_range(4):
            a = qs.slice(chunk * 128, 128, 1).load(A)
            b = ks.slice(chunk * 128, 128, 1).permute([1, 0]).load(B)
            score = mma_v2(a, b, score)
        score = mma_v2(qr.load(A), kr.permute([1, 0]).load(B), score)
        valid_mma = gl.convert_layout(valid, gl.SliceLayout(0, MMA))
        score = gl.where(
            valid_mma[None, :], score * (SCALE * 1.4426950408889634), float("-inf")
        )
        next_m = gl.maximum(m, gl.max(score, 1))
        safe_m = gl.where(next_m == float("-inf"), 0.0, next_m)
        alpha = gl.exp2(m - safe_m)
        p = gl.exp2(score - safe_m[:, None])
        denom = denom * alpha + gl.sum(p, 1)
        acc *= alpha[:, None]
        acc = mma_v2(gl.convert_layout(p.to(gl.bfloat16), A), ks.load(B), acc)
        m = next_m
    oh = gl.arange(0, BH, layout=gl.SliceLayout(1, MMA))
    od = gl.arange(0, 512, layout=gl.SliceLayout(0, MMA))
    base = (token * SPLITS + split) * HEADS + oh
    gl.store(Partial + base[:, None] * 514 + od[None, :], acc, oh[:, None] < HEADS)
    gl.store(Partial + base * 514 + 512, m, oh < HEADS)
    gl.store(Partial + base * 514 + 513, denom, oh < HEADS)


@triton.jit
def _softmax_merge(
    Partial,
    Sinks,
    Out,
    LSE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    t, h = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 512)
    base = ((t * SPLITS + s) * HEADS + h) * 514
    maxima = tl.load(Partial + base + 512)
    sums = tl.load(Partial + base + 513)
    m = tl.max(maxima, 0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    weights = tl.exp2(maxima - safe_m)
    denom = tl.sum(sums * weights, 0)
    lse2 = safe_m + tl.log2(tl.where(denom > 0, denom, 1.0))
    value = tl.sum(tl.load(Partial + base[:, None] + d[None, :]) * weights[:, None], 0)
    value /= tl.where(denom > 0, denom, 1.0)
    if HAS_SINK:
        value /= 1.0 + tl.exp2(tl.load(Sinks + h) * 1.4426950408889634 - lse2)
    tl.store(Out + (t * HEADS + h) * 512 + d, tl.where(denom > 0, value, 0.0))
    tl.store(
        LSE + t * HEADS + h,
        tl.where(denom > 0, lse2 * 0.6931471805599453, float("-inf")),
    )


def smallhead_sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    sinks: torch.Tensor | None = None,
    *,
    topk_length: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 Q [T,H,576], BF16 KV [rows,576], physical indices [T,K].

    Returns BF16 output [T,H,512] and sink-free FP32 LSE [T,H]. Invalid
    indices (including out-of-range slots) and entries beyond lengths are masked.
    """
    tokens, heads, dim = q.shape
    assert q.dtype == kv.dtype == torch.bfloat16
    assert dim == kv.shape[-1] == 576 and heads in (8, 16, 32)
    assert kv.ndim == indices.ndim == 2 and indices.shape[0] == tokens
    assert q.stride(-1) == kv.stride(-1) == 1
    assert q.is_cuda and q.device == kv.device == indices.device
    assert indices.dtype == torch.int32 and indices.shape[1] > 0
    if sinks is not None:
        assert sinks.dtype == torch.float32 and sinks.shape == (heads,)
        assert sinks.device == q.device and sinks.is_contiguous()
    if topk_length is not None:
        assert topk_length.dtype == torch.int32 and topk_length.shape == (tokens,)
        assert topk_length.device == q.device and topk_length.is_contiguous()
    out = q.new_empty((tokens, heads, 512))
    lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    if tokens == 0:
        return out, lse
    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = (
            1
            if envs.VLLM_BATCH_INVARIANT
            else min(32, triton.cdiv(sms, tokens), triton.cdiv(indices.shape[1], 64))
        )
    assert num_splits > 0
    num_splits = triton.next_power_of_2(num_splits)
    partial = torch.empty(
        (tokens, num_splits, heads, 514), device=q.device, dtype=torch.float32
    )
    _split_kv[(tokens, num_splits)](
        q,
        kv,
        indices,
        topk_length,
        partial,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        indices.stride(1),
        kv.shape[0],
        heads,
        indices.shape[1],
        softmax_scale,
        num_splits,
        topk_length is not None,
        q.storage_offset() % 8 == q.stride(0) % 8 == q.stride(1) % 8 == 0,
        kv.storage_offset() % 8 == kv.stride(0) % 8 == 0,
        num_warps=4,
        num_stages=1,
    )
    _softmax_merge[(tokens, heads)](
        partial, sinks, out, lse, heads, num_splits, sinks is not None, num_warps=4
    )
    return out, lse
