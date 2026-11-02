# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton BF16 sparse decode adapted from minatoyukinaa/vllm add_decode_kernel.

Reference: c1252328ad, vllm/models/hy_v4/nvidia/sparse_mla_decode.py (TileLang).
Two pipeline stages let Triton generate the shared-memory copies and layouts.
"""

import torch

from vllm import envs
from vllm.triton_utils import tl, triton


@triton.jit
def _split_kv(
    Q,
    KV,
    Indices,
    Lengths,
    Partial,
    QT: tl.constexpr,
    QH: tl.constexpr,
    KR: tl.constexpr,
    IT: tl.constexpr,
    IK: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    token, split = tl.program_id(0), tl.program_id(1)
    BH: tl.constexpr = max(16, HEADS)
    BN: tl.constexpr = 64
    h = tl.arange(0, BH)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    q = tl.load(
        Q + token * QT + h[:, None] * QH + d[None, :], h[:, None] < HEADS, other=0
    )
    qr = tl.load(
        Q + token * QT + h[:, None] * QH + 512 + r[None, :], h[:, None] < HEADS, other=0
    )
    acc = tl.full((BH, 512), 0, tl.float32)
    m = tl.full((BH,), float("-inf"), tl.float32)
    denom = tl.full((BH,), 0, tl.float32)
    blocks = tl.cdiv(tl.cdiv(TOPK, BN), SPLITS)
    for block in range(
        split * blocks, tl.minimum((split + 1) * blocks, tl.cdiv(TOPK, BN))
    ):
        n = block * BN + tl.arange(0, BN)
        ids = tl.load(Indices + token * IT + n * IK, n < TOPK, other=-1)
        valid = (n < TOPK) & (ids >= 0) & (ids < ROWS)
        if HAS_LENGTHS:
            valid &= n < tl.load(Lengths + token)
        row = KV + tl.where(valid, ids, 0).to(tl.int64) * KR
        kv = tl.load(row[:, None] + d[None, :], valid[:, None], other=0)
        rope = tl.load(row[:, None] + 512 + r[None, :], valid[:, None], other=0)
        score = tl.dot(q, tl.trans(kv)) + tl.dot(qr, tl.trans(rope))
        score = tl.where(
            valid[None, :], score * (SCALE * 1.4426950408889634), float("-inf")
        )
        next_m = tl.maximum(m, tl.max(score, 1))
        safe_m = tl.where(next_m == float("-inf"), 0.0, next_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(score - safe_m[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m = next_m
    base = (token * SPLITS + split) * HEADS + h
    tl.store(Partial + base[:, None] * 514 + d[None, :], acc, h[:, None] < HEADS)
    tl.store(Partial + base * 514 + 512, m, h < HEADS)
    tl.store(Partial + base * 514 + 513, denom, h < HEADS)


@triton.jit
def _softmax_merge(
    Partial,
    Sinks,
    Out,
    LSE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    OUT_T: tl.constexpr = 0,
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
    stride: tl.constexpr = OUT_T or HEADS * 512
    tl.store(Out + t * stride + h * 512 + d, tl.where(denom > 0, value, 0.0))
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
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 Q [T,H,576], BF16 KV [rows,576], physical indices [T,K].

    Returns BF16 output [T,H,512] and sink-free FP32 LSE [T,H]. Invalid
    indices (including out-of-range slots) and entries beyond lengths are masked.
    An optional output buffer may have padding between token rows.
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
    if out is None:
        out = q.new_empty((tokens, heads, 512))
    else:
        assert out.shape == (tokens, heads, 512) and out.dtype == q.dtype
        assert out.device == q.device and out.stride(1) == 512
        assert out.stride(2) == 1 and out.stride(0) >= heads * 512
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
        num_warps=4,
        num_stages=2,
    )
    _softmax_merge[(tokens, heads)](
        partial,
        sinks,
        out,
        lse,
        heads,
        num_splits,
        sinks is not None,
        OUT_T=out.stride(0),
        num_warps=4,
    )
    return out, lse
