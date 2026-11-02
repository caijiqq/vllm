# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sparse MLA for HY4's 656-byte cache and small SM90 head groups.

Physical indices address cache rows, with -1 denoting padding. Unlike DSv4's
page-separated scales, each HY4 row contains 512 FP8 bytes, four FP32 scales,
then 64 BF16 RoPE values. V is the dequantized 512-dimensional latent.

The 16-head MMA tile and Split-KV approach follow vllm-project/vllm#59418;
the cache loader and query geometry here are specific to HY4.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _split_kv(
    Q,
    Cache,
    Indices,
    Partial,
    PartialLSE,
    Q_T: tl.constexpr,
    Q_H: tl.constexpr,
    C_PAGE: tl.constexpr,
    C_ROW: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    I_T: tl.constexpr,
    I_K: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    split = tl.program_id(1)
    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    q_base = Q + token * Q_T + h[:, None] * Q_H
    q = tl.load(q_base + d[None, :], h[:, None] < HEADS, other=0)
    q_rope = tl.load(q_base + 512 + r[None, :], h[:, None] < HEADS, other=0)
    m = tl.full((16,), float("-inf"), tl.float32)
    denom = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 512), tl.float32)
    blocks_per_split = tl.cdiv(tl.cdiv(TOPK, BLOCK_N), SPLITS)
    for block in range(
        split * blocks_per_split,
        tl.minimum((split + 1) * blocks_per_split, tl.cdiv(TOPK, BLOCK_N)),
    ):
        n = block * BLOCK_N + tl.arange(0, BLOCK_N)
        slot = tl.load(Indices + token * I_T + n * I_K, n < TOPK, other=-1)
        valid = (n < TOPK) & (slot >= 0)
        slot = tl.where(valid, slot, 0).to(tl.int64)
        row = Cache + (slot // PAGE_SIZE) * C_PAGE + (slot % PAGE_SIZE) * C_ROW
        raw = tl.load(row[:, None] + d[None, :], valid[:, None], other=0)
        scales = tl.load(
            (row + 512).to(tl.pointer_type(tl.float32))[:, None]
            + tl.arange(0, 4)[None, :],
            valid[:, None],
            other=1,
        )
        kv = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        kv = (kv.reshape(BLOCK_N, 4, 128) * scales[:, :, None]).reshape(BLOCK_N, 512)
        kv = kv.to(tl.bfloat16)
        rope = tl.load(
            (row + 528).to(tl.pointer_type(tl.bfloat16))[:, None] + r[None, :],
            valid[:, None],
            other=0,
        )
        scores = tl.dot(q, tl.trans(kv)) + tl.dot(q_rope, tl.trans(rope))
        scores *= SCALE * 1.4426950408889634
        scores = tl.where(valid[None, :], scores, float("-inf"))
        m_next = tl.maximum(m, tl.max(scores, 1))
        m_safe = tl.where(m_next == float("-inf"), 0.0, m_next)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(scores - m_safe[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m = m_next
    empty = denom == 0
    inv = 1.0 / tl.where(empty, 1.0, denom)
    acc *= inv[:, None]
    lse = tl.where(empty, float("-inf"), m + tl.log2(tl.where(empty, 1.0, denom)))
    base = (token * SPLITS + split) * 16 + h
    tl.store(Partial + base[:, None] * 512 + d[None, :], acc)
    tl.store(PartialLSE + base, lse)


@triton.jit
def _softmax_merge(
    Partial,
    PartialLSE,
    Sinks,
    Out,
    LSE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 512)
    base = (token * SPLITS + s) * 16 + head
    lse = tl.load(PartialLSE + base)
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    weights = tl.exp2(lse - m_safe)
    denom = tl.sum(weights, 0)
    empty = denom == 0
    lse2 = m_safe + tl.log2(tl.where(empty, 1.0, denom))
    value = tl.sum(
        tl.load(Partial + base[:, None] * 512 + d[None, :]) * weights[:, None], 0
    ) / tl.where(empty, 1.0, denom)
    if HAS_SINK:
        sink = tl.load(Sinks + head) * 1.4426950408889634
        value /= 1.0 + tl.exp2(sink - lse2)
    value = tl.where(empty, 0.0, value)
    tl.store(Out + (token * HEADS + head) * 512 + d, value)
    # Match FlashMLA: natural-log LSE excludes the sink.
    tl.store(
        LSE + token * HEADS + head,
        tl.where(empty, float("-inf"), lse2 * 0.6931471805599453),
    )


def smallhead_sparse_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    sinks: torch.Tensor | None = None,
    *,
    num_splits: int = 32,
    block_n: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 output [T,H,512] and FP32 sink-free LSE [T,H].

    Inputs are BF16 q [T,H,576], uint8 cache [pages,page_size,656],
    int32 physical indices [T,K], and optional FP32 sinks [H].
    The last dimension must be contiguous; outer strides are supported.
    """
    tokens, heads, dim = q.shape
    assert q.dtype == torch.bfloat16 and dim == 576 and heads in (8, 16)
    assert cache.dtype == torch.uint8 and cache.ndim == 3 and cache.shape[-1] == 656
    assert indices.dtype == torch.int32 and indices.shape[0] == tokens
    assert q.stride(-1) == cache.stride(-1) == 1
    assert q.is_cuda and q.device == cache.device == indices.device
    assert num_splits in (1, 2, 4, 8, 16, 32) and block_n in (32, 64)
    if sinks is not None:
        assert sinks.shape == (heads,) and sinks.dtype == torch.float32
        assert sinks.device == q.device and sinks.is_contiguous()
    partial = torch.empty(
        (tokens, num_splits, 16, 512), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (tokens, num_splits, 16), device=q.device, dtype=torch.float32
    )
    out = torch.empty((tokens, heads, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    _split_kv[(tokens, num_splits)](
        q,
        cache,
        indices,
        partial,
        partial_lse,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.shape[1],
        indices.stride(0),
        indices.stride(1),
        heads,
        indices.shape[1],
        softmax_scale,
        num_splits,
        block_n,
        num_warps=4,
    )
    _softmax_merge[(tokens, heads)](
        partial,
        partial_lse,
        sinks,
        out,
        lse,
        heads,
        num_splits,
        sinks is not None,
        num_warps=4,
    )
    return out, lse
