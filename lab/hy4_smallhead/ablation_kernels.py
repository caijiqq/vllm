# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen BF16 decode ablations; no serving/backend dependencies.

Gluon defaults reproduce daa54b83bb61f49954d8823a0a3197f448b6e480.
Plain Triton uses the same raw FP32 [T,S,H,514] partial contract
and production softmax merge.
"""

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
def gluon_split(
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
    BN: gl.constexpr = 64,
    WM: gl.constexpr = 1,
    WN: gl.constexpr = 4,
    CM: gl.constexpr = 4,
    CN: gl.constexpr = 1,
    PER_PHASE: gl.constexpr = 1,
    MAX_PHASE: gl.constexpr = 8,
    CHUNK: gl.constexpr = 128,
    ASYNC_KV: gl.constexpr = True,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    BH: gl.constexpr = max(16, HEADS)
    COPY: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [CM, CN], [1, 0])
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8]
    )
    A: gl.constexpr = gl.DotOperandLayout(0, MMA, 2)
    B: gl.constexpr = gl.DotOperandLayout(1, MMA, 2)
    SH: gl.constexpr = gl.SwizzledSharedLayout(8, PER_PHASE, MAX_PHASE, [1, 0])
    h = gl.arange(0, BH, layout=gl.SliceLayout(1, COPY))
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, COPY))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, COPY))
    qs = gl.allocate_shared_memory(gl.bfloat16, [BH, 512], SH)
    qr = gl.allocate_shared_memory(gl.bfloat16, [BH, 64], SH)
    ks = gl.allocate_shared_memory(gl.bfloat16, [BN, 512], SH)
    kr = gl.allocate_shared_memory(gl.bfloat16, [BN, 64], SH)
    qbase = Q + token * QT + h[:, None] * QH
    _copy_shared(qs, qbase + d[None, :], h[:, None] < HEADS, ALIGNED_Q)
    _copy_shared(qr, qbase + 512 + r[None, :], h[:, None] < HEADS, ALIGNED_Q)
    async_copy.commit_group()
    async_copy.wait_group(0)
    acc = gl.full([BH, 512], 0, gl.float32, MMA)
    m = gl.full([BH], float("-inf"), gl.float32, gl.SliceLayout(1, MMA))
    denom = gl.full([BH], 0, gl.float32, gl.SliceLayout(1, MMA))
    blocks = gl.cdiv(gl.cdiv(TOPK, BN), SPLITS)
    for block in range(
        split * blocks, gl.minimum((split + 1) * blocks, gl.cdiv(TOPK, BN))
    ):
        n = block * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, COPY))
        ids = gl.load(Indices + token * IT + n * IK, n < TOPK, other=-1)
        valid = (n < TOPK) & (ids >= 0) & (ids < ROWS)
        if HAS_LENGTHS:
            valid &= n < gl.load(Lengths + token)
        row = KV + gl.where(valid, ids, 0).to(gl.int64) * KR
        _copy_shared(
            ks, row[:, None] + d[None, :], valid[:, None], ALIGNED_KV and ASYNC_KV
        )
        _copy_shared(
            kr, row[:, None] + 512 + r[None, :], valid[:, None], ALIGNED_KV and ASYNC_KV
        )
        async_copy.commit_group()
        async_copy.wait_group(0)
        score = gl.full([BH, BN], 0, gl.float32, MMA)
        for chunk in gl.static_range(512 // CHUNK):
            a = qs.slice(chunk * CHUNK, CHUNK, 1).load(A)
            b = ks.slice(chunk * CHUNK, CHUNK, 1).permute([1, 0]).load(B)
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
def triton_split(
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
    BN: tl.constexpr = 64,
):
    token, split = tl.program_id(0), tl.program_id(1)
    BH: tl.constexpr = max(16, HEADS)
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
