# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 small-head correctness/benchmark demo; run from the repository root."""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.models.hy_v4.nvidia.smallhead_sparse_decode import smallhead_sparse_decode
from vllm.triton_utils import triton
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata


def make_inputs(tokens=4, heads=8, topk=2048, page_size=64, rows=4096):
    """Use the production cache writer, including its per-128 FP32 scales."""
    kv = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    kv *= torch.tensor([0.125, 1.0, 4.0, 16.0], device="cuda").repeat_interleave(128)
    rope = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(
        rows // page_size, page_size, 656, dtype=torch.uint8, device="cuda"
    )
    ops.concat_and_cache_mla(
        kv,
        rope,
        cache,
        torch.arange(rows, device="cuda"),
        "fp8_ds_mla",
        torch.ones(1, device="cuda"),
    )
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16) * 0.25
    indices = torch.randint(rows, (tokens, topk), device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-2, 8, heads, device="cuda")
    return q, cache, indices, sinks


def reference(q, cache, indices, scale, sinks):
    rows = cache.reshape(-1, 656)
    latent = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    latent = (latent * scales.repeat_interleave(128, dim=-1)).bfloat16().float()
    rope = rows[:, 528:].contiguous().view(torch.bfloat16).float()
    keys = torch.cat((latent, rope), dim=-1)[indices.clamp_min(0).long()]
    values = latent[indices.clamp_min(0).long()]
    scores = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    scores.masked_fill_(indices[:, None, :] < 0, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    maximum = torch.where(torch.isneginf(lse), 0.0, lse)
    weights = torch.exp(scores - maximum[:, :, None])
    output = torch.einsum("thk,tkd->thd", weights, values)
    if sinks is not None:
        factor = torch.sigmoid(lse - sinks[None, :])
        factor = torch.where(torch.isneginf(lse), 0.0, factor)
        output *= factor[:, :, None]
    return output, lse


def flashmla_call(q, cache, indices, scale, sinks):
    """Return a callable with the same 64-head padding as HY4's backend."""
    tokens, heads, dim = q.shape
    indices = indices.contiguous()
    padded = q.new_zeros((tokens, 1, 64, dim))
    padded[:, 0, :heads] = q
    padded_sinks = sinks.new_full((64,), float("-inf"))
    padded_sinks[:heads] = sinks
    lens = torch.full((tokens,), 1, dtype=torch.int32, device=q.device)
    table = torch.zeros((tokens, 1), dtype=torch.int32, device=q.device)
    metadata, splits = get_mla_metadata(
        lens,
        64,
        1,
        num_heads_q=64,
        topk=indices.shape[-1],
        is_fp8_kvcache=True,
    )
    # The native SM90 kernel reads 64-row pages even for larger cache blocks.
    pages = cache.view(-1, 64, 1, 656)

    def run():
        out, lse = flash_mla_with_kvcache(
            padded,
            pages,
            table,
            lens,
            512,
            metadata,
            splits,
            softmax_scale=scale,
            is_fp8_kvcache=True,
            indices=indices[:, None, :],
            attn_sink=padded_sinks,
        )
        return out[:, 0, :heads], lse[:, :heads, 0]

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--heads", type=int, nargs="+", default=[8])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--splits", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--block-n", type=int, choices=[32, 64], default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--pool",
        type=int,
        default=64,
        help="Rotate caches (> H100/H200 L2 at default settings)",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(42)
    print(
        torch.cuda.get_device_name(), torch.__version__, triton.__version__, flush=True
    )
    results = []
    for heads in args.heads:
        for tokens in args.tokens:
            inputs = [
                make_inputs(tokens, heads, args.topk, rows=16384)
                for _ in range(args.pool)
            ]
            accessed_kv_bytes = sum(torch.unique(x[2]).numel() * 656 for x in inputs)
            scale = 256**-0.5  # HY4 pre-absorption QK dimension: 192 + 64.
            ref, ref_lse = reference(*inputs[0][:3], scale, inputs[0][3])
            native = [flashmla_call(*x[:3], scale, x[3]) for x in inputs]
            native_out, native_lse = native[0]()
            torch.testing.assert_close(native_out.float(), ref, rtol=0.02, atol=0.04)
            torch.testing.assert_close(native_lse, ref_lse, rtol=1e-4, atol=1e-4)
            for splits in args.splits:

                def candidate(inputs=inputs, scale=scale, splits=splits):
                    for q, cache, indices, sinks in inputs:
                        smallhead_sparse_decode(
                            q,
                            cache,
                            indices,
                            scale,
                            sinks,
                            num_splits=splits,
                            block_n=args.block_n,
                        )

                def baseline(native=native):
                    for run in native:
                        run()

                out, lse = smallhead_sparse_decode(
                    *inputs[0][:3],
                    scale,
                    inputs[0][3],
                    num_splits=splits,
                    block_n=args.block_n,
                )
                torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.04)
                torch.testing.assert_close(lse, ref_lse, rtol=1e-4, atol=1e-4)
                timings = {"triton": [], "flashmla": []}
                functions = {"triton": candidate, "flashmla": baseline}
                for round_id in range(args.rounds):
                    order = (
                        list(functions)
                        if round_id % 2 == 0
                        else list(reversed(functions))
                    )
                    for name in order:
                        ms = triton.testing.do_bench_cudagraph(functions[name], rep=100)
                        timings[name].append(ms * 1000 / args.pool)
                row = dict(
                    tokens=tokens,
                    heads=heads,
                    topk=args.topk,
                    splits=splits,
                    block_n=args.block_n,
                    triton_us=statistics.median(timings["triton"]),
                    flashmla_us=statistics.median(timings["flashmla"]),
                    max_abs_error=(out.float() - ref).abs().max().item(),
                    accessed_kv_bytes=accessed_kv_bytes,
                )
                row["speedup"] = row["flashmla_us"] / row["triton_us"]
                row["estimated_kv_gbps"] = (
                    tokens * args.topk * 656 / (row["triton_us"] * 1000)
                )
                results.append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                dict(
                    gpu=torch.cuda.get_device_name(),
                    pool=args.pool,
                    rounds=args.rounds,
                    torch=torch.__version__,
                    triton=triton.__version__,
                    cuda=torch.version.cuda,
                    timing="cuda_graph_events",
                    flashmla_padding_timed=False,
                    results=results,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
