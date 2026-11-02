# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 HY4 sparse decode demo: Triton vs native FlashMLA, same cache."""

import argparse
import json
import statistics

import torch

from vllm.models.hy_v4.nvidia.smallhead_sparse_decode import smallhead_sparse_decode
from vllm.triton_utils import triton
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd


def make_inputs(tokens=4, heads=8, topk=2048, rows=4096, independent_requests=False):
    rows_per_request = rows
    if independent_requests:
        rows *= tokens
    kv = torch.randn(rows, 576, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        rows_per_request, (tokens, topk), device="cuda", dtype=torch.int32
    )
    if independent_requests:
        indices += (
            torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
            * rows_per_request
        )
    sinks = torch.linspace(-2, 8, heads, device="cuda")
    return q, kv, indices, sinks


def reference(q, kv, indices, scale, sinks, lengths=None):
    valid = (indices >= 0) & (indices < kv.shape[0])
    if lengths is not None:
        valid &= (
            torch.arange(indices.shape[1], device=q.device)[None, :] < lengths[:, None]
        )
    keys = kv[indices.clamp(0, kv.shape[0] - 1).long()].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1)
    safe_lse = torch.where(torch.isneginf(lse), 0.0, lse)
    weights = torch.exp(scores - safe_lse[:, :, None])
    out = torch.einsum("thk,tkd->thd", weights, keys[..., :512])
    if sinks is not None:
        factor = torch.where(
            torch.isneginf(lse), 0.0, torch.sigmoid(lse - sinks[None, :])
        )
        out *= factor[:, :, None]
    return out, lse


def flashmla_call(q, kv, indices, scale, sinks, lengths=None):
    # Installed 0.32.0 FlashMLA requires 64 heads, exactly as the HY4 baseline.
    t, h, d = q.shape
    padded = q.new_zeros((t, 64, d))
    padded[:, :h] = q
    padded_sinks = None
    if sinks is not None:
        padded_sinks = sinks.new_full((64,), -float("inf"))
        padded_sinks[:h] = sinks
    indices = indices.contiguous().view(t, 1, -1)

    def run():
        out, _, lse = flash_mla_sparse_fwd(
            padded,
            kv.view(-1, 1, 576),
            indices,
            scale,
            512,
            attn_sink=padded_sinks,
            topk_length=lengths,
        )
        return out[:, :h], lse[:, :h]

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[8])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0],
        help="0 selects the reference split heuristic",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--pool",
        type=int,
        default=4,
        help="Rotating input sets; use 64 for small token counts",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--independent-requests",
        action="store_true",
        help="Use a separate 16K KV region per decode token",
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    print(
        torch.cuda.get_device_name(), torch.__version__, triton.__version__, flush=True
    )
    results = []
    for heads in args.heads:
        for tokens in args.tokens:
            inputs = [
                make_inputs(
                    tokens,
                    heads,
                    args.topk,
                    rows=16384,
                    independent_requests=args.independent_requests,
                )
                for _ in range(args.pool)
            ]
            accessed_kv_bytes = sum(torch.unique(x[2]).numel() * 1152 for x in inputs)
            scale = 256**-0.5  # HY4 pre-absorption QK dimension: 192 + 64.
            ref, ref_lse = reference(*inputs[0][:3], scale, inputs[0][3])
            native = [flashmla_call(*x[:3], scale, x[3]) for x in inputs]
            native_out, native_lse = native[0]()
            torch.testing.assert_close(native_out.float(), ref, rtol=0.02, atol=0.01)
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
                            num_splits=splits or None,
                        )

                def baseline(native=native):
                    for run in native:
                        run()

                out, lse = smallhead_sparse_decode(
                    *inputs[0][:3],
                    scale,
                    inputs[0][3],
                    num_splits=splits or None,
                )
                torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.01)
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
                    triton_us=statistics.median(timings["triton"]),
                    flashmla_us=statistics.median(timings["flashmla"]),
                    max_abs_error=(out.float() - ref).abs().max().item(),
                    accessed_kv_bytes=accessed_kv_bytes,
                )
                row["speedup"] = row["flashmla_us"] / row["triton_us"]
                row["estimated_kv_gbps"] = (
                    tokens * args.topk * 1152 / (row["triton_us"] * 1000)
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
                    independent_requests=args.independent_requests,
                    results=results,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
