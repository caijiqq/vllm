# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo: joint native (A), split native (B), Triton + native (C) attention."""

import argparse
import hashlib
import itertools
import json
import os
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.kernels.benchmark_hy4_smallhead_decode import make_inputs, reference
from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseImpl,
    HYV4FlashMLASparseMetadata,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)


def split_native(impl, q, kv, indices, md, nd):
    """B: same conversion/output as C, with only the decode implementation changed."""
    rows, block_stride = flat_kv_row_view(kv, md.block_size)
    physical, lengths = triton_convert_req_index_to_global_index(
        md.req_id_per_token[: q.shape[0]],
        md.block_table,
        indices,
        BLOCK_SIZE=md.block_size,
        BLOCK_STRIDE_ROWS=block_stride,
        NUM_TOPK_TOKENS=indices.shape[1],
        return_valid_counts=True,
    )
    output = q.new_empty((q.shape[0], 64, 512))
    for sl in (slice(0, nd), slice(nd, q.shape[0])):
        impl._bf16_flash_mla_kernel(
            q[sl], rows, physical[sl], lengths[sl], 8, out=output[sl]
        )
    return output[:, :8]


def audit_execution(run, impl, mode, nd, np, trace_dir):
    """Log real attention slices at capture, then trace one four-input replay."""
    from vllm.models.hy_v4.nvidia import smallhead_sparse_decode as decode_module

    calls = []

    def record(kind, fn):
        def wrapped(q, *args, **kwargs):
            out = kwargs.get("out")
            calls.append(
                dict(
                    kind=kind,
                    q_shape=list(q.shape),
                    q_token_offset=q.storage_offset() // q.stride(0),
                    q_storage=q.untyped_storage().data_ptr(),
                    out_shape=list(out.shape) if out is not None else None,
                    out_token_offset=(
                        out.storage_offset() // out.stride(0)
                        if out is not None
                        else None
                    ),
                    out_storage=(
                        out.untyped_storage().data_ptr() if out is not None else None
                    ),
                )
            )
            return fn(q, *args, **kwargs)

        return wrapped

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        patch.object(
            impl,
            "_bf16_flash_mla_kernel",
            record("native", impl._bf16_flash_mla_kernel),
        ),
        patch.object(
            decode_module,
            "smallhead_sparse_decode",
            record("triton", decode_module.smallhead_sparse_decode),
        ),
        torch.cuda.graph(graph),
    ):
        captured = run(mode)
    expected = {
        "A": [("native", nd + np, 0)],
        "B": [("native", nd, 0), ("native", np, nd)],
        "C": [("triton", nd, 0), ("native", np, nd)],
    }[mode]
    assert [
        (c["kind"], c["q_shape"][0], c["q_token_offset"]) for c in calls
    ] == expected * 4
    if mode != "A":
        for decode, prefill in zip(calls[::2], calls[1::2]):
            assert decode["q_storage"] == prefill["q_storage"]
            assert decode["out_storage"] == prefill["out_storage"]
            assert decode["out_token_offset"] == 0
            assert prefill["out_token_offset"] == nd
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        graph.replay()
        torch.accelerator.synchronize()
    path = Path(trace_dir) / f"{nd}-{np}-{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
    )
    attention = [
        e
        for e in kernels
        if any(
            n in e["name"]
            for n in ("sparse_attn_fwd_kernel", "_split_kv", "_softmax_merge")
        )
    ]
    assert len(attention) == {"A": 4, "B": 8, "C": 12}[mode]
    assert len({e["args"]["stream"] for e in kernels}) == 1
    return dict(
        calls=calls,
        kernel_counts=dict(Counter(e["name"] for e in kernels)),
        launches=[
            dict(
                name=e["name"],
                grid=e["args"].get("grid"),
                stream=e["args"].get("stream"),
            )
            for e in kernels
        ],
        copies=[
            dict(
                name=e["name"],
                bytes=e["args"].get("bytes"),
                stream=e["args"].get("stream"),
            )
            for e in events
            if e.get("cat") == "gpu_memcpy"
        ],
        trace=str(path),
        output_shape=list(captured.shape),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decodes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--prefills", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--rep", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--audit-dir", help="Trace one replay per variant instead of timing"
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    orders = list(itertools.permutations(("A", "B", "C")))
    for nd in args.decodes:
        for np in args.prefills:
            t = nd + np
            inputs = []
            impl = object.__new__(HYV4FlashMLASparseImpl)
            impl.dcp_world_size = impl.pcp_world_size = 1
            impl.index_group = None
            impl.softmax_scale = 0.0625
            impl.prefill_padding = 64
            for _ in range(4):
                q, kv, indices, sinks = make_inputs(t, rows=(nd + 1) * 16384)
                # Independent 16K context per decode request; prefill shares one.
                indices %= 16384
                req = torch.arange(t, device="cuda", dtype=torch.int32).clamp_max(nd)
                # Causal prompt extension, with an existing 8K prefix.
                indices[nd:] %= torch.arange(8193, 8193 + np, device="cuda")[:, None]
                table = torch.randperm((nd + 1) * 128, device="cuda").int()
                table = table.view(nd + 1, 128)
                md = HYV4FlashMLASparseMetadata(
                    num_reqs=nd + 1,
                    num_actual_tokens=t,
                    max_query_len=np,
                    max_seq_len=16384,
                    query_start_loc=torch.tensor(
                        list(range(nd + 1)) + [t], device="cuda", dtype=torch.int32
                    ),
                    slot_mapping=torch.zeros(t, device="cuda", dtype=torch.int64),
                    block_table=table,
                    req_id_per_token=req,
                    num_decode_tokens=nd,
                    num_decodes=nd,
                    num_prefills=1,
                    block_size=128,
                )
                padded = q.new_zeros((t, 64, 576))
                padded[:, :8] = q
                inputs.append((padded, kv.view(-1, 128, 576), indices, md))
                impl.sinks = sinks

            # All inputs use the same deterministic sink values.
            def run(mode, inputs=inputs, impl=impl, nd=nd):
                for q, kv, indices, md in inputs:
                    if mode == "B":
                        out = split_native(impl, q, kv, indices, md, nd)
                    else:
                        out, _ = impl._forward_bf16_kv(
                            q,
                            kv,
                            indices,
                            replace(
                                md, smallhead_decode_tokens=nd if mode == "C" else 0
                            ),
                            8,
                        )
                return out

            # Warm compilation, then check all three paths against FP32 in small chunks.
            for mode in ("A", "B", "C"):
                out = run(mode)
                q, kv, ids, md = inputs[-1]
                physical = (
                    md.block_table[md.req_id_per_token.long()].gather(
                        1, (ids // 128).long()
                    )
                    * 128
                    + ids % 128
                )
                for start in range(0, t, 32):
                    sl = slice(start, start + 32)
                    expected, _ = reference(
                        q[sl, :8], kv.view(-1, 576), physical[sl], 0.0625, impl.sinks
                    )
                    torch.testing.assert_close(
                        out[sl].float(), expected, rtol=0.02, atol=0.01
                    )
            if args.audit_dir:
                audits = {
                    mode: audit_execution(run, impl, mode, nd, np, args.audit_dir)
                    for mode in ("A", "B", "C")
                }
                rows.append(
                    dict(decode_tokens=nd, prefill_tokens=np, pool=4, audits=audits)
                )
                Path(args.output).write_text(
                    json.dumps(
                        dict(
                            gpu=torch.cuda.get_device_name(),
                            cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                            torch=torch.__version__,
                            triton=triton.__version__,
                            benchmark_sha256=hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            results=rows,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(f"Execution audit passed: {nd}+{np}", flush=True)
                continue
            samples = {mode: [] for mode in ("A", "B", "C")}
            for i in range(args.rounds):
                for mode in orders[i % len(orders)]:
                    ms = triton.testing.do_bench_cudagraph(
                        lambda mode=mode: run(mode), rep=args.rep
                    )
                    samples[mode].append(ms * 1000 / len(inputs))
            row = dict(
                decode_tokens=nd,
                prefill_tokens=np,
                pool=len(inputs),
                native_us=statistics.median(samples["A"]),
                split_dispatch_us=statistics.median(samples["C"]),
                split_native_us=statistics.median(samples["B"]),
                split_native_samples_us=samples["B"],
                native_samples_us=samples["A"],
                split_samples_us=samples["C"],
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        gpu=torch.cuda.get_device_name(),
                        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                        cuda=torch.version.cuda,
                        seed=42,
                        variant_order=[
                            orders[i % len(orders)] for i in range(args.rounds)
                        ],
                        benchmark_sha256=hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                        torch=torch.__version__,
                        triton=triton.__version__,
                        heads=8,
                        topk=2048,
                        kv_dtype="bfloat16",
                        rounds=args.rounds,
                        rep_ms=args.rep,
                        timing="cuda_graph_events_backend_with_index_conversion",
                        results=rows,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
