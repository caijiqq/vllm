# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo: native BF16 mixed attention vs Triton decode + native prefill."""

import argparse
import json
import os
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_hy4_smallhead_decode import make_inputs, reference
from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseImpl,
    HYV4FlashMLASparseMetadata,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decodes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--prefills", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--rep", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
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
            def run(enabled, inputs=inputs, impl=impl, nd=nd):
                for q, kv, indices, md in inputs:
                    out, _ = impl._forward_bf16_kv(
                        q,
                        kv,
                        indices,
                        replace(md, smallhead_decode_tokens=nd if enabled else 0),
                        8,
                    )
                return out

            # Warm compilation, then check both paths against FP32 in small chunks.
            for enabled in (False, True):
                out = run(enabled)
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
            samples = {False: [], True: []}
            for i in range(args.rounds):
                for enabled in (False, True) if i % 2 == 0 else (True, False):
                    ms = triton.testing.do_bench_cudagraph(
                        lambda enabled=enabled: run(enabled), rep=args.rep
                    )
                    samples[enabled].append(ms * 1000 / len(inputs))
            row = dict(
                decode_tokens=nd,
                prefill_tokens=np,
                pool=len(inputs),
                native_us=statistics.median(samples[False]),
                split_dispatch_us=statistics.median(samples[True]),
                native_samples_us=samples[False],
                split_samples_us=samples[True],
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        gpu=torch.cuda.get_device_name(),
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
