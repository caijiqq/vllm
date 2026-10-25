# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile HY4 decode GPU kernels on the three-layer checkpoint."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from lab.ihc_fusion.benchmarks.tp8_validation.validate import model_options
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    assert args.context_len > 0
    assert args.decode_steps >= 2
    assert args.batch_size > 0
    assert args.warmup >= 0
    assert len(os.environ["HY4_TP8_GPUS"].split(",")) == args.tp_size

    trace_dir = Path(os.environ["HY4_TP8_RUN"]) / "decode_profile" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    max_len = args.context_len + args.decode_steps + 256
    token_budget = max(max_len, args.context_len * args.batch_size)

    options = model_options(SimpleNamespace(tp_size=args.tp_size, tokens=token_budget))
    options.update(
        max_model_len=max_len,
        max_num_batched_tokens=token_budget,
        max_num_seqs=max(4, args.batch_size),
        hf_overrides={
            "enable_ihc_post_pre_fusion": False,
            "enable_ihc_cross_layer_fusion": False,
        },
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(trace_dir),
            torch_profiler_with_stack=False,
            ignore_frontend=True,
            delay_iterations=1,
            max_iterations=min(16, args.decode_steps - 1),
        ),
    )

    llm = LLM(**options)

    prompts = [
        {
            "prompt_token_ids": [
                100 + (i + j * 37) % 197 for i in range(args.context_len)
            ]
        }
        for j in range(args.batch_size)
    ]

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.decode_steps,
        ignore_eos=True,
    )

    for _ in range(args.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)

    llm.start_profile()
    try:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
    finally:
        llm.stop_profile()

    lengths = [len(o.outputs[0].token_ids) for o in outputs]
    assert all(n == args.decode_steps for n in lengths), lengths

    print("DECODE_PROFILE_COMPLETE")
    print("Generated tokens:", lengths)
    print("Trace directory:", trace_dir)


if __name__ == "__main__":
    main()
