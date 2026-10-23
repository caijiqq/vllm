"""Three-layer TP4/TP8 eager validation before formal timing."""

import argparse
import datetime
import functools
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None, *, benchmark=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("original", "intra", "cross"),
        default=["original", "intra", "cross"],
    )
    parser.add_argument("--warmup", type=int, default=6 if benchmark else 5)
    if benchmark:
        parser.add_argument("--tokens", type=int, default=8192)
        parser.add_argument("--rounds", type=int, default=9)
    else:
        parser.add_argument("--shapes", nargs="+", type=int, default=[512, 8192])
    options = parser.parse_args(argv)
    shapes = [options.tokens] if benchmark else options.shapes
    if options.warmup < 0 or any(n < 1 for n in shapes):
        parser.error("Shapes must be positive and warmup must be nonnegative")
    if benchmark and options.rounds < 1:
        parser.error("Rounds must be positive")
    if len(set(shapes)) != len(shapes):
        parser.error("Shapes must be unique")
    if options.modes[0] != "original" or len(set(options.modes)) != len(options.modes):
        parser.error("Modes must be unique and start with the original baseline")
    if (selected := os.getenv("HY4_TP8_GPUS")) and len(
        selected.split(",")
    ) != options.tp_size:
        parser.error("--tp-size must match the selected HY4_TP8_GPUS count")
    return options


def model_options(options):
    tokens = max(
        64, options.tokens if hasattr(options, "tokens") else max(options.shapes)
    )
    return dict(
        model=str(Path(os.environ["HY4_TP8_CHECKPOINT"]).resolve()),
        tensor_parallel_size=options.tp_size,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        distributed_executor_backend="mp",
        max_model_len=tokens + 256,
        max_num_batched_tokens=tokens,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        max_num_seqs=4,
        trust_remote_code=True,
        hf_overrides={
            "enable_ihc_post_pre_fusion": True,
            "enable_ihc_cross_layer_fusion": True,
        },
    )


def require_ranks(states, mode=None, *, tp_size=8):
    assert len(states) == tp_size and sorted(x["rank"] for x in states) == list(
        range(tp_size)
    ), states
    assert all(x["world_size"] == tp_size for x in states)
    if mode is None:
        return
    for x in states:
        assert x["flags"] == [
            dict(intra=mode != "original", cross=mode == "cross" and i < 2)
            for i in range(3)
        ], "This eager harness requires exactly three local iHC layers"


def main():
    options = parse_args()
    R = Path(os.environ["HY4_TP8_RUN"]) / "validation"
    RAW = R / "raw"
    RAW.mkdir(parents=True, exist_ok=True)
    require = functools.partial(require_ranks, tp_size=options.tp_size)

    def save(name, data):
        with (RAW / name).open("x") as f:
            json.dump(data, f, indent=2)

    def log(text):
        with (R / "EXPERIMENT_LOG.md").open("a") as f:
            f.write(
                "\n"
                + datetime.datetime.now(datetime.UTC).isoformat()
                + " "
                + text
                + "\n"
            )

    from lab.ihc_fusion.benchmarks.tp8_validation import workers
    from vllm import LLM, SamplingParams

    assert not any(RAW.iterdir()), "Use a fresh validation directory"
    log(
        f"E01 start: TP{options.tp_size}/PP1/mp/eager with existing checkpoint; "
        "both fusion flags initialized True; all-worker RPC will "
        "verify and switch modes. No global class or operator replacement."
    )
    kwargs = model_options(options)
    save("run_config.json", dict(options=vars(options), llm=kwargs))
    llm = LLM(**kwargs)
    states = llm.apply_model(workers.status)
    save("initial_workers.json", states)
    require(states, "cross")
    log(
        f"E01 loaded: {options.tp_size} distinct TP ranks, initialization flags agree. "
        "Partitions saved in initial_workers.json."
    )

    def generate(n):
        return (
            llm.generate(
                [{"prompt_token_ids": [100 + i % 197 for i in range(n)]}],
                SamplingParams(temperature=0, max_tokens=1, seed=7),
                use_tqdm=False,
            )[0]
            .outputs[0]
            .token_ids
        )

    smoke = []
    for mode in options.modes:
        states = llm.apply_model(functools.partial(workers.set_mode, name=mode))
        require(states, mode)
        smoke.append(dict(mode=mode, workers=states, tokens=generate(64)))
    save("smoke.json", smoke)
    smoke_tokens_match = all(r["tokens"] == smoke[0]["tokens"] for r in smoke)
    log(
        f"E02 short N64 inference completed; token_match={smoke_tokens_match}. "
        "This N falls back to original; it does not prove fusion execution."
    )
    assert sorted(llm.apply_model(workers.install_observers)) == list(
        range(options.tp_size)
    )
    for n in options.shapes:
        llm.apply_model(functools.partial(workers.set_mode, name="original"))
        for _ in range(options.warmup):
            generate(n)
        shape = []
        for mode in options.modes:
            states = llm.apply_model(functools.partial(workers.set_mode, name=mode))
            require(states, mode)
            llm.apply_model(workers.capture)
            tokens = generate(n)
            comparison = llm.apply_model(
                functools.partial(workers.captured, baseline=mode == "original")
            )
            require(comparison)
            shape.append(
                dict(mode=mode, n=n, tokens=tokens, workers=comparison)
            )
        save(f"correctness_{n}.json", shape)
        functional_pass = all(
            all(
                w["finite"]
                and w["shape_match"]
                and w["dtype_match"]
                and w["path"]["pass"]
                and w["boundaries_pass"]
                for w in row["workers"]
            )
            for row in shape
        )
        log(
            f"E03 N{n}: functional pass={functional_pass}; "
            "Original/Intra and Original/Cross hidden-state metrics are diagnostics. "
            "Kernel hit evidence remains unverified until profiler analysis."
        )
        if not functional_pass:
            save(
                "STOP.json",
                dict(
                    reason="Functional gate did not pass; no formal benchmark",
                    n=n,
                    checks=(
                        "all ranks, finite output, mode flags, temperature=0 tokens, "
                        "layer and pending path"
                    ),
                ),
            )
            print("STOP_FUNCTIONAL_GATE", n, flush=True)
            return 1
    save(
        "NUMERICAL_DIAGNOSTIC.json",
        {
            "status": "recorded_only",
            "comparisons": ["Original/Intra", "Original/Cross"],
            "runtime_boundaries": [
                "layer1_input_norm",
                "layer2_input_norm",
                "layer2_final_post",
            ],
            "strict_elementwise_check": "not_run",
            "numerical_accepted": "not_yet_accepted",
            "shapes": options.shapes,
            "kernel_hit": "unverified; inspect profiler traces with analyze_perf.py",
        },
    )
    save(
        "FUNCTIONAL_GATE.json",
        dict(
            passed=True,
            functional_pass=True,
            world_size=options.tp_size,
            checkpoint=str(Path(os.environ["HY4_TP8_CHECKPOINT"]).resolve()),
            shapes=options.shapes,
            modes=options.modes,
            expected_fallback_shapes=sorted(
                {64, *(n for n in options.shapes if not 512 <= n <= 8192)}
            ),
            token_diagnostic="recorded_only",
            numerical_diagnostic="recorded_only",
            numerical_accepted="not_yet_accepted",
            kernel_hit_verified="not_verified",
            performance_recorded=False,
            performance_result="not_run",
            kernel_evidence="unverified; inspect profiler traces with analyze_perf.py",
        ),
    )
    log(
        "E03 FUNCTIONAL PASS complete. Numerical diagnostics are recorded without "
        "an acceptance threshold; performance is a separate result."
    )
    print("FUNCTIONAL_GATE_COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
