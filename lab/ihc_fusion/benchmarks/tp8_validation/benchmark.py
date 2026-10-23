"""Paired TP4/TP8 eager timing, followed by a separate profiling phase."""

import csv
import functools
import json
import os
import time
from pathlib import Path

from lab.ihc_fusion.benchmarks.tp8_validation.validate import (
    model_options,
    parse_args,
    require_ranks,
)


def main():
    options = parse_args(benchmark=True)
    R = Path(os.environ["HY4_TP8_RUN"]) / "performance"
    require = functools.partial(require_ranks, tp_size=options.tp_size)

    def save(name, value):
        with (R / "raw" / name).open("x") as f:
            json.dump(value, f, indent=2)

    def log(message):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(stamp, message, flush=True)
        with (R / "EXPERIMENT_LOG.md").open("a") as f:
            f.write(f"\n{stamp} {message}\n")

    gate_dir = R.parent / "validation" / "raw"
    gate = json.loads((gate_dir / "FUNCTIONAL_GATE.json").read_text())
    assert gate.get("functional_pass", gate.get("passed")) and gate[
        "world_size"
    ] == options.tp_size, gate
    kwargs = model_options(options)
    assert gate["checkpoint"] == kwargs["model"], gate
    assert options.tokens in gate["shapes"], "Validate this shape before timing"
    assert set(options.modes) <= set(
        gate.get("modes", ["original", "intra", "cross"])
    ), gate
    (R / "raw").mkdir(parents=True, exist_ok=True)
    assert not any((R / "raw").iterdir()), "Use a fresh performance directory"
    save("run_config.json", dict(options=vars(options), llm=kwargs))

    import torch
    import triton

    import vllm
    from lab.ihc_fusion.benchmarks.tp8_validation import perf_workers, workers
    from vllm import LLM, SamplingParams

    save(
        "versions.json",
        dict(
            torch=torch.__version__,
            triton=triton.__version__,
            vllm=vllm.__version__,
            cuda=torch.version.cuda,
        ),
    )
    log(
        f"E01 initialize TP{options.tp_size} eager harness; "
        "functional gate passed; numerical diagnostics remain separate."
    )
    llm = LLM(**kwargs)
    initial = llm.apply_model(workers.status)
    require(initial, "cross")
    save("initial_workers.json", initial)
    parallel = llm.apply_model(perf_workers.install)
    require(parallel)
    save("parallelism.json", parallel)
    prompt = [{"prompt_token_ids": [100 + i % 197 for i in range(options.tokens)]}]
    params = SamplingParams(temperature=0, max_tokens=1, seed=7)

    def generate():
        return llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].token_ids

    flags = []

    def mode(name):
        states = llm.apply_model(functools.partial(workers.set_mode, name=name))
        require(states, name)
        flags.append(
            {
                "mode": name,
                "workers": [{"rank": s["rank"], "flags": s["flags"]} for s in states],
            }
        )

    for name in options.modes:
        mode(name)
        for _ in range(options.warmup):
            generate()
        llm.apply_model(perf_workers.arm)
        tokens = generate()
        values = llm.apply_model(
            functools.partial(perf_workers.read, check_finite=True)
        )
        require(values)
        save(f"warm_{name}.json", dict(mode=name, tokens=tokens, workers=values))
        assert all(x["finite"] for x in values), "Nonfinite model output"
        log(f"E02 warmed {name}; finite on all ranks; tokens={tokens}")

    with (R / "raw/timing.csv").open("x", newline="") as f:
        fields = [
            "round",
            "order",
            "mode",
            *[f"rank{i}_ms" for i in range(options.tp_size)],
            "forward_max_ms",
            "generate_wall_ms",
            "tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for round_id in range(options.rounds):
            offset = round_id % len(options.modes)
            order_modes = options.modes[offset:] + options.modes[:offset]
            for order, name in enumerate(order_modes):
                mode(name)
                generate()  # one untimed settling request after each mode switch
                llm.apply_model(perf_workers.arm)
                begin = time.perf_counter()
                tokens = generate()
                wall = (time.perf_counter() - begin) * 1000
                values = llm.apply_model(perf_workers.read)
                require(values)
                assert all(
                    len(x["records"]) == 1
                    and x["records"][0]["shape"] == [options.tokens]
                    for x in values
                ), values
                row = dict(
                    round=round_id,
                    order=order,
                    mode=name,
                    generate_wall_ms=wall,
                    tokens=str(tokens),
                )
                for value in values:
                    row[f"rank{value['rank']}_ms"] = value["records"][0]["ms"]
                row["forward_max_ms"] = max(
                    row[f"rank{i}_ms"] for i in range(options.tp_size)
                )
                writer.writerow(row)
                f.flush()
            log(f"E03 paired round {round_id + 1}/{options.rounds} persisted")
    save("mode_checks.json", flags)
    log("E03 timing COMPLETE; profiler starts only after all measurements saved.")
    for name in options.modes:
        mode(name)
        for _ in range(3):
            generate()
        llm.apply_model(functools.partial(perf_workers.start_profile, mode=name))
        tokens = generate()
        traces = llm.apply_model(perf_workers.stop_profile)
        require(traces)
        save(f"profile_{name}.json", {"tokens": tokens, "workers": traces})
        log(f"E04 {name}: {options.tp_size} rank activity traces saved")
    save("profile_mode_checks.json", flags[-len(options.modes) :])
    save(
        "PERFORMANCE_RESULT.json",
        {
            "status": "recorded",
            "functional_gate": "pass",
            "numerical_diagnostic": "not_an_acceptance_gate",
            "numerical_accepted": "not_yet_accepted",
            "kernel_hit_verified": "not_verified",
            "performance_recorded": True,
            "kernel_evidence": "run analyze_perf.py on profiler traces",
        },
    )
    log(
        "PERFORMANCE RESULT recorded; profiler trace analysis is required before "
        "claiming a fused kernel hit or performance conclusion."
    )


if __name__ == "__main__":
    main()
