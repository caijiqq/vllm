"""N8192 paired TP4 timing; known model A/A noise is not an acceptance gate."""

import csv
import functools
import json
import os
import time
from pathlib import Path

import perf_workers
import torch
import triton
import workers
from validate import require_ranks

import vllm
from vllm import LLM, SamplingParams

R = Path(os.environ["HY4_TP4_RUN"])


def save(name, value):
    with (R / "raw" / name).open("x") as f:
        json.dump(value, f, indent=2)


def log(message):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(stamp, message, flush=True)
    with (R / "EXPERIMENT_LOG.md").open("a") as f:
        f.write(f"\n{stamp} {message}\n")


def main():
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
        "E01 initialize existing TP4 harness; "
        "strict model numerical acceptance remains open."
    )
    llm = LLM(
        model=os.environ["HY4_TP4_CHECKPOINT"],
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        enforce_eager=True,
        distributed_executor_backend="mp",
        max_model_len=8448,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        max_num_seqs=4,
        trust_remote_code=True,
        hf_overrides={
            "enable_ihc_post_pre_fusion": True,
            "enable_ihc_cross_layer_fusion": True,
        },
    )
    save("initial_workers.json", llm.apply_model(workers.status))
    save("parallelism.json", llm.apply_model(perf_workers.install))
    prompt = [{"prompt_token_ids": [100 + i % 197 for i in range(8192)]}]
    params = SamplingParams(temperature=0, max_tokens=1, seed=7)

    def generate():
        return llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].token_ids

    flags = []

    def mode(name):
        states = llm.apply_model(functools.partial(workers.set_mode, name=name))
        require_ranks(states, name)
        flags.append(
            {
                "mode": name,
                "workers": [{"rank": s["rank"], "flags": s["flags"]} for s in states],
            }
        )

    warm = []
    for name in ("original", "intra", "cross"):
        mode(name)
        for _ in range(6):
            generate()
        llm.apply_model(perf_workers.arm)
        tokens = generate()
        values = llm.apply_model(
            functools.partial(perf_workers.read, check_finite=True)
        )
        warm.append({"mode": name, "tokens": tokens, "workers": values})
        save(f"warm_{name}.json", warm[-1])
        assert all(x["finite"] for x in values), "Nonfinite model output"
        log(f"E02 warmed {name}; finite on all ranks; tokens={tokens}")

    rows = []
    with (R / "raw/timing.csv").open("x", newline="") as f:
        fields = [
            "round",
            "order",
            "mode",
            "rank0_ms",
            "rank1_ms",
            "rank2_ms",
            "rank3_ms",
            "forward_max_ms",
            "generate_wall_ms",
            "tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        orders = [
            ("original", "intra", "cross"),
            ("intra", "cross", "original"),
            ("cross", "original", "intra"),
        ]
        for round_id in range(9):
            for order, name in enumerate(orders[round_id % 3]):
                mode(name)
                generate()  # one untimed settling request after each mode switch
                llm.apply_model(perf_workers.arm)
                begin = time.perf_counter()
                tokens = generate()
                wall = (time.perf_counter() - begin) * 1000
                values = llm.apply_model(perf_workers.read)
                assert sorted(x["rank"] for x in values) == list(range(4))
                assert all(
                    len(x["records"]) == 1 and x["records"][0]["shape"] == [8192]
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
                row["forward_max_ms"] = max(row[f"rank{i}_ms"] for i in range(4))
                writer.writerow(row)
                f.flush()
                rows.append(row)
            log(f"E03 paired round {round_id + 1}/9 persisted")
    save("mode_checks.json", flags)
    log("E03 timing COMPLETE; profiler starts only after all 27 measurements saved.")
    for name in ("original", "intra", "cross"):
        mode(name)
        for _ in range(3):
            generate()
        llm.apply_model(functools.partial(perf_workers.start_profile, mode=name))
        tokens = generate()
        traces = llm.apply_model(perf_workers.stop_profile)
        save(f"profile_{name}.json", {"tokens": tokens, "workers": traces})
        log(f"E04 {name}: four rank activity traces saved")
    save("profile_mode_checks.json", flags[-3:])
    log(
        "COMPLETE: performance research only; model A/A and A/B acceptance not claimed."
    )


if __name__ == "__main__":
    main()
