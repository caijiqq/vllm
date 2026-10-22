"""TP4 startup and numerical gate before any formal timing."""

import datetime
import functools
import json
import os
from pathlib import Path

import workers

from vllm import LLM, SamplingParams

R = Path(os.environ["HY4_TP4_RUN"])
RAW = R / "raw"


def save(name, data):
    with (RAW / name).open("x") as f:
        json.dump(data, f, indent=2)


def log(text):
    with (R / "EXPERIMENT_LOG.md").open("a") as f:
        f.write(
            "\n" + datetime.datetime.now(datetime.UTC).isoformat() + " " + text + "\n"
        )


def require_ranks(states, mode):
    assert len(states) == 4 and sorted(x["rank"] for x in states) == list(range(4)), (
        states
    )
    assert all(x["world_size"] == 4 for x in states)
    for x in states:
        assert x["flags"] == [
            dict(intra=mode != "original", cross=mode == "cross" and i < 2)
            for i in range(3)
        ]


def main():
    log(
        "E01 start: standard LLM initialization TP4/PP1/mp/eager with existing "
        "checkpoint, both fusion flags initialized True; all-worker RPC will "
        "verify and switch modes. No global class or operator replacement."
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
    states = llm.apply_model(workers.status)
    save("initial_workers.json", states)
    require_ranks(states, "cross")
    log(
        "E01 loaded: four distinct TP ranks returned, initialization flags agree. "
        "Runtime partitions saved in initial_workers.json."
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
    for mode in ("original", "intra", "cross"):
        states = llm.apply_model(functools.partial(workers.set_mode, name=mode))
        require_ranks(states, mode)
        smoke.append(dict(mode=mode, workers=states, tokens=generate(64)))
    save("smoke.json", smoke)
    assert all(r["tokens"] == smoke[0]["tokens"] for r in smoke), (
        "Short-prompt mode token mismatch"
    )
    log(
        "E02 short N64 inference completed in three modes; tokens match. "
        "This N falls back to original; it does not prove fusion execution."
    )
    assert sorted(llm.apply_model(workers.install_observers)) == list(range(4))
    results = []
    for n in (512, 8192):
        llm.apply_model(functools.partial(workers.set_mode, name="original"))
        for _ in range(5):
            generate(n)
        shape = []
        for label, mode in [
            ("a", "original"),
            ("aa", "original"),
            ("intra", "intra"),
            ("cross", "cross"),
        ]:
            states = llm.apply_model(functools.partial(workers.set_mode, name=mode))
            require_ranks(states, mode)
            llm.apply_model(workers.capture)
            tokens = generate(n)
            comparison = llm.apply_model(
                functools.partial(workers.captured, baseline=label == "a")
            )
            shape.append(
                dict(label=label, mode=mode, n=n, tokens=tokens, workers=comparison)
            )
        save(f"correctness_{n}.json", shape)
        results.extend(shape)
        passed = all(
            all(
                w["finite"] and w.get("existing_chain_rule_pass", True)
                for w in row["workers"]
            )
            and row["tokens"] == shape[0]["tokens"]
            for row in shape
        )
        log(
            f"E03 N{n}: all-rank final hidden comparison and A/A recorded; "
            f"existing-chain diagnostic rule pass={passed}. "
            "Strict full-model acceptance is not inferred."
        )
        if not passed:
            save(
                "STOP.json",
                dict(
                    reason="TP4 numerical gate did not pass; no formal benchmark",
                    n=n,
                    rule="existing chain rtol=2*BF16 eps, atol=2e-5; unchanged",
                    next="read-only diagnose earliest stage; no kernel/runtime changes",
                ),
            )
            print("STOP_NUMERICAL_GATE", n, flush=True)
            return
    log(
        "E03 numerical gate complete. Dedicated performance harness may proceed "
        "after evidence review."
    )
    print("FUNCTIONAL_GATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
