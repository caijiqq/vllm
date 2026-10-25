# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent HY4 model checks or frozen local correctness/Graph regressions."""

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

from lab.ihc_fusion.benchmarks.tp4_validation.analyze_perf import write


def parse_args(argv=None, *, benchmark=False):
    parser = argparse.ArgumentParser(description=__doc__)
    selected = (
        os.getenv("HY4_TP8_GPUS", "").split(",") if os.getenv("HY4_TP8_GPUS") else []
    )
    parser.add_argument("--tp-size", type=int, default=len(selected) or 8)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("original", "intra", "cross"),
        default=["original", "intra", "cross"],
    )
    parser.add_argument("--warmup", type=int, default=6 if benchmark else 5)
    parser.add_argument("--llm-overrides", type=json.loads, default={}, metavar="JSON")
    if benchmark:
        parser.add_argument("--tokens", type=int, default=8192)
        parser.add_argument("--rounds", type=int, default=9)
        parser.add_argument("--profile", action="store_true")
    else:
        parser.add_argument("--shapes", nargs="+", type=int, default=[512, 8192])
        parser.add_argument("--suite", choices=("model", "local"), default="model")
        parser.add_argument("--aa-repeats", type=int, default=2)
    options = parser.parse_args(argv)
    shapes = [options.tokens] if benchmark else options.shapes
    if options.tp_size < 1 or (selected and len(selected) != options.tp_size):
        parser.error("--tp-size must be positive and match HY4_TP8_GPUS")
    if options.warmup < 0 or any(n < 1 for n in shapes):
        parser.error("Shapes must be positive and warmup must be nonnegative")
    if benchmark and options.rounds < 1:
        parser.error("Rounds must be positive")
    if not benchmark and options.aa_repeats < 2:
        parser.error("A/A requires at least two Original captures")
    if len(set(shapes)) != len(shapes):
        parser.error("Shapes must be unique")
    if options.modes[0] != "original" or len(set(options.modes)) != len(options.modes):
        parser.error("Modes must be unique and start with the original baseline")
    if not isinstance(options.llm_overrides, dict):
        parser.error("--llm-overrides must be a JSON object")
    return options


def model_options(options):
    tokens = max(
        64, options.tokens if hasattr(options, "tokens") else max(options.shapes)
    )
    overrides = getattr(options, "llm_overrides", {}).copy()
    if {"model", "tensor_parallel_size"} & overrides.keys():
        raise ValueError("Use HY4_TP8_CHECKPOINT and --tp-size for model and TP")
    kwargs = dict(
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
            **overrides.pop("hf_overrides", {}),
        },
    )
    kwargs.update(overrides)
    if not kwargs["enforce_eager"] or kwargs["pipeline_parallel_size"] != 1:
        raise ValueError(
            "Model hooks require Eager/PP1; "
            "use validate.py --suite local for Graph regression"
        )
    return kwargs


def prepare_run(options, phase):
    root = Path(os.environ["HY4_TP8_RUN"]) / phase
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if any(raw.iterdir()):
        raise ValueError(f"Use a fresh output directory: {raw}")
    kwargs = (
        None if getattr(options, "suite", None) == "local" else model_options(options)
    )
    write(raw, "run_config.json", dict(options=vars(options), llm=kwargs))
    return root, kwargs


def require_ranks(states, mode=None, *, tp_size):
    assert len(states) == tp_size and sorted(x["rank"] for x in states) == list(
        range(tp_size)
    ), states
    assert all(x["world_size"] == tp_size for x in states), states
    if mode is not None:
        for state in states:
            eligible = state["fusion_eligible"]
            assert state["flags"] == [
                dict(
                    intra=mode != "original" and enabled,
                    cross=mode == "cross" and i < len(eligible) - 1,
                )
                for i, enabled in enumerate(eligible)
            ], state


def main():
    options = parse_args()
    root, kwargs = prepare_run(options, "validation")
    raw = root / "raw"
    if options.suite == "local":
        repo = Path(__file__).resolve().parents[4]
        env = dict(
            os.environ,
            IHC_CORRECTNESS_JSON=str(raw / "kernel_errors.json"),
            IHC_LAUNCH_JSON=str(raw / "launches.json"),
        )
        # Isolate CUPTI activity state and Runtime fixtures between test files.
        targets = [
            (path.stem, ["--confcutdir=lab/ihc_fusion", str(path.relative_to(repo))])
            for path in sorted((repo / "lab/ihc_fusion/tests").glob("test_*.py"))
        ]
        targets.append(
            (
                "runtime",
                ["tests/models/test_hyv4_ihc_fusion.py"],
            )
        )
        results = []
        for name, tests in targets:
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "-q",
                "-p",
                "no:cacheprovider",
                *tests,
                f"--junitxml={raw / (name + '.xml')}",
            ]
            with (root / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            print((root / f"{name}.log").read_text(), flush=True)
            results.append(
                dict(suite=name, returncode=result.returncode, command=command)
            )
        passed = all(item["returncode"] == 0 for item in results)
        write(
            raw,
            "correctness_summary.json",
            dict(suite="local", passed=passed, results=results),
        )
        return 0 if passed else 1

    from lab.ihc_fusion.benchmarks.tp8_validation import workers
    from vllm import LLM, SamplingParams

    require = functools.partial(require_ranks, tp_size=options.tp_size)
    llm = LLM(**kwargs)
    states = llm.apply_model(workers.status)
    require(states)
    write(raw, "initial_workers.json", states)
    assert sorted(llm.apply_model(workers.install_observers)) == list(
        range(options.tp_size)
    )
    params = SamplingParams(temperature=0, max_tokens=1, seed=7)
    summary = []
    for n in dict.fromkeys([64, *options.shapes]):
        prompt = [{"prompt_token_ids": [100 + i % 197 for i in range(n)]}]
        llm.apply_model(functools.partial(workers.set_mode, name="original"))
        for _ in range(options.warmup):
            llm.generate(prompt, params, use_tqdm=False)
        rows = []
        for index, mode in enumerate(
            ["original"] * (options.aa_repeats - 1) + options.modes
        ):
            require(
                llm.apply_model(functools.partial(workers.set_mode, name=mode)), mode
            )
            llm.apply_model(
                functools.partial(
                    workers.capture, expected_ids=prompt[0]["prompt_token_ids"]
                )
            )
            tokens = (
                llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].token_ids
            )
            values = llm.apply_model(
                functools.partial(workers.captured, baseline=index == 0)
            )
            require(values)
            rows.append(
                dict(
                    mode=mode,
                    comparison="baseline" if index == 0 else f"original/{mode}",
                    n=n,
                    tokens=tokens,
                    workers=values,
                )
            )
        write(raw, f"correctness_{n}.json", rows)
        token_match = all(row["tokens"] == rows[0]["tokens"] for row in rows)
        passed = token_match and all(w["pass"] for row in rows for w in row["workers"])
        status = (
            "INVALID"
            if any(not w["valid"] for row in rows for w in row["workers"])
            else "PASS"
            if passed
            else "FAIL"
        )
        summary.append(
            dict(tokens=n, passed=passed, token_match=token_match, status=status)
        )
        print(
            f"N={n}: {status} (strict hidden states, boundaries and route)",
            flush=True,
        )
    passed = all(row["passed"] for row in summary)
    write(
        raw,
        "correctness_summary.json",
        dict(
            suite="model",
            passed=passed,
            shapes=summary,
            rule="existing chain rtol=2*BF16 eps, atol=2e-5; unchanged",
            scope="checkpoint comparisons; frozen Golden tests are in --suite local",
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
