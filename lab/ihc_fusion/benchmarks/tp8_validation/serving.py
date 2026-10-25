# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial HY4 serving comparisons using the native vllm serve / bench serve CLI."""

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from lab.ihc_fusion.benchmarks.tp4_validation.analyze_perf import csvwrite, write

METRICS = (
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "request_throughput",
    "output_throughput",
)


def server_arguments(config_path, config):
    """Pass prefix caching explicitly across the CLI subparser boundary."""
    prefix_flag = (
        "--enable-prefix-caching"
        if config["enable_prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    return ["--config", str(config_path), prefix_flag]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tp-size", type=int, default=len(os.getenv("HY4_TP8_GPUS", "").split(","))
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1])
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--request-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--profile", action="store_true", help="Diagnostic traces; no speedup claims"
    )
    parser.add_argument("--modes", nargs="+", choices=("original", "intra", "cross"))
    parser.add_argument("--llm-overrides", type=json.loads, default={})
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Deadline per startup/benchmark, seconds",
    )
    args = parser.parse_args()
    args.modes = args.modes or (
        ["original"] if args.execution == "graph" else ["original", "intra", "cross"]
    )
    if args.execution == "graph" and args.modes != ["original"]:
        parser.error("Graph is an Original fallback check, not a fusion A/B")
    if args.modes[0] != "original" or len(set(args.modes)) != len(args.modes):
        parser.error("Unique modes must start with original")
    if (
        min(
            args.tp_size,
            args.input_len,
            args.request_count,
            args.rounds,
            args.timeout,
            *args.concurrency,
        )
        < 1
        or args.output_len < 2
        or args.warmup < 0
        or not args.request_rate > 0
    ):
        parser.error(
            "Positive sizes/rounds/timeout, output-len >= 2, warmup >= 0 required"
        )
    if len(set(args.concurrency)) != len(args.concurrency) or not 0 < args.port < 65536:
        parser.error("Concurrency values must be unique; port must be 1..65535")
    if args.tp_size != len(os.environ["HY4_TP8_GPUS"].split(",")) or not isinstance(
        args.llm_overrides, dict
    ):
        parser.error("TP must match selected GPUs; llm-overrides must be an object")
    if {
        "host",
        "port",
        "enforce_eager",
        "served_model_name",
    } & args.llm_overrides.keys():
        parser.error("Use dedicated host/port/execution options; model name is managed")
    return args


def result_row(result, args, mode, round_id, concurrency):
    valid = (
        result["completed"] == args.request_count
        and not result["failed"]
        and not any(result["errors"])
        and result["input_lens"] == [args.input_len] * args.request_count
        and result["output_lens"] == [args.output_len] * args.request_count
        and len(result["ttfts"]) == len(result["itls"]) == args.request_count
        and result["max_concurrency"] == concurrency
        and all(math.isfinite(result[key]) and result[key] > 0 for key in METRICS)
    )
    return dict(
        execution=args.execution,
        round=round_id,
        mode=mode,
        concurrency=concurrency,
        valid=valid,
        success_rate=result["completed"] / args.request_count,
        **{key: result[key] if valid else None for key in METRICS},
    )


def summarize(root, rows, args):
    for row in rows:
        group = [
            x
            for x in rows
            if (x["round"], x["concurrency"]) == (row["round"], row["concurrency"])
        ]
        paired = len(group) == len(args.modes) and all(x["valid"] for x in group)
        baseline = next((x for x in group if x["mode"] == "original"), None)
        row["comparison_valid"] = paired
        for key in METRICS:
            row[key + "_speedup"] = (
                (
                    baseline[key] / row[key]
                    if key.endswith("_ms")
                    else row[key] / baseline[key]
                )
                if paired and args.execution == "eager" and not args.profile
                else None
            )
    write(
        root,
        "summary.json",
        dict(
            numerical_accepted=False,
            kernel_hit_verified=False,
            profiling=args.profile,
            rows=rows,
        ),
    )
    csvwrite(root, "summary.csv", rows)
    print(
        "mode     round conc valid success TTFT P50/P90 ms "
        "TPOT P50/P90 ms  req/s  out tok/s  TTFTx TPOTx",
        flush=True,
    )
    for row in rows:
        values = [row[k] for k in METRICS] + [
            row[k + "_speedup"] for k in ("p50_ttft_ms", "p50_tpot_ms")
        ]
        print(
            f"{row['mode']:8} {row['round']:3} {row['concurrency']:4} "
            f"{str(row['comparison_valid']):5} {row['success_rate']:.1%} "
            + " ".join("-" if v is None else f"{v:.3f}" for v in values),
            flush=True,
        )
    print(
        f"Saved: {root}; performance does not establish numerical acceptance.",
        flush=True,
    )


def main():
    from lab.ihc_fusion.benchmarks.tp8_validation.validate import model_options
    from tests.utils import RemoteOpenAIServer

    class _Server(RemoteOpenAIServer):
        """Use this harness's interpreter; native server owns readiness and cleanup."""

        def _start_server(self, model, vllm_serve_args, env_dict):
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "serve",
                    model,
                    *vllm_serve_args,
                ],
                env={
                    **os.environ,
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    **(env_dict or {}),
                },
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
            )

    args = parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    label = args.execution + ("_profile" if args.profile else "")
    root = Path(os.environ["HY4_TP8_RUN"]) / "serving" / label
    root.mkdir(parents=True, exist_ok=False)
    args.tokens = args.input_len
    config = model_options(args)
    config.setdefault("seed", args.seed)
    config.update(
        host=args.host,
        port=args.port,
        enforce_eager=args.execution == "eager",
        served_model_name="hy4-ihc",
        generation_config="vllm",
    )
    if "max_model_len" not in args.llm_overrides:
        config["max_model_len"] = args.input_len + args.output_len
    if "max_num_seqs" not in args.llm_overrides:
        config["max_num_seqs"] = max(args.concurrency)
    if config["max_model_len"] < args.input_len + args.output_len:
        raise ValueError("max_model_len is smaller than the requested workload")
    workload = {
        "dataset-name": "random",
        "input-len": args.input_len,
        "output-len": args.output_len,
        "random-range-ratio": 0,
        "random-prefix-len": 0,
        "num-prompts": args.request_count,
        "seed": args.seed,
        "tokenizer": config.get("tokenizer", config["model"]),
    }
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    write(
        root,
        "run_config.json",
        dict(
            options=vars(args),
            server=config,
            workload=workload,
            workload_sha256=digest,
            runtime_sha256=hashlib.sha256(
                Path("vllm/models/hy_v4/nvidia/model.py").read_bytes()
            ).hexdigest(),
        ),
    )
    cli = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base_url = f"http://{host}:{args.port}"
    rows = []
    try:
        for round_id in range(args.rounds):
            offset = round_id % len(args.modes)
            for mode in args.modes[offset:] + args.modes[:offset]:
                run = root / f"r{round_id}_{mode}"
                run.mkdir()
                # Reuse the existing idle gate before each owned server starts.
                subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        str(Path(__file__).with_name("check_gpus.py")),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                with socket.socket() as probe:
                    probe.bind((args.host, args.port))
                flags = dict(
                    enable_ihc_post_pre_fusion=mode != "original",
                    enable_ihc_cross_layer_fusion=mode == "cross",
                )
                config["hf_overrides"].update(flags)
                if args.profile:
                    config["profiler_config"] = dict(
                        profiler="torch",
                        torch_profiler_dir=str(run / "traces"),
                        torch_profiler_with_stack=False,
                        ignore_frontend=True,
                        max_iterations=12,
                    )
                # JSON is valid YAML; vLLM's own config loader handles CLI conversion.
                config_path = run / "server.yaml"
                config_path.write_text(
                    json.dumps(
                        {k.replace("_", "-"): v for k, v in config.items()}, indent=2
                    )
                )
                serve_args = server_arguments(config_path, config)
                command = cli + ["serve", config["model"], *serve_args]
                write(run, "command.json", command)
                with (
                    (run / "server.log").open("w") as log,
                    redirect_stdout(log),
                    redirect_stderr(log),
                    _Server(
                        config["model"],
                        serve_args,
                        seed=None,
                        auto_port=False,
                        max_wait_seconds=args.timeout,
                    ) as server,
                ):
                    if server.proc.poll() is not None:
                        raise RuntimeError("Owned server exited")
                    lines = (run / "server.log").read_text().splitlines()
                    effective = next(
                        line for line in lines if "non-default args:" in line
                    )
                    if not all(
                        f"'{key}': {value}" in effective for key, value in flags.items()
                    ):
                        raise RuntimeError(
                            "Startup did not confirm the requested HY4 flags"
                        )
                    engine_config = next(
                        s for s in lines if "Initializing a V1 LLM engine" in s
                    )
                    expected_cache = config["enable_prefix_caching"]
                    if f"enable_prefix_caching={expected_cache}," not in engine_config:
                        raise RuntimeError(
                            "Effective prefix caching differs from config"
                        )
                    captures = [s for s in lines if "Graph capturing finished" in s]
                    if args.execution == "graph" and not captures:
                        raise RuntimeError("Graph capture was not confirmed by vLLM")
                    write(
                        run,
                        "startup.json",
                        dict(
                            pid=server.proc.pid,
                            flags=flags,
                            enable_prefix_caching=expected_cache,
                            native_parsed_args=effective,
                            graph_capture=captures,
                            native_engine_config=[
                                s
                                for s in lines
                                if "Initializing a V1 LLM engine" in s
                                or "Overriding to" in s
                            ],
                        ),
                    )
                    for concurrency in args.concurrency:
                        filename = f"c{concurrency}.json"
                        options = {
                            "backend": "vllm",
                            "base-url": base_url,
                            "model": "hy4-ihc",
                            **workload,
                            "served-model-name": "hy4-ihc",
                            "max-concurrency": concurrency,
                            "num-warmups": args.warmup,
                            "temperature": 0,
                            "request-rate": args.request_rate,
                            "extra-body": json.dumps(dict(seed=args.seed)),
                            "percentile-metrics": "ttft,tpot,itl,e2el",
                            "metric-percentiles": "50,90",
                            "result-dir": run,
                            "result-filename": filename,
                            "ready-check-timeout-sec": args.timeout,
                        }
                        command = (
                            cli
                            + ["bench", "serve"]
                            + [
                                "--ignore-eos",
                                "--trust-remote-code",
                                "--save-result",
                                "--save-detailed",
                                "--disable-tqdm",
                            ]
                            + [
                                s
                                for k, v in options.items()
                                for s in ("--" + k, str(v))
                            ]
                        )
                        if args.profile:
                            command.append("--profile")
                        write(run, f"c{concurrency}_command.json", command)
                        with (run / f"c{concurrency}.log").open("w") as output:
                            subprocess.run(
                                command,
                                check=True,
                                timeout=args.timeout,
                                stdout=output,
                                stderr=subprocess.STDOUT,
                            )
                        result = json.loads((run / filename).read_text())
                        rows.append(
                            result_row(result, args, mode, round_id, concurrency)
                        )
                time.sleep(1)  # Let driver utilization accounting settle after exit.
    finally:
        summarize(root, rows, args)
    return (
        0
        if len(rows) == args.rounds * len(args.modes) * len(args.concurrency)
        and all(row["valid"] for row in rows)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
