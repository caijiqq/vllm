# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the public HY4 D=6144 contract absent from the original suite."""

import pytest
import torch


@pytest.mark.parametrize("enabled", [False, True])
def test_serving_prefix_cache_survives_subparser(tmp_path, enabled):
    """The effective server option must match the benchmark's cache policy."""
    import argparse
    import json

    from lab.ihc_fusion.benchmarks.tp8_validation.serving import server_arguments
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    serve = parser.add_subparsers().add_parser("serve")
    serve.add_argument("model_tag")
    serve.add_argument("--config")
    serve.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    path = tmp_path / "server.yaml"
    path.write_text(json.dumps({"enable-prefix-caching": enabled}))
    args = parser.parse_args(
        ["serve", "hy4", *server_arguments(path, {"enable_prefix_caching": enabled})]
    )
    assert args.enable_prefix_caching is enabled


def test_model_harness_configuration(monkeypatch):
    """The shared entrypoints preserve TP, shapes, modes and runtime overrides."""
    from lab.ihc_fusion.benchmarks.tp8_validation.validate import (
        model_options,
        parse_args,
    )

    monkeypatch.setenv("HY4_TP8_GPUS", "2,5")
    monkeypatch.setenv("HY4_TP8_CHECKPOINT", "/tmp/hy4-checkpoint")
    options = parse_args(
        [
            "--tokens",
            "513",
            "--rounds",
            "2",
            "--warmup",
            "0",
            "--modes",
            "original",
            "cross",
            "--llm-overrides",
            '{"max_num_seqs": 7, "hf_overrides": {"custom_field": 3}}',
        ],
        benchmark=True,
    )
    config = model_options(options)
    assert config["tensor_parallel_size"] == options.tp_size == 2
    assert config["max_num_batched_tokens"] == 513
    assert config["max_num_seqs"] == 7 and config["hf_overrides"]["custom_field"] == 3
    assert config["hf_overrides"]["enable_ihc_cross_layer_fusion"] is True
    assert options.modes == ["original", "cross"] and not options.profile
    options = parse_args(["--suite", "local", "--shapes", "128", "512"])
    assert options.suite == "local" and options.shapes == [128, 512]


def test_timing_analysis_without_gate_or_profiler(tmp_path, capsys):
    """Use per-sample rank maxima and paired statistics without requiring traces."""
    import csv
    import json

    from lab.ihc_fusion.benchmarks.tp4_validation.analyze_perf import analyze, write

    raw = tmp_path / "raw"
    raw.mkdir()
    modes = ["original", "intra", "cross"]
    workers = [
        dict(
            rank=r,
            world_size=2,
            partitions=[dict(layer=i) for i in range(5)],
            flags=[dict(intra=True) for _ in range(5)],
        )
        for r in range(2)
    ]
    write(raw, "initial_workers.json", workers)
    write(
        raw,
        "warm_original.json",
        dict(workers=[dict(rank=r, records=[dict(shape=[512])]) for r in range(2)]),
    )
    write(
        raw,
        "run_config.json",
        dict(
            options=dict(tokens=512, modes=modes, rounds=3),
            llm=dict(tensor_parallel_size=2, enforce_eager=True),
        ),
    )
    times = dict(original=[10, 14, 12], intra=[9, 11, 10], cross=[8, 9, 7])
    with (raw / "timing.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, ["round", "mode", "rank0_ms", "rank1_ms", "forward_max_ms"]
        )
        writer.writeheader()
        for step in range(3):
            for mode in modes:
                ms = times[mode][step]
                writer.writerow(
                    dict(
                        round=step,
                        mode=mode,
                        rank0_ms=ms - 1,
                        rank1_ms=ms,
                        forward_max_ms=ms,
                    )
                )
    result = analyze(tmp_path)
    assert result["forward"]["original"]["median"] == 12
    assert result["paired"]["original/cross"]["speedup"]["median"] == 14 / 9
    assert result["paired"]["original/cross"]["saved_ms"]["median"] == 5
    assert not result["metadata"]["kernel_hit_verified"]
    saved = json.loads((raw / "timing_summary.json").read_text())
    assert saved["paired"] == result["paired"]
    assert "saved 5.000 ms" in capsys.readouterr().out
    # Reanalysis replaces derived summaries, preserving the raw samples.
    assert analyze(tmp_path) == result


@pytest.mark.parametrize(
    "fault",
    ["missing", "stale", "multiple", "graph", "tokens", "positions", "reference"],
)
def test_invalid_capture_is_not_a_numerical_failure(monkeypatch, fault):
    """Never compare overwritten, stale or logically different request outputs."""
    from types import SimpleNamespace

    from lab.ihc_fusion.benchmarks.tp8_validation import workers

    monkeypatch.setattr(
        workers, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    record = dict(
        graph="NONE",
        input_ids=[100, 101],
        positions=[0, 1],
        query_start_loc=[0, 2],
        num_actual_tokens=2,
        max_seq_len=2,
    )
    state = dict(
        latest=torch.zeros(2, 4),
        inputs=[record],
        expected_ids=[100, 101],
        snapshot_id=2,
        capture_id=2,
        reference=None,
        reference_inputs=None,
    )
    if fault == "missing":
        state["latest"] = None
    elif fault == "stale":
        state["snapshot_id"] = 1
    elif fault == "multiple":
        state["inputs"] *= 2
    elif fault == "graph":
        record["graph"] = "FULL"
    elif fault == "tokens":
        record["input_ids"] = [101, 100]
    elif fault == "positions":
        record["positions"] = [2, 3]
    result = workers.captured(
        SimpleNamespace(_tp8_experiment=state), baseline=fault != "reference"
    )
    assert result["status"] == "INVALID" and not result["pass"]
    assert "metrics" not in result and state["latest"] is None


def test_hidden_comparison_preserves_strict_failure():
    from lab.ihc_fusion.benchmarks.tp8_validation.workers import compare

    reference = torch.ones(2, 4, dtype=torch.bfloat16)
    assert compare(reference, reference)["bitwise_equal"]
    assert compare(reference[:1], reference)["status"] == "INVALID"
    result = compare(reference + 0.125, reference)
    assert result["valid"] and result["status"] == "FAIL"
    assert result["unequal_elements"] == 8 and result["ulp_at_max_abs"] == 16


def test_full_prompt_comparison_ignores_physical_cache_addresses(monkeypatch):
    """Fresh full prompts can use different KV blocks without changing input data."""
    from types import SimpleNamespace

    from lab.ihc_fusion.benchmarks.tp8_validation import workers

    monkeypatch.setattr(
        workers, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    state = dict(capture_id=0, reference=None, mode="original")
    model = SimpleNamespace(
        _tp8_experiment=state,
        _ihc_fusion_eligible=[False],
        model=SimpleNamespace(layers=[None]),
    )
    for baseline, block in [(True, 10), (False, 20)]:
        workers.capture(model, expected_ids=[100, 101])
        state.update(
            latest=torch.ones(2, 4, dtype=torch.bfloat16),
            snapshot_id=state["capture_id"],
            layers=[0],
            shapes=[dict(layer=0, module="hc_pre", kind="attention")],
            boundaries={
                name: torch.ones(2, 4)
                for name in ["layer0_input_norm", "layer0_final_post"]
            },
            inputs=[
                dict(
                    graph="NONE",
                    input_ids=[100, 101],
                    positions=[0, 1],
                    query_start_loc=[0, 2],
                    num_actual_tokens=2,
                    max_seq_len=2,
                    block_table=[[block]],
                    slot_mapping=[block * 64, block * 64 + 1],
                )
            ],
        )
        result = workers.captured(model, baseline=baseline)
        assert result["valid"] and result["status"] == "PASS"
    assert result["bitwise_equal"] and result["forward_calls"] == 1
