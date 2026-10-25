# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze saved TP4/TP8 timing and traces using only the standard library."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path


def write(output, name, obj):
    with (output / name).open("w") as f:
        json.dump(obj, f, indent=2)


def csvwrite(output, name, rows):
    if not rows:
        return
    with (output / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(vals):
    vals = sorted(vals)

    def pct(q):
        p = (len(vals) - 1) * q
        i = int(p)
        return vals[i] + (vals[min(i + 1, len(vals) - 1)] - vals[i]) * (p - i)

    return dict(
        median=statistics.median(vals),
        p10=pct(0.1),
        p90=pct(0.9),
        min=min(vals),
        max=max(vals),
        n=len(vals),
    )


def trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"] in ("TP4/forward", "TP8/forward")
        and "dur" in e
    ]
    assert scopes, (path, "Missing Forward scope")
    launches = {
        e["args"]["correlation"]
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
        and any(
            t["pid"] == e["pid"]
            and t["tid"] == e["tid"]
            and t["ts"] <= e["ts"] < t["ts"] + t["dur"]
            for t in scopes
        )
    }
    kernels = [
        e
        for e in events
        if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in launches
    ]
    assert kernels, (path, "No GPU kernels correlated to Forward")
    return dict(counts=dict(Counter(e["name"] for e in kernels)))


def analyze(run, output_dir=None):
    raw = Path(run) / "raw"
    output = Path(output_dir) if output_dir else raw
    output.mkdir(parents=True, exist_ok=True)
    workers = json.loads((raw / "initial_workers.json").read_text())
    tp = workers[0]["world_size"]
    ranks = list(range(tp))
    assert sorted(w["rank"] for w in workers) == ranks
    assert all(w["world_size"] == tp for w in workers)
    layers = [p["layer"] for p in workers[0]["partitions"]]
    assert layers and all(
        [p["layer"] for p in w["partitions"]] == layers for w in workers
    )
    warm = json.loads((raw / "warm_original.json").read_text())["workers"]
    assert sorted(w["rank"] for w in warm) == ranks
    n = warm[0]["records"][0]["shape"][0]
    assert all(len(w["records"]) == 1 and w["records"][0]["shape"] == [n] for w in warm)
    config_path = raw / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    modes = config["options"]["modes"] if config else ["original", "intra", "cross"]
    with (raw / "timing.csv").open() as f:
        reader = csv.DictReader(f)
        assert {k for k in reader.fieldnames if k.startswith("rank")} == {
            f"rank{i}_ms" for i in ranks
        }
        rows = list(reader)
    rounds = sorted({int(r["round"]) for r in rows})
    assert rounds and rounds == list(range(len(rounds))), "Missing timing rounds"
    if config:
        assert config["llm"]["tensor_parallel_size"] == tp
        assert config["options"]["tokens"] == n
        assert len(rounds) == config["options"]["rounds"]
    else:
        # Both legacy producers fixed nine rounds without recording this setting.
        assert len(rounds) == 9, "Incomplete legacy run (expected nine rounds)"
    for row in rows:
        times = [float(row[f"rank{i}_ms"]) for i in ranks]
        assert all(math.isfinite(ms) and ms > 0 for ms in times), row
        assert math.isclose(float(row["forward_max_ms"]), max(times)), row
    for i in rounds:
        assert sorted(r["mode"] for r in rows if int(r["round"]) == i) == sorted(
            modes
        ), i
    timing = {
        m: stats([float(r["forward_max_ms"]) for r in rows if r["mode"] == m])
        for m in modes
    }
    paired = {}
    for a, b in combinations(modes, 2):
        vals, deltas = [], []
        for i in rounds:
            values = {
                r["mode"]: float(r["forward_max_ms"])
                for r in rows
                if int(r["round"]) == i
            }
            vals.append(values[a] / values[b])
            deltas.append(values[a] - values[b])
        paired[a + "/" + b] = {"speedup": stats(vals), "saved_ms": stats(deltas)}
    write(output, "timing_summary.json", {"forward": timing, "paired": paired})
    print("Mode       median ms     P10 ms     P90 ms", flush=True)
    for mode, values in timing.items():
        print(
            f"{mode:10s} {values['median']:9.3f} "
            f"{values['p10']:10.3f} {values['p90']:10.3f}",
            flush=True,
        )
    for pair, values in paired.items():
        print(
            f"{pair}: paired median {values['speedup']['median']:.4f}x, "
            f"saved {values['saved_ms']['median']:.3f} ms",
            flush=True,
        )
    paths = sorted(raw.glob("*.trace.json"))
    if paths:
        assert {p.name for p in paths} == {
            f"{mode}_rank{rank}.trace.json" for mode in modes for rank in ranks
        }, "Missing or unexpected rank/mode traces"
    profiles = {p.stem.replace(".trace", ""): trace(p) for p in paths}
    for mode in modes if profiles else []:
        fused = {"original": 0, "intra": len(layers), "cross": 2 * len(layers) - 1}[
            mode
        ]
        if (
            not 512 <= n <= 8192
            or not all(f["intra"] for f in workers[0]["flags"])
            or (
                config
                and (
                    not config["llm"]["enforce_eager"]
                    or config["llm"].get("dtype", "bfloat16") != "bfloat16"
                )
            )
        ):
            fused = 0
        for rank in ranks:
            counts = profiles[f"{mode}_rank{rank}"]["counts"]
            assert counts.get("_fused_post_pre", 0) == fused, (mode, rank, counts)
            for name in ("_ihc_pre_stage1", "_ihc_pre_stage2", "_ihc_post_kernel"):
                assert counts.get(name, 0) == 2 * len(layers) - fused, (
                    mode,
                    rank,
                    counts,
                )
    if profiles:
        write(output, "activity_summary.json", profiles)
    metadata = dict(
        world_size=tp,
        layers=layers,
        tokens=n,
        rounds=len(rounds),
        modes=modes,
        run_config=config,
    )
    metadata["kernel_hit_verified"] = bool(profiles)
    write(output, "analysis_metadata.json", metadata)
    for name, profile in profiles.items():
        print(
            name,
            {
                k: v
                for k, v in profile["counts"].items()
                if "ihc" in k or "fused_post_pre" in k
            },
        )
    return dict(metadata=metadata, forward=timing, paired=paired)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved run directory containing raw/")
    parser.add_argument("--output-dir", type=Path, help="Directory for derived results")
    args = parser.parse_args(argv)
    analyze(args.run, args.output_dir)


if __name__ == "__main__":
    main()
