"""Analyze saved TP4/TP8 timing and traces using only the standard library."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def write(output, name, obj):
    with (output / name).open("x") as f:
        json.dump(obj, f, indent=2)


def csvwrite(output, name, rows):
    if not rows:
        return
    with (output / name).open("x", newline="") as f:
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


def union(rows):
    total = 0
    end = float("-inf")
    for a, b in sorted((g["start_us"], g["start_us"] + g["duration_us"]) for g in rows):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def trace(path, output, layers):
    events = json.loads(path.read_text())["traceEvents"]
    tags = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"].startswith(("TP4/", "TP8/"))
        and "dur" in e
    ]
    # These are historical producer namespaces, not evidence of the TP size.
    prefixes = {e["name"].split("/", 1)[0] for e in tags}
    assert len(prefixes) == 1, (path, "Missing or mixed TP annotations", prefixes)
    prefix = prefixes.pop() + "/"
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    rows = []
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(e.get("args", {}).get("correlation"))
        parents = (
            []
            if launch is None
            else sorted(
                [
                    t
                    for t in tags
                    if t["pid"] == launch["pid"]
                    and t["tid"] == launch["tid"]
                    and t["ts"] <= launch["ts"] < t["ts"] + t["dur"]
                ],
                key=lambda t: t["dur"],
            )
        )
        inside = any(t["name"] == prefix + "forward" for t in parents)
        module = parents[0]["name"].removeprefix(prefix) if parents else "outside"
        layer = next(
            (
                t["name"].rsplit("/", 1)[1]
                for t in parents
                if t["name"].startswith(prefix + "layer/")
            ),
            "",
        )
        name = e["name"]
        category = "other"
        if name == "_fused_post_pre":
            category = "fused"
        elif name in ("_ihc_pre_stage1", "_ihc_pre_stage2"):
            category = "ihc_pre"
        elif name == "_ihc_post_kernel":
            category = "ihc_post"
        elif any(
            x in name.lower()
            for x in ("nccl", "allreduce", "all_reduce", "allgather", "reduce_scatter")
        ):
            category = "communication"
        elif module.startswith("attention/"):
            category = "attention"
        elif module.startswith("dense/"):
            category = "dense_mlp"
        elif module.startswith("shared_moe/"):
            category = "shared_moe"
        elif module.startswith("routed_moe/"):
            category = "routed_moe"
        elif module.startswith("moe/"):
            category = "moe_other"
        elif module == "head":
            category = "hc_head"
        target = (
            category == "fused"
            or (
                category == "ihc_pre"
                and (
                    module.startswith("pre/mlp/")
                    or (
                        module.startswith("pre/attention/")
                        and module != f"pre/attention/{layers[0]}"
                    )
                )
            )
            or (
                category == "ihc_post"
                and (
                    module.startswith("post/attention/")
                    or (
                        module.startswith("post/mlp/")
                        and module != f"post/mlp/{layers[-1]}"
                    )
                )
            )
        )
        rows.append(
            dict(
                name=name,
                category=category,
                module=module,
                layer=layer,
                inside_forward=inside,
                target_boundary=target,
                start_us=e["ts"],
                duration_us=e["dur"],
                stream=e.get("args", {}).get("stream"),
                grid=e.get("args", {}).get("grid"),
                block=e.get("args", {}).get("block"),
            )
        )
    active = [g for g in rows if g["inside_forward"]]
    assert active, (path, "No GPU activity correlated to an annotated forward")
    csvwrite(output, path.stem.replace(".trace", "") + "_activity.csv", rows)
    categories = defaultdict(lambda: {"count": 0, "sum_us": 0})
    for g in active:
        categories[g["category"]]["count"] += 1
        categories[g["category"]]["sum_us"] += g["duration_us"]
    total = sum(g["duration_us"] for g in active)
    span = max(g["start_us"] + g["duration_us"] for g in active) - min(
        g["start_us"] for g in active
    )
    ihc = sum(
        g["duration_us"]
        for g in active
        if g["category"] in ("ihc_pre", "ihc_post", "fused")
    )
    target = sum(g["duration_us"] for g in active if g["target_boundary"])
    return dict(
        categories=dict(categories),
        activity_sum_us=total,
        union_us=union(active),
        span_us=span,
        overlap_us=total - union(active),
        gaps_us=span - union(active),
        ihc_us=ihc,
        ihc_activity_pct=100 * ihc / total,
        target_us=target,
        target_activity_pct=100 * target / total,
        counts=dict(
            Counter(
                g["name"]
                for g in active
                if g["category"] in ("ihc_pre", "ihc_post", "fused")
            )
        ),
        fused=[g for g in active if g["category"] == "fused"],
        communication_names=sorted(
            {g["name"] for g in active if g["category"] == "communication"}
        ),
        outside_gpu_count=sum(not g["inside_forward"] for g in rows),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved run directory containing raw/")
    parser.add_argument(
        "--output-dir", type=Path, help="Fresh analysis output directory"
    )
    args = parser.parse_args(argv)
    raw = args.run / "raw"
    output = args.output_dir or raw
    if args.output_dir:
        output.mkdir(parents=True)
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
    paths = sorted(raw.glob("*.trace.json"))
    assert {p.name for p in paths} == {
        f"{mode}_rank{rank}.trace.json" for mode in modes for rank in ranks
    }, "Missing or unexpected rank/mode traces"
    profiles = {p.stem.replace(".trace", ""): trace(p, output, layers) for p in paths}
    for mode in modes:
        fused = {"original": 0, "intra": len(layers), "cross": 2 * len(layers) - 1}[
            mode
        ]
        if not 512 <= n <= 8192:
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
    write(output, "timing_summary.json", {"forward": timing, "paired": paired})
    write(output, "activity_summary.json", profiles)
    metadata = dict(
        world_size=tp,
        layers=layers,
        tokens=n,
        rounds=len(rounds),
        modes=modes,
        run_config=config,
    )
    write(output, "analysis_metadata.json", metadata)
    print(json.dumps(dict(metadata=metadata, forward=timing, paired=paired), indent=2))
    for k, v in profiles.items():
        print(
            k,
            {
                s: v[s]
                for s in (
                    "activity_sum_us",
                    "span_us",
                    "ihc_activity_pct",
                    "target_activity_pct",
                    "counts",
                )
            },
        )


if __name__ == "__main__":
    main()
