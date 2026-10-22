"""Derive nonoverlapping categories and paired statistics from saved TP4 data."""

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(sys.argv[1])
RAW = R / "raw"


def write(name, obj):
    with (RAW / name).open("x") as f:
        json.dump(obj, f, indent=2)


def csvwrite(name, rows):
    if not rows:
        return
    with (RAW / name).open("x", newline="") as f:
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


def trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    tags = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"].startswith("TP4/")
        and "dur" in e
    ]
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
                    if t["tid"] == launch["tid"]
                    and t["ts"] <= launch["ts"] < t["ts"] + t["dur"]
                ],
                key=lambda t: t["dur"],
            )
        )
        inside = any(t["name"] == "TP4/forward" for t in parents)
        module = parents[0]["name"][4:] if parents else "outside"
        layer = next(
            (t["name"][10:] for t in parents if t["name"].startswith("TP4/layer/")), ""
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
                    or module in ("pre/attention/1", "pre/attention/2")
                )
            )
            or (
                category == "ihc_post"
                and (
                    module.startswith("post/attention/")
                    or module in ("post/mlp/0", "post/mlp/1")
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
    csvwrite(path.stem.replace(".trace", "") + "_activity.csv", rows)
    active = [g for g in rows if g["inside_forward"]]
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


rows = list(csv.DictReader((RAW / "timing.csv").open()))
timing = {
    m: stats([float(r["forward_max_ms"]) for r in rows if r["mode"] == m])
    for m in ("original", "intra", "cross")
}
paired = {}
for a, b in [("original", "intra"), ("original", "cross"), ("intra", "cross")]:
    vals = []
    deltas = []
    for i in range(9):
        values = {
            r["mode"]: float(r["forward_max_ms"]) for r in rows if int(r["round"]) == i
        }
        vals.append(values[a] / values[b])
        deltas.append(values[a] - values[b])
    paired[a + "/" + b] = {"speedup": stats(vals), "saved_ms": stats(deltas)}
write("timing_summary.json", {"forward": timing, "paired": paired})
profiles = {
    p.stem.replace(".trace", ""): trace(p) for p in sorted(RAW.glob("*.trace.json"))
}
write("activity_summary.json", profiles)
print(json.dumps({"forward": timing, "paired": paired}, indent=2))
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
