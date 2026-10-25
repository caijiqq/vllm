# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail before CUDA initialization unless the selected GPUs are idle."""

import csv
import datetime
import io
import json
import os
import subprocess
from pathlib import Path


def check(selected, gpus, processes):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("HY4_TP8_GPUS must contain distinct physical indices")
    by_index = {g["index"]: g for g in gpus}
    errors = []
    for index in selected:
        if index not in by_index:
            errors.append(f"GPU {index}: missing")
            continue
        gpu = by_index[index]
        active = [p for p in processes if p["gpu_uuid"] == gpu["uuid"]]
        if active or int(gpu["memory.used"]) > 512 or int(gpu["utilization.gpu"]) > 0:
            errors.append(
                f"GPU {index}: busy, memory={gpu['memory.used']} MiB, "
                f"utilization={gpu['utilization.gpu']}%, "
                f"PIDs={[p['pid'] for p in active]}"
            )
    return errors


def query(kind, fields):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-{kind}={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(zip(fields, (s.strip() for s in row), strict=True))
        for row in csv.reader(io.StringIO(output))
        if row
    ]


def main():
    selected = os.environ["HY4_TP8_GPUS"].split(",")
    gpus = query(
        "gpu",
        ("index", "uuid", "name", "memory.total", "memory.used", "utilization.gpu"),
    )
    processes = query("compute-apps", ("gpu_uuid", "pid", "used_gpu_memory"))
    errors = check(selected, gpus, processes)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report = dict(
        time=stamp,
        selected=selected,
        gpus=gpus,
        processes=processes,
        errors=errors,
        passed=not errors,
    )
    path = Path(os.environ["HY4_TP8_RUN"]) / "environment" / f"gpu_gate_{stamp}.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit("BLOCKED_GPU_AVAILABILITY: " + "; ".join(errors))


if __name__ == "__main__":
    main()
