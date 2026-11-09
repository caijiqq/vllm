# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recheck archived paired Serving results without a GPU or vLLM imports."""

import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
recorded = json.loads((root / "paired_analysis.json").read_text())
summary = {}
requests = 0
for length in (8192, 16384, 32768):
    pairs = []
    for round_id in range(9):
        pair = {}
        for mode in ("original", "fused"):
            path = root / f"r{round_id}" / mode / f"n{length}.json"
            data = json.loads(path.read_text())
            assert data["completed"] == 64 and data["failed"] == 0, path
            assert data["total_input_tokens"] == 64 * length, path
            assert data["total_output_tokens"] == 64 * 512, path
            requests += data["completed"]
            pair[mode] = data
        pairs.append(pair)
    summary[str(length)] = {}
    for metric, old in recorded["summary"][str(length)].items():
        changes = []
        for pair in pairs:
            baseline, fused = pair["original"][metric], pair["fused"][metric]
            assert baseline > 0 and math.isfinite(fused)
            changes.append((fused / baseline - 1) * 100)
        median = statistics.median(changes)
        higher_better = "throughput" in metric
        wins = sum(value > 0 if higher_better else value < 0 for value in changes)
        assert math.isclose(median, old["median_change_pct"], abs_tol=1e-9), metric
        assert wins == old["wins"], metric
        assert all(
            math.isclose(a, b, abs_tol=1e-9)
            for a, b in zip(changes, old["paired_change_pct"], strict=True)
        ), metric
        summary[str(length)][metric] = {"median_change_pct": median, "wins": wins}
assert requests == 3456
print(
    json.dumps(
        {
            "files": 54,
            "completed_requests": requests,
            "matches_archived_analysis": True,
            "summary": summary,
        },
        indent=2,
    )
)
