"""Compatibility entrypoint for the shared, CPU-only TP trace analyzer."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tp4_validation/analyze_perf.py"),
        run_name="__main__",
    )
