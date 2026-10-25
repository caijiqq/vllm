# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Record checkpoint layout and the environment without model initialization."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open

import vllm
from lab.ihc_fusion.benchmarks.tp4_validation.analyze_perf import write


def main():
    checkpoint = Path(os.environ["HY4_TP8_CHECKPOINT"]).resolve()
    config = checkpoint / "config.json"
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors weights in {checkpoint}")
    shapes = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                item = tensors.get_slice(name)
                shapes[name] = dict(shape=item.get_shape(), dtype=item.get_dtype())
    versions = {}
    for name in ("torch", "triton", "vllm", "humming-kernels", "flashinfer-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(
        checkpoint=str(checkpoint),
        config=json.loads(config.read_text()),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        weight_bytes=sum(p.stat().st_size for p in shards),
        weight_layout=shapes,
        python=sys.executable,
        vllm_source=vllm.__file__,
        versions=versions,
        selected_gpus=os.environ["HY4_TP8_GPUS"].split(","),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status=subprocess.check_output(["git", "status", "--short"], text=True),
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
    )
    raw = Path(os.environ["HY4_TP8_RUN"]) / "environment/raw"
    raw.mkdir(parents=True, exist_ok=True)
    write(raw, "environment.json", report)
    print(f"Environment saved to {raw}; {len(shards)} shards, {len(shapes)} tensors")


if __name__ == "__main__":
    main()
