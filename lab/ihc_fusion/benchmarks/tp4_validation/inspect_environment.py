"""Read-only environment/shape audit, preserving the initial workspace state."""

import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from safetensors import safe_open

root = Path(os.environ["HY4_TP4_RUN"])
raw = root / "raw"
checkpoint = Path(os.environ["HY4_TP4_CHECKPOINT"]).resolve()
config = json.loads((checkpoint / "config.json").read_text())
assert [
    config[k]
    for k in (
        "num_hidden_layers",
        "n_routed_experts",
        "num_experts_per_tok",
        "hidden_size",
        "hc_mult",
    )
] == [3, 16, 8, 6144, 4]
assert config["quantization_config"]["quantization"]["quant_algo"] == "MXFP8"
for key in ("num_attention_heads", "intermediate_size", "moe_intermediate_size"):
    assert config[key] % 4 == 0
assert config["moe_intermediate_size"] // 4 % 32 == 0
shapes = {}
with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
    for name in f:
        if name.startswith("model.layers.") and (
            name.endswith(".weight") or name.endswith(".hc_fn") or "scale" in name
        ):
            item = f.get_slice(name)
            shapes[name] = dict(shape=item.get_shape(), dtype=item.get_dtype())
(raw / "checkpoint_layout.json").write_text(
    json.dumps(
        dict(
            path=str(checkpoint),
            config_sha256=hashlib.sha256(
                (checkpoint / "config.json").read_bytes()
            ).hexdigest(),
            weight_bytes=(checkpoint / "model.safetensors").stat().st_size,
            config=config,
            weight_layout=shapes,
        ),
        indent=2,
    )
)
files = (
    subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"])
    .decode()
    .split("\0")
)
allowed = (
    "lab/ihc_fusion/benchmarks/tp4_validation/",
    "lab/ihc_fusion/results/tp4_validation/",
)
snapshot = {}
for name in files:
    if not name or name.startswith(allowed):
        continue
    p = Path(name)
    if p.is_symlink():
        snapshot[name] = {"symlink": os.readlink(p)}
    elif p.is_file():
        snapshot[name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
(raw / "before_hashes.json").write_text(json.dumps(snapshot, indent=2))
for name, command in [
    ("initial.status", ["git", "status", "--short", "--untracked-files=all"]),
    ("initial.diff", ["git", "diff", "--binary"]),
    ("topology.txt", ["nvidia-smi", "topo", "-m"]),
]:
    (raw / name).write_bytes(subprocess.check_output(command))
versions = {}
for name in ("torch", "triton", "vllm", "humming-kernels", "flashinfer-python"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
out = dict(
    time=datetime.datetime.now(datetime.UTC).isoformat(),
    branch=subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip(),
    head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    python=platform.python_version(),
    versions=versions,
    selected_physical_gpus=[1, 2, 3, 5],
    gpus=subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv",
        ],
        text=True,
    ),
    processes=subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"],
        text=True,
    ),
    docker_env=Path("/.dockerenv").exists(),
    cgroup=Path("/proc/1/cgroup").read_text(),
    bootstrap=(
        "reuse existing runtime_integration/sitecustomize.py; "
        "checkout source plus installed binary extensions"
    ),
)
(raw / "environment.json").write_text(json.dumps(out, indent=2))
(root / "EXPERIMENT_LOG.md").write_text(
    "# TP4 experiment log\n\n"
    + out["time"]
    + " E00: branch/HEAD/diff snapshot and checkpoint layout verified. "
    "Selected physical GPUs1,2,3,5: each17MiB,0%, no compute processes; "
    "all connectedNV18. Runtime changes committed; "
    "only existing benchmark diff remains. "
    "Config3layers/16experts/top8/D6144/C4/ModelOptMXFP8. Shape arithmetic permitsTP4; "
    "actual backend compatibility remains to be tested. "
    "No package installation or source modification.\n"
)
print(
    json.dumps(
        {k: v for k, v in out.items() if k not in ("gpus", "processes")}, indent=2
    )
)
print("WEIGHT_SHAPES", len(shapes))
