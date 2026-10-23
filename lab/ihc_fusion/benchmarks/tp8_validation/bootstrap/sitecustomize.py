"""Use checkout Python and existing wheel extensions for this TP8 harness only."""

import importlib.metadata
import importlib.util
import os
import sys
import traceback
from pathlib import Path

try:
    import torch  # noqa: F401 -- load libtorch before the compiled extensions

    import vllm

    source = Path(__file__).resolve().parents[5]
    installed = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    assert Path(vllm.__file__).resolve() == source / "vllm/__init__.py"
    assert installed.resolve() != source / "vllm"
    vllm.__path__.append(str(installed))
    for name in ("_vllm_fa2_C", "_vllm_fa3_C"):
        candidates = list((installed / "vllm_flash_attn").glob(name + "*.so"))
        assert len(candidates) == 1, (name, candidates)
        qualified = "vllm.vllm_flash_attn." + name
        spec = importlib.util.spec_from_file_location(qualified, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    import vllm.third_party.flashmla

    vllm.third_party.flashmla.__path__.append(str(installed / "third_party/flashmla"))
    vllm.third_party.__path__.append(str(installed / "third_party"))
except Exception:
    traceback.print_exc()
    os._exit(78)
