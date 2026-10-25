# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Instance-local timing and profiling for the isolated TP8 experiment."""

import os
from pathlib import Path

import torch

from vllm.distributed import get_tp_group


def install(model):
    state = {"phase": "off", "records": [], "handles": [], "profile_handles": []}
    model._tp8_perf = state

    def before(module, args, kwargs):
        if state["phase"] != "timing":
            return
        torch.distributed.barrier(group=get_tp_group().device_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        value = kwargs.get("input_ids", args[0] if args else None)
        state["records"].append((start, end, list(value.shape)))
        start.record()

    def after(module, args, kwargs, output):
        if state["phase"] == "timing":
            state["records"][-1][1].record()
            state["output"] = output

    state["handles"] = [
        model.register_forward_pre_hook(before, with_kwargs=True),
        model.register_forward_hook(after, with_kwargs=True),
    ]
    return dict(rank=get_tp_group().rank_in_group, world_size=get_tp_group().world_size)


def arm(model):
    state = model._tp8_perf
    state.update(phase="timing", records=[], output=None)
    return get_tp_group().rank_in_group


def read(model, check_finite=False):
    state = model._tp8_perf
    state["phase"] = "off"
    torch.cuda.synchronize()
    out = {
        "rank": get_tp_group().rank_in_group,
        "world_size": get_tp_group().world_size,
        "records": [
            {"ms": a.elapsed_time(b), "shape": shape}
            for a, b, shape in state["records"]
        ],
    }
    if check_finite:
        out["finite"] = bool(torch.isfinite(state["output"]).all().item())
    state["output"] = None
    return out


def start_profile(model, mode):
    state = model._tp8_perf
    state["phase"] = "profile"
    state["mode"] = mode

    def before(module, args):
        state["scope"] = torch.profiler.record_function("TP8/forward")
        state["scope"].__enter__()

    def after(module, args, result):
        state["scope"].__exit__(None, None, None)

    state["profile_handles"] = [
        model.register_forward_pre_hook(before),
        model.register_forward_hook(after),
    ]
    torch.cuda.synchronize()
    state["profiler"] = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    state["profiler"].start()
    return get_tp_group().rank_in_group


def stop_profile(model):
    state = model._tp8_perf
    torch.cuda.synchronize()
    state["profiler"].stop()
    rank = get_tp_group().rank_in_group
    path = (
        Path(os.environ["HY4_TP8_RUN"])
        / "performance"
        / "raw"
        / f"{state['mode']}_rank{rank}.trace.json"
    )
    state["profiler"].export_chrome_trace(str(path))
    for handle in state["profile_handles"]:
        handle.remove()
    state["profile_handles"] = []
    state["phase"] = "off"
    return {
        "rank": rank,
        "world_size": get_tp_group().world_size,
        "trace": path.name,
    }
