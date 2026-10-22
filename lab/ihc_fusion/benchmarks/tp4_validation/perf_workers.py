"""Instance-local timing and profiling for the isolated TP4 experiment."""

import os
from pathlib import Path

import torch

from vllm.distributed import get_tp_group


def install(model):
    state = {"phase": "off", "records": [], "handles": [], "profile_handles": []}
    model._tp4_perf = state

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
    parallel = []
    for i, layer in enumerate(model.model.layers):
        for name, module in layer.mlp.named_modules():
            cfg = getattr(module, "moe_config", None)
            pcfg = getattr(cfg, "moe_parallel_config", None)
            if pcfg is not None:
                parallel.append(
                    {
                        "layer": i,
                        "module": name,
                        "parallel": repr(pcfg),
                        "config": repr(cfg),
                    }
                )
    return {"rank": get_tp_group().rank_in_group, "moe": parallel}


def arm(model):
    state = model._tp4_perf
    state.update(phase="timing", records=[], output=None)
    return get_tp_group().rank_in_group


def read(model, check_finite=False):
    state = model._tp4_perf
    state["phase"] = "off"
    torch.cuda.synchronize()
    out = {
        "rank": get_tp_group().rank_in_group,
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
    state = model._tp4_perf
    state["phase"] = "profile"
    state["observations"] = []
    state["mode"] = mode

    def attach(module, tag):
        stack = []

        def before(mod, args, kwargs):
            scope = torch.profiler.record_function("TP4/" + tag)
            scope.__enter__()
            stack.append(scope)
            if tag.startswith("pre/") and args:
                state["observations"].append({"tag": tag, "shape": list(args[0].shape)})

        def after(mod, args, kwargs, output):
            if tag.startswith("layer/") and isinstance(output, tuple):
                pending = output[1]
                if isinstance(pending, tuple):
                    state["observations"].append(
                        {"tag": tag + "/pending", "shape": list(pending[0].shape)}
                    )
            stack.pop().__exit__(None, None, None)

        state["profile_handles"].extend(
            [
                module.register_forward_pre_hook(before, with_kwargs=True),
                module.register_forward_hook(after, with_kwargs=True),
            ]
        )

    attach(model, "forward")
    for i, layer in enumerate(model.model.layers):
        attach(layer, f"layer/{i}")
        attach(layer.self_attn, f"attention/{i}")
        attach(layer.mlp, f"{'dense' if i == 0 else 'moe'}/{i}")
        for name, module in layer.mlp.named_modules():
            if name == "shared_experts":
                attach(module, f"shared_moe/{i}")
            elif name == "experts.routed_experts":
                attach(module, f"routed_moe/{i}")
        for kind, hc in [
            ("attention", layer.hc_attn_layer),
            ("mlp", layer.hc_mlp_layer),
        ]:
            attach(hc.hc_pre, f"pre/{kind}/{i}")
            attach(hc.hc_post, f"post/{kind}/{i}")
    attach(model.model.hc_head, "head")
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
    state = model._tp4_perf
    torch.cuda.synchronize()
    state["profiler"].stop()
    rank = get_tp_group().rank_in_group
    path = (
        Path(os.environ["HY4_TP4_RUN"])
        / "raw"
        / f"{state['mode']}_rank{rank}.trace.json"
    )
    state["profiler"].export_chrome_trace(str(path))
    for handle in state["profile_handles"]:
        handle.remove()
    state["profile_handles"] = []
    state["phase"] = "off"
    return {"rank": rank, "trace": path.name, "observations": state["observations"]}
