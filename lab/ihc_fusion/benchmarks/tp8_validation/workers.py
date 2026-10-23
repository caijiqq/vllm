"""Per-instance observation through the existing all-worker apply_model RPC."""

import functools
import os

import torch

from lab.ihc_fusion.tests.cases import metrics
from vllm.distributed import get_pp_group, get_tp_group


def status(model):
    group = get_tp_group()
    backbone = model.model
    assert get_pp_group().world_size == 1 and (
        backbone.start_layer == 0
        and backbone.end_layer == backbone.config.num_hidden_layers == 3
        and len(backbone.layers) == 3
    ), "This eager harness requires PP1 and exactly three local iHC layers"
    layers = list(backbone.layers)
    partition = []
    for i, layer in enumerate(layers):
        linears = {}
        for name, module in layer.named_modules():
            if hasattr(module, "input_size_per_partition") or hasattr(
                module, "output_size_per_partition"
            ):
                linears[name] = {
                    key: getattr(module, key, None)
                    for key in (
                        "input_size_per_partition",
                        "output_size_per_partition",
                        "tp_size",
                    )
                }
                method = getattr(module, "quant_method", None)
                linears[name]["quant_method"] = type(method).__name__
                linears[name]["kernel"] = type(getattr(method, "kernel", None)).__name__
        partition.append(
            dict(
                layer=i,
                local_heads=layer.self_attn.num_local_heads,
                hc_weight=list(layer.hc_attn_layer.hc_pre.hc_fn.weight.shape),
                mlp_class=type(layer.mlp).__name__,
                mlp_tp=getattr(layer.mlp, "tp_size", None),
                mlp_ep=getattr(layer.mlp, "ep_size", None),
                linears=linears,
            )
        )
    return dict(
        rank=group.rank_in_group,
        world_size=group.world_size,
        pid=os.getpid(),
        device=torch.cuda.current_device(),
        gpu=torch.cuda.get_device_name(),
        worker_module=__file__,
        flags=[
            dict(intra=layer._ihc_fusion_enabled, cross=layer._ihc_defer_post)
            for layer in layers
        ],
        partitions=partition,
    )


def set_mode(model, name):
    backbone = model.model
    for i in range(backbone.start_layer, backbone.end_layer):
        layer = backbone.layers[i]
        layer._ihc_fusion_enabled = name != "original"
        layer._ihc_defer_post = name == "cross" and i < backbone.end_layer - 1
    if hasattr(model, "_tp8_experiment"):
        model._tp8_experiment["mode"] = name
    return status(model)


def install_observers(model):
    state = {
        "phase": "none",
        "mode": None,
        "reference": None,
        "latest": None,
        "shapes": [],
        "layers": [],
        "boundaries": {},
        "boundary_reference": {},
        "handles": [],
    }
    model._tp8_experiment = state

    def observed_pre(module, args, layer_index, kind):
        if state["phase"] == "capture":
            state["shapes"].append(
                dict(
                    module="hc_pre",
                    shape=list(args[0].shape),
                    dtype=str(args[0].dtype),
                    layer=layer_index,
                    kind=kind,
                )
            )

    def observed_layer(module, args, kwargs, result, layer_index):
        pending = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if state["phase"] == "capture" and isinstance(pending, tuple):
            state["shapes"].append(
                dict(
                    module="pending",
                    shape=list(pending[0].shape),
                    dtype=str(pending[0].dtype),
                    layer=layer_index,
                )
            )
        if state["phase"] == "capture":
            state["layers"].append(layer_index)

    def observed_forward(module, args, kwargs, result):
        if state["phase"] == "capture":
            state["latest"] = result.detach().clone()

    def observed_boundary(module, args, layer_index):
        if state["phase"] == "capture":
            state["boundaries"][f"layer{layer_index}_input_norm"] = (
                args[0].detach().clone()
            )

    def observed_head(module, args):
        if state["phase"] == "capture":
            state["boundaries"]["layer2_final_post"] = args[0].detach().clone()

    for layer_index, layer in enumerate(model.model.layers):
        state["handles"].append(
            layer.register_forward_hook(
                functools.partial(observed_layer, layer_index=layer_index),
                with_kwargs=True,
            )
        )
        for kind, hc in (
            ("attention", layer.hc_attn_layer),
            ("mlp", layer.hc_mlp_layer),
        ):
            state["handles"].append(
                hc.hc_pre.register_forward_pre_hook(
                    functools.partial(
                        observed_pre, layer_index=layer_index, kind=kind
                    )
                )
            )
        if layer_index in (1, 2):
            state["handles"].append(
                layer.input_layernorm.register_forward_pre_hook(
                    functools.partial(observed_boundary, layer_index=layer_index)
                )
            )
    state["handles"].append(
        model.model.hc_head.register_forward_pre_hook(observed_head)
    )
    state["handles"].append(
        model.register_forward_hook(observed_forward, with_kwargs=True)
    )
    return get_tp_group().rank_in_group


def capture(model):
    state = model._tp8_experiment
    state.update(phase="capture", latest=None, shapes=[], layers=[], boundaries={})
    torch.manual_seed(7)
    return get_tp_group().rank_in_group


def captured(model, baseline=False):
    state = model._tp8_experiment
    state["phase"] = "none"
    actual = state["latest"]
    assert isinstance(actual, torch.Tensor)
    torch.cuda.synchronize()
    out = dict(
        rank=get_tp_group().rank_in_group,
        world_size=get_tp_group().world_size,
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        finite=torch.isfinite(actual).all().item(),
        hc_shapes=state["shapes"],
    )
    n = actual.shape[0]
    eligible = 512 <= n <= 8192 and actual.shape[1:] == (6144,)
    expected_pending = (
        list(range(2)) if state["mode"] == "cross" and eligible else []
    )
    observed_pending = sorted(
        item["layer"] for item in state["shapes"] if item["module"] == "pending"
    )
    out["path"] = {
        "mode": state["mode"],
        "eligible_shape": eligible,
        "observed_layers": sorted(set(state["layers"])),
        "observed_pending_layers": observed_pending,
        "expected_pending_layers": expected_pending,
        "first_pre_observed": any(
            item["layer"] == 0 and item.get("kind") == "attention"
            for item in state["shapes"]
        ),
        "last_post_observed": "layer2_final_post" in state["boundaries"],
        "pass": sorted(set(state["layers"])) == list(range(3))
        and observed_pending == expected_pending,
        "kernel_hit": "unverified",
    }
    out["path"]["pass"] = (
        out["path"]["pass"]
        and out["path"]["first_pre_observed"]
        and out["path"]["last_post_observed"]
    )
    if baseline:
        state["reference"] = actual
        state["boundary_reference"] = {
            name: value.detach().clone()
            for name, value in state["boundaries"].items()
        }
        out.update(comparison="baseline", shape_match=True, dtype_match=True)
    else:
        reference = state["reference"]
        out["shape_match"] = tuple(actual.shape) == tuple(reference.shape)
        out["dtype_match"] = actual.dtype == reference.dtype
        if out["shape_match"] and out["dtype_match"]:
            out["metrics"] = metrics(actual, reference)
            out["cosine"] = torch.nn.functional.cosine_similarity(
                actual.double().flatten(), reference.double().flatten(), dim=0
            ).item()
    boundaries = {}
    for name, value in state["boundaries"].items():
        item = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item()),
        }
        expected = state["boundary_reference"].get(name)
        if expected is None:
            item["comparison"] = "baseline"
        else:
            item["shape_match"] = tuple(value.shape) == tuple(expected.shape)
            item["dtype_match"] = value.dtype == expected.dtype
            if item["shape_match"] and item["dtype_match"]:
                item["metrics"] = metrics(value, expected)
                item["cosine"] = torch.nn.functional.cosine_similarity(
                    value.double().flatten(), expected.double().flatten(), dim=0
                ).item()
        item["pass"] = (
            item["finite"]
            and item.get("shape_match", True)
            and item.get("dtype_match", True)
        )
        boundaries[name] = item
    out["boundaries"] = boundaries
    out["boundaries_pass"] = set(boundaries) == {
        "layer1_input_norm",
        "layer2_input_norm",
        "layer2_final_post",
    } and all(item["pass"] for item in boundaries.values())
    state["latest"] = None
    return out
