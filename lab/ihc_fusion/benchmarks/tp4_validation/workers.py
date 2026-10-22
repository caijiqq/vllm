"""Per-instance observation through the existing all-worker apply_model RPC."""

import os

import torch

from lab.ihc_fusion.tests.cases import metrics
from vllm.distributed import get_tp_group


def status(model):
    group = get_tp_group()
    layers = list(model.model.layers)
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
        flags=[
            dict(intra=layer._ihc_fusion_enabled, cross=layer._ihc_defer_post)
            for layer in layers
        ],
        partitions=partition,
    )


def set_mode(model, name):
    for i, layer in enumerate(model.model.layers):
        layer._ihc_fusion_enabled = name != "original"
        layer._ihc_defer_post = name == "cross" and i < 2
    return status(model)


def install_observers(model):
    state = {
        "phase": "none",
        "reference": None,
        "latest": None,
        "shapes": [],
        "handles": [],
    }
    model._tp4_experiment = state

    def observed_pre(module, args):
        if state["phase"] == "capture":
            state["shapes"].append(
                dict(
                    module="hc_pre", shape=list(args[0].shape), dtype=str(args[0].dtype)
                )
            )

    def observed_layer(module, args, kwargs, result):
        if state["phase"] == "capture" and isinstance(result[1], tuple):
            state["shapes"].append(
                dict(
                    module="pending",
                    shape=list(result[1][0].shape),
                    dtype=str(result[1][0].dtype),
                )
            )

    def observed_forward(module, args, kwargs, result):
        if state["phase"] == "capture":
            state["latest"] = result.detach().clone()

    for layer in model.model.layers:
        state["handles"].append(
            layer.register_forward_hook(observed_layer, with_kwargs=True)
        )
        for hc in (layer.hc_attn_layer, layer.hc_mlp_layer):
            state["handles"].append(hc.hc_pre.register_forward_pre_hook(observed_pre))
    state["handles"].append(
        model.register_forward_hook(observed_forward, with_kwargs=True)
    )
    return get_tp_group().rank_in_group


def capture(model):
    state = model._tp4_experiment
    state.update(phase="capture", latest=None, shapes=[])
    torch.manual_seed(7)
    return get_tp_group().rank_in_group


def captured(model, baseline=False):
    state = model._tp4_experiment
    state["phase"] = "none"
    actual = state["latest"]
    assert isinstance(actual, torch.Tensor)
    torch.cuda.synchronize()
    out = dict(
        rank=get_tp_group().rank_in_group,
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        finite=torch.isfinite(actual).all().item(),
        hc_shapes=state["shapes"],
    )
    if baseline:
        state["reference"] = actual
        out["comparison"] = "baseline"
    else:
        reference = state["reference"]
        out["metrics"] = metrics(actual, reference)
        out["cosine"] = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), reference.double().flatten(), dim=0
        ).item()
        # Existing synthetic-chain rule: a diagnostic, not a new threshold.
        try:
            torch.testing.assert_close(
                actual, reference, rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5
            )
            out["existing_chain_rule_pass"] = True
        except AssertionError as e:
            out["existing_chain_rule_pass"] = False
            out["failure"] = str(e)
    state["latest"] = None
    return out
