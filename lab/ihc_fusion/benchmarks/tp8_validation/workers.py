# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-instance observation through the existing all-worker apply_model RPC."""

import functools
import os

import torch

from lab.ihc_fusion.tests.cases import metrics
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context


def status(model):
    group = get_tp_group()
    backbone = model.model
    assert get_pp_group().world_size == 1, "Model observations require PP1"
    layers = list(backbone.layers)
    if not hasattr(model, "_ihc_fusion_eligible"):
        model._ihc_fusion_eligible = [layer._ihc_fusion_enabled for layer in layers]
    return dict(
        rank=group.rank_in_group,
        world_size=group.world_size,
        pid=os.getpid(),
        device=torch.cuda.current_device(),
        gpu=torch.cuda.get_device_name(),
        worker_module=__file__,
        fusion_eligible=model._ihc_fusion_eligible,
        flags=[
            dict(intra=layer._ihc_fusion_enabled, cross=layer._ihc_defer_post)
            for layer in layers
        ],
        partitions=[dict(layer=i) for i in range(len(layers))],
    )


def set_mode(model, name):
    backbone = model.model
    for i in range(backbone.start_layer, backbone.end_layer):
        layer = backbone.layers[i]
        layer._ihc_fusion_enabled = name != "original" and model._ihc_fusion_eligible[i]
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
        "capture_id": 0,
        "reference_inputs": None,
    }
    model._tp8_experiment = state

    def observed_inputs(module, args, kwargs):
        if state["phase"] != "capture":
            return
        ctx = get_forward_context()
        metadata = ctx.attn_metadata
        name = model.model.layers[0].self_attn.mla_attn.layer_name
        attn = metadata.get(name) if isinstance(metadata, dict) else None
        table = getattr(attn, "block_table", None)
        if table is None:
            table = getattr(getattr(attn, "prefill", None), "block_table", None)

        def values(tensor):
            return (
                tensor.detach().cpu().tolist()
                if isinstance(tensor, torch.Tensor)
                else None
            )

        state["inputs"].append(
            dict(
                graph=str(ctx.cudagraph_runtime_mode),
                input_ids=values(kwargs.get("input_ids", args[0] if args else None)),
                positions=values(
                    kwargs.get("positions", args[1] if len(args) > 1 else None)
                ),
                query_start_loc=values(getattr(attn, "query_start_loc", None)),
                num_actual_tokens=getattr(attn, "num_actual_tokens", None),
                max_seq_len=getattr(attn, "max_seq_len", None),
                num_prefills=getattr(attn, "num_prefills", None),
                num_decodes=getattr(attn, "num_decodes", None),
                seq_lens=values(getattr(attn, "seq_lens", None)),
                slot_mapping=values(getattr(attn, "slot_mapping", None)),
                block_table=values(table),
            )
        )

    state["handles"].append(
        model.register_forward_pre_hook(observed_inputs, with_kwargs=True)
    )

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
            state["snapshot_id"] = state["capture_id"]

    def observed_boundary(module, args, layer_index):
        if state["phase"] == "capture":
            state["boundaries"][f"layer{layer_index}_input_norm"] = (
                args[0].detach().clone()
            )

    def observed_head(module, args):
        if state["phase"] == "capture":
            state["boundaries"][f"layer{len(model.model.layers) - 1}_final_post"] = (
                args[0].detach().clone()
            )

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
                    functools.partial(observed_pre, layer_index=layer_index, kind=kind)
                )
            )
        if layer_index in (1, len(model.model.layers) - 1):
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


def capture(model, expected_ids=None):
    state = model._tp8_experiment
    state.update(
        phase="capture",
        latest=None,
        shapes=[],
        layers=[],
        boundaries={},
        inputs=[],
        snapshot_id=None,
        expected_ids=expected_ids,
        capture_id=state["capture_id"] + 1,
    )
    torch.manual_seed(7)
    return get_tp_group().rank_in_group


def compare(actual, reference=None):
    result = dict(
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        device=str(actual.device),
        valid=True,
        finite=bool(torch.isfinite(actual).all().item()),
    )
    if reference is not None:
        result.update(
            shape_match=actual.shape == reference.shape,
            dtype_match=actual.dtype == reference.dtype,
        )
        if (
            not result["shape_match"]
            or not result["dtype_match"]
            or actual.device != reference.device
        ):
            return dict(result, valid=False, status="INVALID", **{"pass": False})
        if result["shape_match"] and result["dtype_match"]:
            result["metrics"] = metrics(actual, reference)
            result["cosine"] = torch.nn.functional.cosine_similarity(
                actual.double().flatten(), reference.double().flatten(), dim=0
            ).item()
            result["bitwise_equal"] = torch.equal(
                actual.contiguous().view(torch.uint8),
                reference.contiguous().view(torch.uint8),
            )
            result["unequal_elements"] = int(torch.count_nonzero(actual != reference))
            if (
                actual.dtype == torch.bfloat16
                and result["finite"]
                and torch.isfinite(reference).all()
            ):
                index = (actual.float() - reference.float()).abs().flatten().argmax()
                bits = [
                    int(t.flatten()[index].view(torch.int16))
                    for t in (actual, reference)
                ]
                ordered = [32768 - (b & 32767) if b < 0 else 32768 + b for b in bits]
                result["ulp_at_max_abs"] = abs(ordered[0] - ordered[1])
        try:
            # Reuse the existing TP4/synthetic-chain rule without changing tolerances.
            torch.testing.assert_close(
                actual, reference, rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5
            )
            result["existing_chain_rule_pass"] = True
        except AssertionError as error:
            result.update(existing_chain_rule_pass=False, failure=str(error))
    result["pass"] = result["finite"] and result.get("existing_chain_rule_pass", True)
    result["status"] = "PASS" if result["pass"] else "FAIL"
    return result


def captured(model, baseline=False):
    state = model._tp8_experiment
    state["phase"] = "none"
    actual = state["latest"]
    inputs = state["inputs"]
    record = inputs[0] if len(inputs) == 1 else {}
    expected = (
        state["expected_ids"]
        if state["expected_ids"] is not None
        else record.get("input_ids")
    )
    n = len(expected) if expected is not None else 0
    logical = {
        k: record.get(k)
        for k in (
            "input_ids",
            "positions",
            "query_start_loc",
            "num_actual_tokens",
            "max_seq_len",
        )
    }
    valid = (
        n > 0
        and isinstance(actual, torch.Tensor)
        and actual.shape[0] == n
        and state["snapshot_id"] == state["capture_id"]
        and record.get("graph") == "NONE"
        and logical
        == dict(
            input_ids=expected,
            positions=list(range(n)),
            query_start_loc=[0, n],
            num_actual_tokens=n,
            max_seq_len=n,
        )
    )
    if baseline:
        state["reference"] = None
    info = dict(
        rank=get_tp_group().rank_in_group,
        world_size=get_tp_group().world_size,
        capture_id=state["capture_id"],
        snapshot_id=state["snapshot_id"],
        forward_calls=len(inputs),
        inputs=inputs,
    )
    if not valid or (
        not baseline
        and (state["reference"] is None or logical != state["reference_inputs"])
    ):
        state["latest"] = None
        return dict(
            info,
            valid=False,
            status="INVALID",
            reason="Expected one fresh Eager full-prompt Forward and matching inputs",
            **{"pass": False},
        )
    torch.cuda.synchronize()
    count = len(model.model.layers)
    if baseline:
        state["reference"] = actual
        state["reference_inputs"] = logical
        state["boundary_reference"] = state["boundaries"].copy()
    out = compare(actual, None if baseline else state["reference"])
    out.update(info)
    out.update(
        rank=get_tp_group().rank_in_group,
        world_size=get_tp_group().world_size,
        hc_shapes=state["shapes"],
    )
    eligible = (
        512 <= actual.shape[0] <= 8192
        and actual.shape[1:] == (6144,)
        and actual.dtype == torch.bfloat16
        and all(model._ihc_fusion_eligible)
    )
    expected_pending = (
        list(range(count - 1)) if state["mode"] == "cross" and eligible else []
    )
    observed_pending = sorted(
        item["layer"] for item in state["shapes"] if item["module"] == "pending"
    )
    final_post = f"layer{count - 1}_final_post"
    out["path"] = dict(
        mode=state["mode"],
        eligible_shape=eligible,
        observed_layers=sorted(set(state["layers"])),
        observed_pending_layers=observed_pending,
        expected_pending_layers=expected_pending,
        first_pre_observed=any(
            item["layer"] == 0 and item.get("kind") == "attention"
            for item in state["shapes"]
        ),
        last_post_observed=final_post in state["boundaries"],
    )
    out["path"]["pass"] = (
        out["path"]["observed_layers"] == list(range(count))
        and observed_pending == expected_pending
        and out["path"]["first_pre_observed"]
        and out["path"]["last_post_observed"]
    )
    out["boundaries"] = {
        name: compare(
            value, None if baseline else state["boundary_reference"].get(name)
        )
        for name, value in state["boundaries"].items()
    }
    expected = {f"layer{i}_input_norm" for i in (1, count - 1) if 0 <= i < count}
    expected.add(final_post)
    out["boundaries_pass"] = set(out["boundaries"]) == expected and all(
        item["pass"] for item in out["boundaries"].values()
    )
    out["pass"] &= out["path"]["pass"] and out["boundaries_pass"]
    out["valid"] &= (
        (baseline or set(state["boundary_reference"]) == expected)
        and out["path"]["pass"]
        and set(out["boundaries"]) == expected
        and all(b["valid"] for b in out["boundaries"].values())
    )
    if baseline and not out["valid"]:
        state["reference"] = None
    out["pass"] &= out["valid"]
    out["status"] = "INVALID" if not out["valid"] else "PASS" if out["pass"] else "FAIL"
    state["latest"] = None
    return out
