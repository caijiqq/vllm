# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4-only Post/Pre boundary: metadata routing and rounded residual ownership."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from lab.ihc_fusion.baselines.official import official_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.models.hy_v4.nvidia import model as hy4
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.triton_ihc_post_pre import triton_ihc_post_pre

RECORDS = []


@pytest.fixture(scope="module", autouse=True)
def record_errors():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if path := os.getenv("HY4_FUSION_TEST_REPORT"):
        Path(path).write_text(json.dumps(RECORDS, indent=2))


def context(n, kind="prefill"):
    metadata = MLACommonMetadata(
        num_reqs=1,
        max_query_len=n,
        max_seq_len=n,
        num_actual_tokens=n,
        query_start_loc=torch.tensor([0, n]),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
    )
    if kind in ("mixed", "decode"):
        metadata.num_decodes = 1
        metadata.num_decode_tokens = 1
        metadata.num_prefills = int(kind == "mixed")
    if kind == "padding":
        metadata.num_actual_tokens -= 1
    ctx = ForwardContext(
        no_compile_layers={}, attn_metadata={"attn": metadata}, slot_mapping={}
    )
    if kind == "none":
        return None
    if kind == "unknown":
        ctx.attn_metadata = {}
    if kind == "graph":
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
    if kind == "ubatch":
        ctx.attn_metadata = [ctx.attn_metadata]
    return ctx


@pytest.mark.parametrize("n", [64, 128, 256, 511, 512, 513, 768, 1024, 4096])
def test_kernel_and_updated_input_replay(n):
    args = (*make_case(n, 6144, torch.bfloat16)[:6], 2.0, 1e-6, 1e-5)
    actual = triton_ihc_post_pre(*args)
    baseline = official_post_pre(*args)
    error = check_outputs(actual, baseline)
    check_outputs(actual, torch_post_pre(*args))
    cosine = torch.nn.functional.cosine_similarity(
        actual[0].double().flatten(), baseline[0].double().flatten(), dim=0
    ).item()
    RECORDS.append(dict(n=n, errors=error, cosine=cosine))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_ihc_post_pre(*args)
    args[0].neg_()
    graph.replay()
    check_outputs(captured, official_post_pre(*args))


@pytest.mark.parametrize(
    "n,kind,dtype,enabled,expected",
    [
        (256, "prefill", torch.bfloat16, True, False),
        (511, "prefill", torch.bfloat16, True, False),
        (512, "prefill", torch.bfloat16, True, True),
        (513, "prefill", torch.bfloat16, True, True),
        (8192, "prefill", torch.bfloat16, True, True),
        (8193, "prefill", torch.bfloat16, True, False),
        (512, "prefill", torch.bfloat16, False, False),
        (512, "prefill", torch.float16, True, False),
        (512, "prefill", torch.float32, True, False),
        *[
            (512, kind, torch.bfloat16, True, False)
            for kind in (
                "mixed",
                "decode",
                "unknown",
                "none",
                "padding",
                "graph",
                "ubatch",
                "layout",
                "hidden",
            )
        ],
    ],
)
@pytest.mark.parametrize("pre_attention", [False, True])
def test_boundary_routes_and_preserves_ownership(
    dist_init, monkeypatch, n, kind, dtype, enabled, expected, pre_attention
):
    d = 4096 if kind == "hidden" else 6144
    args = (*make_case(n, d, dtype)[:6], 2.0, 1e-6, 1e-5)
    if kind == "layout":
        x = torch.empty((n, d * 2), device="cuda", dtype=dtype)[:, ::2]
        x.copy_(args[0])
        args = (x, *args[1:])
    config = PretrainedConfig(
        enable_ihc=True,
        hidden_size=d,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    # Test the real boundary and HC modules independently of Attention/MLP.
    layer = hy4.HYV4DecoderLayer.__new__(hy4.HYV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_attn_layer = HYV4HCLayer(config, 0).cuda()
    layer.hc_mlp_layer = HYV4HCLayer(config, 0).cuda()
    layer.self_attn = SimpleNamespace(mla_attn=SimpleNamespace(layer_name="attn"))
    layer._ihc_fusion_enabled = enabled
    pre = (layer.hc_attn_layer if pre_attention else layer.hc_mlp_layer).hc_pre
    with torch.no_grad():
        pre.hc_fn.weight.copy_(args[3])
        pre.hc_scale.copy_(args[4])
        pre.hc_base.copy_(args[5])
    calls = []

    def fused(*a):
        calls.append(True)
        return triton_ihc_post_pre(*a)

    monkeypatch.setattr(hy4, "triton_ihc_post_pre", fused)
    with override_forward_context(context(n, kind)):
        output, post, residual = layer._ihc_post_pre(
            *args[:3], pre_attention=pre_attention
        )
    assert bool(calls) == expected
    check_outputs((output, residual, post), official_post_pre(*args))


class _TestAttention(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.mla_attn = SimpleNamespace(layer_name=name)

    def forward(self, positions, hidden_states):
        return hidden_states.tanh()


@pytest.mark.parametrize("n", [128, 256, 512, 1024, 4096, 8192])
def test_cross_layer_pending_chain(dist_init, monkeypatch, n):
    """Distinct Pre parameters catch gate/weight ownership and missing final Post."""
    torch.manual_seed(7)
    config = PretrainedConfig(
        enable_ihc=True,
        hidden_size=6144,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    layers = []
    for i in range(3):
        layer = hy4.HYV4DecoderLayer.__new__(hy4.HYV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.enable_ihc = True
        layer.hc_attn_layer = HYV4HCLayer(config, i).cuda()
        layer.hc_mlp_layer = HYV4HCLayer(config, i).cuda()
        layer.self_attn = _TestAttention(f"attn{i}")
        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.mlp = torch.nn.SiLU()
        with torch.no_grad():
            for j, hc in enumerate((layer.hc_attn_layer, layer.hc_mlp_layer)):
                hc.hc_pre.hc_fn.weight.normal_(std=0.002 * (1 + i + j))
                hc.hc_pre.hc_base.fill_(0.1 * (i + j))
        layers.append(layer)
    x = make_case(n, 6144, torch.bfloat16)[1]
    positions = torch.arange(n, device="cuda")
    ctx = context(n)
    ctx.attn_metadata = {
        f"attn{i}": next(iter(ctx.attn_metadata.values())) for i in range(3)
    }
    calls = []
    original = hy4.triton_ihc_post_pre

    def fused(*args):
        calls.append(args[3].data_ptr())
        actual = original(*args)
        check_outputs(actual, official_post_pre(*args))
        return actual

    monkeypatch.setattr(hy4, "triton_ihc_post_pre", fused)
    results = []
    with override_forward_context(ctx):
        for mode in ("original", "intra", "cross"):
            hidden, pending = x, None
            calls.clear()
            for i, layer in enumerate(layers):
                layer._ihc_fusion_enabled = mode != "original"
                layer._ihc_defer_post = mode == "cross" and i < 2
                hidden, pending = layer(positions, hidden, pending)
                assert (pending is not None) == (mode == "cross" and n >= 512 and i < 2)
            assert pending is None
            results.append(hidden)
            expected = []
            if n >= 512 and mode != "original":
                for i, layer in enumerate(layers):
                    if mode == "cross" and i:
                        expected.append(
                            layer.hc_attn_layer.hc_pre.hc_fn.weight.data_ptr()
                        )
                    expected.append(layer.hc_mlp_layer.hc_pre.hc_fn.weight.data_ptr())
            assert calls == expected
    # Exact residuals are checked per boundary in the frozen accuracy helper.
    torch.testing.assert_close(
        results[1], results[0], rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5
    )
    torch.testing.assert_close(
        results[2], results[0], rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5
    )


@pytest.mark.parametrize("kind", ["disabled", "mixed", "graph"])
def test_pending_post_falls_back_at_consumer(dist_init, kind):
    """A pending Post must be materialized even if the next layer cannot fuse."""
    args = make_case(512, 6144, torch.bfloat16)
    config = PretrainedConfig(
        enable_ihc=True,
        hidden_size=6144,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    layer = hy4.HYV4DecoderLayer.__new__(hy4.HYV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.enable_ihc = True
    layer._ihc_defer_post = False
    layer.hc_attn_layer = HYV4HCLayer(config, 1).cuda()
    layer.hc_mlp_layer = HYV4HCLayer(config, 1).cuda()
    layer.self_attn = _TestAttention("attn")
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.mlp = torch.nn.SiLU()
    with torch.no_grad():
        for hc in (layer.hc_attn_layer, layer.hc_mlp_layer):
            hc.hc_pre.hc_fn.weight.copy_(args[3])
            hc.hc_pre.hc_scale.copy_(args[4])
            hc.hc_pre.hc_base.copy_(args[5])
    positions = torch.arange(512, device="cuda")
    layer._ihc_fusion_enabled = False
    with override_forward_context(context(512)):
        materialized = layer.hc_mlp_layer.post(*args[:3])
        expected, _ = layer(positions, materialized, None)
    layer._ihc_fusion_enabled = kind != "disabled"
    with override_forward_context(context(512, kind)):
        actual, pending = layer(positions, args[0], (args[1], args[2]))
    assert pending is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
