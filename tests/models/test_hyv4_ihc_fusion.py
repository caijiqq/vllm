# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4-only Post/Pre boundary: metadata routing and rounded residual ownership."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedConfig

from lab.ihc_fusion.baselines.official import official_post_pre
from lab.ihc_fusion.reference.torch_post_pre import torch_post_pre
from lab.ihc_fusion.tests.cases import check_outputs, make_case, metrics
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.models.hy_v4.nvidia import model as hy4
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.triton_ihc_post_pre import triton_ihc_post_pre

RECORDS: list[dict[str, object]] = []


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
    if kind in ("graph", "graph_ubatch"):
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
    if kind == "graph_ubatch":
        ctx.ubatch_slices = []
    if kind == "compiled_graph":
        ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
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
        (512, "graph", torch.bfloat16, True, True),
        (511, "graph", torch.bfloat16, True, False),
        (8193, "graph", torch.bfloat16, True, False),
        (512, "graph", torch.float16, True, False),
        (512, "graph_ubatch", torch.bfloat16, True, False),
        (512, "compiled_graph", torch.bfloat16, True, False),
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
    if kind == "compiled_graph":
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    d = 4096 if kind == "hidden" else 6144
    args = (*make_case(n, d, dtype)[:6], 2.0, 1e-6, 1e-5)
    if kind == "layout":
        x = torch.empty((n, d * 2), device="cuda", dtype=dtype)[:, ::2]
        x.copy_(args[0])
        args = (x, *args[1:])
    config = PreTrainedConfig(
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


def make_layers():
    torch.manual_seed(7)
    config = PreTrainedConfig(
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
    return layers


@pytest.mark.parametrize("n", [128, 256, 512, 1024, 4096, 8192])
def test_cross_layer_pending_chain(dist_init, monkeypatch, n):
    """Distinct Pre parameters catch gate/weight ownership and missing final Post."""
    layers = make_layers()
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


@pytest.mark.parametrize("kind", ["disabled", "mixed", "padding"])
def test_pending_post_falls_back_at_consumer(dist_init, kind):
    """A pending Post must be materialized even if the next layer cannot fuse."""
    args = make_case(512, 6144, torch.bfloat16)
    config = PreTrainedConfig(
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


@pytest.mark.parametrize(
    "name,capability,expected",
    [
        ("NVIDIA H100 80GB HBM3", (9, 0), True),
        ("NVIDIA H200", (9, 0), True),
        ("NVIDIA A100", (8, 0), False),
        ("NVIDIA B200", (10, 0), False),
        ("unknown", None, False),
    ],
)
def test_architecture_dispatch(dist_init, monkeypatch, name, capability, expected):
    """Exercise constructor and dispatch without loading Attention/MLP weights."""
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(hy4.current_platform, "get_device_name", lambda: name)
    monkeypatch.setattr(
        type(hy4.current_platform),
        "get_device_capability",
        classmethod(
            lambda cls, device_id=0: (
                DeviceCapability(*capability) if capability else None
            )
        ),
    )
    monkeypatch.setattr(hy4, "HYV4MLAAttention", lambda **kw: _TestAttention("attn"))
    monkeypatch.setattr(hy4, "HYV4FeedForward", lambda **kw: torch.nn.Identity())
    config = PreTrainedConfig(
        enable_ihc=True,
        enable_ihc_post_pre_fusion=True,
        hidden_size=6144,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        num_attention_heads=1,
        qk_nope_head_dim=1,
        qk_rope_head_dim=2,
        v_head_dim=1,
        q_lora_rank=None,
        kv_lora_rank=1,
        mlp_layer_types=["dense"],
        intermediate_size=16,
        hidden_act="silu",
    )
    layer = hy4.HYV4DecoderLayer(
        config,
        SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False), speculative_config=None
        ),
        prefix="layers.0",
    ).cuda()
    assert layer._ihc_fusion_enabled == expected
    args = make_case(512, 6144, torch.bfloat16)
    pre = layer.hc_mlp_layer.hc_pre
    with torch.no_grad():
        pre.hc_fn.weight.copy_(args[3])
        pre.hc_scale.copy_(args[4])
        pre.hc_base.copy_(args[5])
    calls = []

    def fused(*inputs):
        calls.append(True)
        return triton_ihc_post_pre(*inputs)

    monkeypatch.setattr(hy4, "triton_ihc_post_pre", fused)
    with override_forward_context(context(512, "graph")):
        output, post, residual = layer._ihc_post_pre(*args[:3])
    assert bool(calls) == expected
    check_outputs(
        (output, residual, post), official_post_pre(*args[:6], 2.0, 1e-6, 1e-5)
    )


@pytest.mark.parametrize("n", [128, 512, 1024])
@pytest.mark.parametrize("graph_kind", ["full", "breakable"])
@pytest.mark.parametrize("boundary", ["chain", "consumer_off", "pp"])
@torch.inference_mode()
def test_graph_chain_replay(dist_init, monkeypatch, n, graph_kind, boundary):
    """Replay new inputs through real wrappers, including pending across breaks."""
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        BreakableCUDAGraphWrapper,
    )
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import VllmConfig
    from vllm.forward_context import BatchDescriptor

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    layers = make_layers()
    positions = torch.arange(n, device="cuda")
    x = make_case(n, 6144, torch.bfloat16)[1]
    mode = CUDAGraphMode.FULL if graph_kind == "full" else CUDAGraphMode.PIECEWISE
    ctx = context(n)
    ctx.cudagraph_runtime_mode = mode
    ctx.batch_descriptor = BatchDescriptor(num_tokens=n)
    ctx.attn_metadata = None  # Real piecewise warmup/capture has no metadata.
    calls = []

    def fused(*args):
        calls.append(args[3].data_ptr())
        return triton_ihc_post_pre(*args)

    monkeypatch.setattr(hy4, "triton_ihc_post_pre", fused)

    if graph_kind == "breakable":

        def attention(positions, hidden_states):
            out = torch.empty_like(hidden_states)

            def run():
                torch.tanh(hidden_states, out=out)

            cap = BreakableCUDAGraphCapture.current()
            if cap is not None:
                cap.add_eager(run)
            else:
                run()
            return out

        for layer in layers:
            monkeypatch.setattr(layer.self_attn, "forward", attention)

    def set_mode(enabled):
        for i, layer in enumerate(layers):
            layer._ihc_fusion_enabled = enabled and not (
                boundary == "consumer_off" and i == 1
            )
            layer._ihc_defer_post = (
                enabled and i < 2 and not (boundary == "pp" and i == 0)
            )

    def forward(value):
        hidden, pending = value, None
        for i, layer in enumerate(layers):
            if boundary == "pp" and i == 1:
                assert pending is None
                hidden = hidden.flatten(1)  # PP receive contract.
            hidden, pending = layer(positions, hidden, pending)
        assert pending is None
        return hidden

    wrappers = []
    counts = []
    numerical_failures = []
    with override_forward_context(ctx):
        for enabled in (False, True):
            set_mode(enabled)
            for _ in range(3):
                forward(x)
            wrapper_cls = (
                CUDAGraphWrapper if graph_kind == "full" else BreakableCUDAGraphWrapper
            )
            wrapper = wrapper_cls(forward, VllmConfig(), runtime_mode=mode)
            wrapper.graph_pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            calls.clear()
            with torch.cuda.stream(stream):
                wrapper(x)
            torch.cuda.current_stream().wait_stream(stream)
            counts.append(len(calls))
            wrappers.append(wrapper)
        expected = (
            (5 if boundary == "chain" else 4 if boundary == "pp" else 2)
            if n >= 512
            else 0
        )
        assert counts == [0, expected]
        for step, kind in enumerate(("prefill", "mixed", "padding", "decode")):
            x.copy_(torch.roll(x, 1, 0).neg() * (0.9 + 0.01 * step))
            metadata = context(n, kind).attn_metadata["attn"]
            ctx.attn_metadata = {f"attn{i}": metadata for i in range(3)}
            before = len(calls)
            off = wrappers[0](x).clone()
            on = wrappers[1](x).clone()
            assert len(calls) == before  # Replay must not re-enter Python dispatch.
            torch.testing.assert_close(wrappers[1](x), on, rtol=0, atol=0)
            set_mode(False)
            torch.testing.assert_close(off, forward(x), rtol=0, atol=0)
            set_mode(True)
            eager = forward(x)
            torch.testing.assert_close(on, eager, rtol=0, atol=0)
            if kind == "prefill":
                ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
                torch.testing.assert_close(on, forward(x), rtol=0, atol=0)
                ctx.cudagraph_runtime_mode = mode
            strict_pass = True
            try:
                torch.testing.assert_close(
                    on, off, rtol=2 * torch.finfo(torch.bfloat16).eps, atol=2e-5
                )
            except AssertionError as error:
                strict_pass = False
                numerical_failures.append(str(error))
            RECORDS.append(
                dict(
                    n=n,
                    graph=graph_kind,
                    boundary=boundary,
                    batch=kind,
                    strict_pass=strict_pass,
                    graph_eager_exact=True,
                    errors=metrics(on, off),
                    cosine=torch.nn.functional.cosine_similarity(
                        on.double().flatten(), off.double().flatten(), dim=0
                    ).item(),
                )
            )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            wrappers[1](x)
            torch.accelerator.synchronize()
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert sum("_fused_post_pre" in name for name in names) == expected, names
        for kernel in ("_ihc_pre_stage1", "_ihc_pre_stage2", "_ihc_post_kernel"):
            assert sum(kernel in name for name in names) == 6 - expected, names
        if path := os.getenv("HY4_FUSION_TRACE_DIR"):
            prof.export_chrome_trace(
                str(Path(path) / f"{graph_kind}_{boundary}_{n}.json")
            )

        assert not numerical_failures, numerical_failures
