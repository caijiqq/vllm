# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import PreTrainedConfig

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.hy_v4.nvidia import model as hy4
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_post, triton_ihc_pre
from vllm.models.hy_v4.nvidia.triton_ihc_post_pre import triton_ihc_post_pre


def metric(actual, expected):
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    return {
        "max_abs": (actual - expected).abs().max().item(),
        "cosine": F.cosine_similarity(actual[None], expected[None]).item(),
    }


def config():
    return PreTrainedConfig(
        enable_ihc=True,
        hidden_size=6144,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )


def operator_case(tokens, layer):
    pre = layer.hc_pre
    hidden = pre.hidden_dim
    channels = pre.hc_mult
    with torch.no_grad():
        pre.hc_fn.weight.normal_(std=0.002)
        pre.hc_scale.fill_(0.01)
        pre.hc_base.zero_()
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    x = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    residual = torch.randn(
        (tokens, channels, hidden),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    previous = torch.randn(
        (tokens, channels), device="cuda", dtype=torch.float32, generator=generator
    )
    rms_weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
    return (
        x,
        residual,
        previous,
        pre.hc_fn.weight,
        pre.hc_scale,
        pre.hc_base,
        pre.magnitude,
        pre.hc_eps,
        pre.layernorm_epsilon,
        rms_weight,
        1e-5,
    )


def reference_post_pre(args):
    updated = triton_ihc_post(args[0], args[1], args[2])
    output, post = triton_ihc_pre(updated, *args[3:9])
    output = (
        output
        * torch.rsqrt(output.float().square().mean(-1, keepdim=True) + args[10])
        * args[9]
    )
    return output.to(args[0].dtype), updated, post


@pytest.mark.parametrize("tokens", (8192, 16384, 32768))
@torch.inference_mode()
def test_fused_operator_metrics(dist_init, tokens):
    layer = HYV4HCLayer(config(), 0).cuda()
    args = operator_case(tokens, layer)
    fused = triton_ihc_post_pre(*args)
    reference = reference_post_pre(args)
    result = {
        name: metric(actual, expected)
        for name, actual, expected in zip(
            ("output", "updated", "post"), fused, reference
        )
    }
    print(json.dumps({"tokens": tokens, "operator": result}))


class DummyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mla_attn = SimpleNamespace(layer_name="attn")

    def forward(self, positions, hidden_states):
        return hidden_states.tanh()


def layers():
    result = []
    for index in range(3):
        layer = hy4.HYV4DecoderLayer.__new__(hy4.HYV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.enable_ihc = True
        layer.hc_attn_layer = HYV4HCLayer(config(), index).cuda()
        layer.hc_mlp_layer = HYV4HCLayer(config(), index).cuda()
        layer.self_attn = DummyAttention().cuda()
        layer.input_layernorm = RMSNorm(
            config().hidden_size, config().rms_norm_eps
        ).cuda()
        layer.post_attention_layernorm = RMSNorm(
            config().hidden_size, config().rms_norm_eps
        ).cuda()
        layer.mlp = torch.nn.SiLU().cuda()
        with torch.no_grad():
            for hc in (layer.hc_attn_layer, layer.hc_mlp_layer):
                hc.hc_pre.hc_fn.weight.normal_(std=0.002)
                hc.hc_pre.hc_scale.fill_(0.01)
                hc.hc_pre.hc_base.zero_()
        result.append(layer)
    return result


def run_layers(model_layers, hidden, fused):
    positions = torch.arange(hidden.shape[0], device="cuda")
    states = []
    pending = None
    for index, layer in enumerate(model_layers):
        layer._ihc_fusion_enabled = fused
        layer._ihc_defer_post = fused and index < len(model_layers) - 1
        hidden, pending = layer(positions, hidden, pending)
        state = (
            layer.hc_mlp_layer.post(hidden, *pending) if pending is not None else hidden
        )
        states.append(state.clone())
    return states


@torch.inference_mode()
def test_fused_model_hidden_metrics(dist_init):
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    hidden = torch.randn(
        (8192, config().hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    original = torch.nn.ModuleList(layers())
    fused = torch.nn.ModuleList(layers())
    fused.load_state_dict(original.state_dict())
    original_states = run_layers(original, hidden, False)
    fused_states = run_layers(fused, hidden, True)
    result = {
        f"layer_{index}": metric(actual, expected)
        for index, (actual, expected) in enumerate(zip(fused_states, original_states))
    }
    print(json.dumps({"hidden": result}))


class HY4HiddenProbe:
    def _layers(self):
        return sorted(
            (
                layer
                for layer in self.model_runner.model.modules()
                if isinstance(layer, hy4.HYV4DecoderLayer)
            ),
            key=lambda layer: layer.layer_idx,
        )

    def capture_hidden(self):
        self.hidden = {}

        def capture(layer, args, kwargs, output):
            hidden, pending = output
            if pending is not None:
                hidden = layer.hc_mlp_layer.post(hidden, *pending)
            self.hidden[layer.layer_idx] = hidden[-1:].detach().float().cpu()

        for layer in self._layers():
            layer.register_forward_hook(capture, with_kwargs=True)

    def get_hidden(self):
        return self.hidden


def _full_hy4_hidden(vllm_runner, checkpoint, fused, tp, tokens):
    from vllm import SamplingParams, TokensPrompt

    with vllm_runner(
        checkpoint,
        tensor_parallel_size=tp,
        hf_overrides={"enable_ihc_post_pre_fusion": fused},
        worker_extension_cls=f"{__name__}.HY4HiddenProbe",
        enforce_eager=True,
        max_model_len=tokens + 1,
        max_num_batched_tokens=tokens,
        max_num_seqs=1,
        enable_prefix_caching=False,
    ) as runner:
        runner.llm.collective_rpc("capture_hidden")
        runner.llm.generate(
            [TokensPrompt(prompt_token_ids=list(range(1, tokens + 1)))],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        return runner.llm.collective_rpc("get_hidden")[0]


@pytest.mark.skipif(
    not os.getenv("HY4_FULL_CHECKPOINT"), reason="full HY4 checkpoint not set"
)
def test_full_hy4_hidden_metrics(vllm_runner):
    checkpoint = os.environ["HY4_FULL_CHECKPOINT"]
    tp = int(os.getenv("HY4_FULL_TP", "8"))
    tokens = int(os.getenv("HY4_FULL_TOKENS", "128"))
    original = _full_hy4_hidden(vllm_runner, checkpoint, False, tp, tokens)
    fused = _full_hy4_hidden(vllm_runner, checkpoint, True, tp, tokens)
    result = {str(layer): metric(fused[layer], original[layer]) for layer in original}
    print(json.dumps({"layers": len(result), "hidden": result}))
