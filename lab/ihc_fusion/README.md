# HY4 iHC

这个目录只保留 iHC 的实现、参考实现和历史结果。正确性验证使用
`tests/models/test_hyv4_ihc_fusion.py`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/models/test_hyv4_ihc_fusion.py -s -q
```

测试只打印两类指标：

- 单算子：融合 Post+Pre 与原始 Post、Pre 的每个输出的最大绝对误差和余弦相似度。
- 端到端：三层 Decoder 的每层 hidden state 的最大绝对误差和余弦相似度。

78 层全量 HY4 使用同一个测试文件中的可选入口。它通过 vLLM worker extension
在真实模型层上采样最后一个 prompt token 的 hidden state：

```bash
HY4_FULL_CHECKPOINT=/path/to/full-HY4-checkpoint \
HY4_FULL_TP=8 \
.venv/bin/python -m pytest tests/models/test_hyv4_ihc_fusion.py \
  -k full_hy4_hidden_metrics -s -q
```

性能测试直接使用 vLLM 自带的 `vllm serve` 和 `vllm bench serve`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/vllm serve /path/to/HY4-checkpoint \
  --tensor-parallel-size 4 --enforce-eager

.venv/bin/vllm bench serve \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/HY4-checkpoint \
  --input-len 8192 --output-len 512 --num-prompts 16
```

测试输入 shape 从层参数读取；权重是否切分、GPU 是 H100 还是 H200
不改变测试入口。融合 Kernel 的 autotune 由 Triton 按 `(N, D)` 自动选择配置，
Cross 输出的 Decoder RMSNorm 也在同一个 Kernel 中完成。
