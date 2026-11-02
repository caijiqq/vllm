# HY4 TP8 Small-head Sparse MLA Decode

## 基线和隔离

- 基线：`5df63a7203cdb9328d824a27b682e77c7b9928be`，本地标签 `v0.32.0`。
  这是 `feat/hy4-ihc-dev` 与 `exp/hy4-ihc-lab` 的共同祖先，已包含 HY4
  原生实现，未包含之后的 iHC PostPre 融合开发。这里的“不含 iHC 改动”指不含
  **新融合优化**；模型架构原有的 iHC 必须保留。
- 分支：`feat/hy4-smallhead-triton`。
- worktree：`/data/qiuyuhao/projects/vllm-hy4-smallhead-triton`。
- 原目录 `vllm-hy4-dev` 的 HEAD 为 `e3e9cdee60`，已有两个未跟踪的 serving
  脚本；`vllm-hy4-ihc` 的 HEAD 为 `da9bc79158`。本任务没有改写这些目录或分支。
- 未修改权重、Scheduler、CUDA FlashMLA 或 iHC 实现；没有 merge/cherry-pick iHC。

## 实际算子契约

只读核对本机 `/data/chenhao/Hy4-preview-FP8-3-layers/config.json`：

| 字段 | 实际值 |
| --- | --- |
| num_attention_heads / TP8 每卡 | 64 / 8 |
| qk_nope_head_dim / qk_rope_head_dim | 192 / 64 |
| kv_lora_rank / v_head_dim | 512 / 256 |
| index_topk | 2048 |
| learnable_sink | true |

经过 MLA absorption，Decode Q/K 是 `[T,H,576]`，attention 输出 latent
是 `[T,H,512]`，之后才投影到配置中的 V dim 256。softmax scale 沿用
Backend 的值；该配置是 `(192+64)^-0.5 = 0.0625`，不能用 `576^-0.5`。

FP8 KV 的逻辑形状是 `[pages,page_size,656]`，每行连续存放：

| 字节偏移 | 内容 |
| --- | --- |
| 0–511 | 512 个 E4M3 FP8 latent |
| 512–527 | 4 个 FP32 scale，每 128 维一组 |
| 528–655 | 64 个 BF16 RoPE |

没有 DSv4 的页尾 UE8M0 scale 分区。生产 Backend 将较大的 block 解释成
64-token FlashMLA pages，再调用现有索引组转换逻辑索引；新 kernel 接收的是
**物理 cache slot**，`-1` 为无效项。Indexer 自己的 FP8 cache 是另一套布局。

参考 [PR #59418](https://github.com/vllm-project/vllm/pull/59418) 的
16-head tile / Split-KV 思路，缓存加载器按 HY4 重写，没有移植其模型专用路径。

## 实现和开关

- `vllm/models/hy_v4/nvidia/smallhead_sparse_decode.py`：16-head MMA，分别计算
  512 维 latent 和 64 维 RoPE 的点积；Split-KV 在线 softmax，FP32 partials，
  第二个 kernel 用 LSE 合并，最后应用 sink。返回的自然对数 LSE 不包含 sink，
  与 FlashMLA 一致；空行输出零、LSE 为负无穷。
- `hy4_smallhead_triton` 是 HF override，缺省 false；权重目录无需修改。
- 开关 ON 后仅路由 SM90、8 heads、FP8 656B KV、纯 Decode、1–8 tokens、
  PCP=DCP=1。固定 32 splits / 32-key tile。其他情况保留原生 FlashMLA，
  不套用参考 PR 的 DSv4 参数表。独立算子另外支持 16 heads。
- HY4 当前通过已有 sink-capable `HYV4FlashMLASparseBackend` 接入；
  显式选择其他 backend 或不使用 learnable sink 的模型不属于该接入范围。
- 不自动启用，不改变模型的 `enable_ihc`；新 iHC 融合代码根本不在本分支。

## 正确性和 benchmark

复用现有 `tests/kernels/attention/test_flashmla_sparse.py`，输入由生产
`concat_and_cache_mla` 写入。FP32 参考和原生 FlashMLA 双对照，覆盖 8/16 heads、
64/128-token pages、不同 splits、strided views、空行、单有效项、无穷 sink
以及修改 Q 后 CUDA Graph replay。Backend demo 使用真实页表转换，比较 OFF/ON。

容差：BF16 output `rtol=0.02, atol=0.04`；FP32 LSE `rtol=atol=1e-4`。
输入刻意跨多个 scale 范围；JSON 记录最大绝对误差，不将其解释成模型精度指标。

在已安装匹配版本 vLLM/CUDA 扩展的环境中，从本分支根目录运行：

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py \
  -k hy4_smallhead -q --confcutdir=tests/kernels/attention

.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_decode.py \
  --heads 8 --tokens 1 2 4 8 16 32 --topk 2048 \
  --splits 8 16 32 --block-n 32 --pool 64 --rounds 3 --output kernel.json
```

`--confcutdir` 使算子 demo 不加载整个模型测试套件。没有 try/catch、复杂 fixtures
或 serving 自动化框架。benchmark 先检查正确性，再按轮次交错测量两种实现。
计时用 CUDA Graph / GPU events，预热、编译、输入生成和 metadata 准备不计入。
FlashMLA 的 Head padding 也在计时外，所以这是保守的 **kernel-only** 比较。

本机没有 CUPTI Python 包，使用仓库 microbenchmark 指南允许的 graph/event
轮换缓存方式。默认 64 份缓存：JSON 的 `accessed_kv_bytes` 统计实际选中 slot
的总字节数，默认 K=2048 时超过 H100/H200 的 L2；不是只比较已分配缓存容量。
这不是每次调用前显式清空 L2。`estimated_kv_gbps` 按 `T*K*656/time` 估算，
不包括 Q、输出和 partials，也没有扣除重复 slot。

本地环境：8×H100 80GB SM90；只使用空闲单卡做算子测试。Torch 2.13.0+cu129，
Triton 3.7.1，Python 3.12.3，已安装 vLLM 二进制为 `0.32.0+cu129`。
新 worktree 使用独立 `.venv`，复用已安装扩展；
本机 overlay 在 `/tmp/hy4-smallhead-bootstrap/sitecustomize.py`，仅将本 worktree
源码与已安装二进制连接，没有覆盖系统安装或原 iHC 环境。

结果位于 `results/`。`h100_block32.json` / `h100_block64.json` 是 8 缓存的
参数探索，存在 L2 复用；`h100_rotating64.json` 是最终 64 缓存测量。
不代表 H200、TP8 collective、完整模型或 serving 的收益。

## H200 手动验证

使用本分支及与基线匹配的 CUDA 扩展，先运行上面的算子测试和 benchmark。
确认实际 checkpoint 仍是上述维度、TopK 和 learnable sink 配置；本机只有三层
切片，不将它当完整模型 Serving A/B。以下使用现有完整 checkpoint，不修改它：

```bash
export MODEL=/path/to/full/HY4/checkpoint
export FUSED=false  # A；B 改为 true，重启相同命令

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --kv-cache-dtype fp8 --attention-backend FLASHMLA_SPARSE \
  --served-model-name hy4-test --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --max-num-seqs 32 \
  --hf-overrides "{\"hy4_smallhead_triton\":$FUSED}"
```

先在 eager 模式加 `--enforce-eager` 做两端固定请求和输出检查，再使用上面相同
Graph 配置重复 A/B。正常 warmup 后再计时；本版本没有扩展公共 JIT warmup 注册表，
未覆盖的 shape 第一次调用可能触发 Triton 编译。Graph replay 已通过算子级验证，
完整模型 Graph 仍应现场验证。可用现有 profiler 确认 B 的 `_split_kv` 和
`_softmax_merge` 命中，而 A 没有；两端 prefill 都应保留 FlashMLA。

每端用现有 CLI，依次测试并发 1/2/4/8/16/32，至少按 A-B-B-A 交错三轮，
比较 ITL/TPOT、TTFT、吞吐和生成结果。保持 tokenizer、seed、长度、输出 token
数和 prefix cache 设置相同，先预热；不要把首轮 JIT 当稳态耗时。

```bash
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 \
  --model hy4-test --tokenizer "$MODEL" --dataset-name random \
  --random-input-len 8192 --random-output-len 512 --ignore-eos \
  --seed 42 --num-prompts 32 --max-concurrency 1 \
  --save-result --result-dir results-serving --result-filename A-c1-r1.json
```

并发大于 8 或含 prefill 的混合 batch 可回退原生实现，这是本次小 batch 优化的
预期范围。不要为了扩大命中率修改 Scheduler。此阶段单独测本分支，不合并 iHC。
