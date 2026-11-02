# HY4 TP8 Triton Sparse MLA Decode

## 基线与参考来源

- 稳定基线：`5df63a7203cdb9328d824a27b682e77c7b9928be`（`v0.32.0`），
  是本地两条 iHC 开发分支的共同祖先，包含 HY4 模型原有 iHC 架构，
  不包含之后开发的 iHC PostPre 融合优化。
- 独立分支 `feat/hy4-smallhead-triton`，worktree
  `/data/qiuyuhao/projects/vllm-hy4-smallhead-triton`。
- 参考 [minatoyukinaa/vllm 的 add_decode_kernel](https://github.com/minatoyukinaa/vllm/tree/c1252328ad5d653939a7f2eaaa5bcdcc1baa2aa3)，
  固定到 `c1252328ad5d653939a7f2eaaa5bcdcc1baa2aa3`；实现提交为
  `87113d03b369ee56218cf54868895e41ce5ae3b7`。只借鉴其 decode 设计，未合并分支。
- 参考实现是 TileLang + BF16 KV。生产路径使用 **普通 Triton + BF16 KV**，
  Gluon 版本保留在实验目录和 Git 历史中供对照；没有引入 TileLang 依赖。
- 初次审计时 `vllm-hy4-dev` HEAD 为 `e3e9cdee60`、`vllm-hy4-ihc` HEAD 为
  `da9bc79158`。本任务没有修改这些 worktree、权重、Scheduler、CUDA FlashMLA 或 iHC。

## 实际布局与实现

只读核对本机 `/data/chenhao/Hy4-preview-FP8-3-layers/config.json`：

| 字段 | 值 |
| --- | --- |
| num_attention_heads / TP8 每卡 | 64 / 8 |
| qk_nope_head_dim / qk_rope_head_dim | 192 / 64 |
| kv_lora_rank / v_head_dim | 512 / 256 |
| index_topk / learnable_sink | 2048 / true |

MLA absorption 后 Q/K 维度为 576，attention 输出 latent 维度为 512，之后
投影到 V dim 256。softmax scale 沿用 Backend，此配置为 `(192+64)^-0.5=0.0625`。

BF16 KV 逻辑布局为 `[pages,page_size,576]`：512 个 latent 和 64 个 RoPE，
每 token **1152 bytes**。现有 `flat_kv_row_view` 保留页间距，现有 index group
将每请求逻辑 TopK 转成物理 slot。Kernel 接收 `[rows,576]` 和 `[T,K]` int32 索引，
屏蔽负数、越界 slot 及可选长度以外的项；cache 地址使用 int64。

原 FP8 cache 是 656B：512B E4M3 latent、16B FP32 分组 scale、128B BF16 RoPE。
它保留原生 FlashMLA。FP8 模型权重与 KV cache dtype 是两个独立选项；新 kernel
不会把 FP8 KV 隐式转成 BF16。BF16 KV 相比 656B FP8 KV 占用约 1.76 倍空间。

`smallhead_sparse_decode.py` 使用普通 `@triton.jit`、`tl.load` 和 `tl.dot`，
固定 64-key tile、4 warps、`num_stages=2`，支持 8/16/32 heads。对齐 BF16 gather
的异步拷贝、shared swizzle 和 MMA 布局由编译器生成；生产代码没有 Gluon 布局
或手写 shared-memory descriptor。未对齐视图沿用相同接口并已通过测试。
QK/PV 使用 Tensor Core，softmax 和 accumulator 为 FP32。每个 split 保存 512 维 accumulator、
max 和 denominator，第二个 kernel 合并后只加一次 sink。返回的自然对数 LSE
不包含 sink；空行输出零、LSE 为负无穷。

沿用参考分支的 host-shape split 策略：
`next_pow2(min(32, ceil(SM/T), ceil(K/64)))`。
`VLLM_BATCH_INVARIANT=1` 默认固定为 1 split；demo 可用 `--splits` 手动对照。

HF override `hy4_smallhead_triton` 默认 false。开启后在 metadata 构造时选择路径：
SM90、BF16 cache/Q、8/16/32 heads、512+64 维、PCP=DCP=1、无 HiSparse、
单 token 纯 Decode、`T*heads <= 4096` 才进入 Triton。

实验选项 `hy4_smallhead_mixed` 默认 false；与 `hy4_smallhead_triton=true` 一起
开启后，混合 batch 中连续的单 token decode 前缀使用 Triton，其余 token 使用
原生 FlashMLA。依据 CPU 上的真实阶段与 query 长度识别前缀，短 prefill 不会
误入 Triton；遇到短 prefill 或多 token decode 后的请求也保留原生路径。
`T*heads` 的限制只计算选中的 decode 前缀。两段共用 BF16 cache，一次索引转换，
同一 stream 顺序执行，直接写入 head-padded 输出的不同切片，无输出拼接。
没有修改 Scheduler。混合路径并非所有比例都更快，实测见 [RESULTS](RESULTS.md)。

复用 `benchmark_hy4_smallhead_mixed.py` 做三组对照：A 整批原生；B 拆批后两段
均用原生；C 拆批后 Triton decode + 原生 prefill。GPU7 的 H100 实测九组、
每组 12 轮：C 均快于 B，但相对 A 有六组加速、三组回退，净延迟变化
为 -28.08% 至 +3.29%。这说明拆批成本可能抵消 decode 收益；当前保持混合
实验开关默认关闭。完整数据、正确性范围与复测命令见 [RESULTS](RESULTS.md)。

## 本地验证

复用 `tests/kernels/attention/test_flashmla_sparse.py` 和现有 FlashMLA 算子；
只增加 demo 级用例。FP32 和原生 FlashMLA 双对照，覆盖 splits、strided/未对齐
视图、无效索引、长度、空行、无穷 sink、TopK 尾块、Graph replay，以及 Backend
ON/OFF 的真实页表映射和 prefill 路由。output 容差 `rtol=0.02, atol=0.01`，
LSE 容差 `rtol=atol=1e-4`。相关回归 23 项通过，未做完整模型精度评估。

本机为 H100 80GB SM90，Torch 2.13.0+cu129、Triton 3.7.1、Python 3.12.3，
已安装 vLLM 扩展为 `0.32.0+cu129`。本 worktree 使用独立 `.venv`，通过
`/tmp/hy4-smallhead-bootstrap/sitecustomize.py` 连接源码与已有扩展，
未覆盖系统安装或 iHC 环境。在本机命令前设置
`CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/tmp/hy4-smallhead-bootstrap`；
目标机器应安装本分支及匹配的扩展，不依赖这个临时 overlay。

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py \
  -k 'hy4_smallhead or metadata_smoke or matches_cache_writer_scales or prefill_smoke' \
  -q --confcutdir=tests/kernels/attention

.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_decode.py \
  --tokens 1 2 4 --pool 64 --rounds 3 --independent-requests --output small.json
.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_decode.py \
  --tokens 8 16 32 64 128 --pool 4 --rounds 3 --independent-requests --output decode.json
.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_decode.py \
  --tokens 256 512 --pool 2 --rounds 3 --independent-requests --output large.json
```

每个请求使用独立 16K KV 区域，轮换输入使实际选中 slot 的总字节数超过 L2。
先验证正确性，再三轮交错测量，取中位数。使用 CUDA Graph/GPU events，编译、
输入生成、索引转换、原生 Head padding 均不计时；Triton 包含 split 和 merge。
JSON 的 `splits=0` 表示上述自适应策略。`estimated_kv_gbps` 按 `T*K*1152/time`
估算，包含重复 slot，不含 Q、输出与 partials；它不是实测 HBM 带宽。

本机原生 BF16 FlashMLA 不接受 8 heads（报 `Unsupported h_q: 8`），因此沿用
本地 HY4 baseline 的 64-head padding。参考分支已移除该 padding，两者原生
内核版本有差异，不能直接拿本机加速比与同事的 Serving 数字对比。
详见 [结果及 SM 分析](RESULTS.md)。

普通 Triton 生产配置来自已完成的 BF16 同口径消融，1–512 tokens 的延迟与
Gluon 相差约 -0.24% 至 +2.24%。迁移后 23 项回归通过；按用户选择沿用已有
性能验证，没有在其他任务占卡期间重新计时。数据来源及限制见
[迁移验证](RESULTS.md)，历史布局消融见 [Gluon 消融报告](ABLATION.md)。

可选的 Nsight Compute demo，编译和输入准备在采样区间之外：

```bash
ncu --profile-from-start off --section SpeedOfLight --section Occupancy \
  --section LaunchStats --export hy4-bf16 \
  .venv/bin/python lab/hy4_smallhead/profile_decode.py \
  --tokens 128 --independent-requests
```

## H200 TP8 手动 Serving A/B

按用户安排，H200 测试由用户手动执行。先跑上面的正确性及 kernel benchmark，
然后使用完整 checkpoint。两端都使用 BF16 KV，只改开关；FP8 baseline 如需
比较应单独记录，避免把 cache dtype 的变化算成 kernel 收益。

```bash
export MODEL=/path/to/full/HY4/checkpoint
export SMALLHEAD=false  # A；B 改为 true，重启相同命令

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --tensor-parallel-size 8 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-backend FLASHMLA_SPARSE --served-model-name hy4-test \
  --host 127.0.0.1 --port 8000 --max-model-len 32768 --max-num-seqs 128 \
  --hf-overrides "{\"hy4_smallhead_triton\":$SMALLHEAD}"

.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 \
  --model hy4-test --tokenizer "$MODEL" --dataset-name random \
  --random-input-len 8192 --random-output-len 512 --ignore-eos \
  --seed 42 --num-prompts 512 --max-concurrency 128 \
  --save-result --result-dir results-serving --result-filename A-c128-r1.json
```

先加 `--enforce-eager` 检查固定请求输出，再用相同 Graph 配置重复 A/B。
本地验证覆盖算子 Graph replay，尚未覆盖完整 HY4 模型 Graph。预热后逐点测试
Serving 并发 1/8/16/32/64/128，保持 tokenizer、生成长度、seed、prefix cache、
GPU memory 设置相同；按 A-B-B-A 交错测量 TPOT/ITL、TTFT、吞吐和请求延迟。
同时记录实际 Decode token batch、KV 容量与 preemption：请求并发数不等于
每步 Decode token 数，混入 prefill 的 batch 会使用原生路径。用 profiler
确认 B 命中 `_split_kv` / `_softmax_merge`。两个优化分支继续分别测试。
