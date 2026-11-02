# HY4 small-head 验证结果

## 当前方案：仅纯 decode 加速

按用户决定，生产 Backend 和 Triton kernel 已恢复为 `d0665c674c` 的纯 decode
实现。仅显式开启 `hy4_smallhead_triton` 且满足适用条件的纯单 token decode
批次使用 Triton；纯 prefill、混合批次及多 token decode 整体走原生 FlashMLA。
`hy4_smallhead_mixed` 和混合拆分路径已移除，混合 benchmark 也从当前目录移除。

本次在 GPU7 H100 复用已有测试，**34 passed**：覆盖纯 decode ON/OFF、
prefill/混合批次整体原生调用、短 prefill 阶段识别、FP32/FlashMLA 对照和
CUDA Graph replay。代码检查通过。两个生产文件与 `d0665c674c` 完全一致；
此次未重新进行性能计时，纯 decode 性能数据仍为下方此前的验证结果。

以下三个混合分发章节为历史实验记录；其中的开关及命令只适用于历史提交
`f775dcc1d3`，不能在当前版本用于开启或测量混合加速。复现时应使用该历史提交
的独立 worktree。原始结果保留用于分析，不再继续混合分发开发。

## 混合分发的实际执行计数

2026-11-02，GPU7 H100；生产代码保持 `b9efac57c3`，性能实验提交为
`833dbc24f5`。给原 demo 添加 `--audit-dir`：在 CUDA Graph capture 时包装
真实函数并记录 Q/输出切片，随后移除包装，仅追踪一次 graph replay。
每次 replay 仍轮换四组输入；下面计数已除以四，表示一次混合 batch 调用。
这次只核验执行路径，不用 profiler 时间更新前面的性能结果。

另核对本机 Triton 3.7.1 的 `do_bench_cudagraph` 源码：graph 内展开
`n_repeat` 次 `run`，每次 `run` 包含四组完整输入；event 耗时先除以
`n_repeat`，对 10 次 replay 求均值后返回。demo 再乘 1000 并除以 4，
得到每个混合 batch 的 µs。因此没有把拆批后的两段再除以 2，也没有漏除
四组输入。前表是外层 12 轮返回值的中位数。

以 **8 decode + 512 prefill** 为例：

| 项目 | A 整批原生 | B 拆批全原生 | C Triton + 原生 |
| --- | --- | --- | --- |
| 原生 attention 调用的 Q shape | `[520,64,576]` | `[8,64,576]`、`[512,64,576]` | `[512,64,576]` |
| Triton 调用的 Q shape | 无 | 无 | `[8,8,576]` |
| 索引 remap kernel 次数 | 1 | 1 | 1 |
| FlashMLA attention kernel 次数 | 1 | 2 | 1 |
| Triton Split-KV 次数 | 0 | 0 | 1 |
| Triton Merge 次数 | 0 | 0 | 1 |
| 原生 sink padding fill 次数 | 1 | 2 | 1 |
| 原生 attention grid | `(520,1,1)` | `(8,1,1)`、`(512,1,1)` | `(512,1,1)` |

C 的 Split-KV grid 为 `(8,32,1)`，Merge 为 `(8,8,1)`。
说明 splits 根据 **拆分后的 8 decode tokens** 选择，而不是整批 520 tokens；
若误传 520 tokens，默认策略会变为 1 split。128 decode 的两组 C 则实际发射
`(128,2,1)` 和 `(128,8,1)`。这些是 GPU trace 的实际 grid，不只是源码推断。

四组形状 `8+128 / 8+512 / 128+128 / 128+512` 全部符合上述调用计数。
每组单次 replay：A 的原生 attention 共 4 次；B 共 8 次；C 的原生、Split-KV、
Merge 各 4 次。B/C 的 Q 共用输入 storage，prefill token 偏移恰为 `nd`；
两段输出共用 storage，偏移分别为 0 和 `nd`，切片互不重叠。没有整批进入
Triton、重复处理 prefill 或额外执行一次整批 FlashMLA 的现象。

所有 GPU kernel 位于同一 stream，按 decode 后 prefill 顺序执行，**没有并行
重叠**。原生 sink 的 8 个 FP32 值仍需复制到 padding buffer；trace 中表现为
32-byte DtoD copy，部分捕获表现为 `memcpy32_post` kernel。这不是输出拼接。
因此不能只按总 kernel 数解释性能，还必须区分 attention、辅助 fill/copy。

本次四组 A/B/C 的 FP32 对照通过；复用现有 `hy4_smallhead_backend` 和
`hy4_smallhead_metadata_phase` 用例，**26 passed**。demo 手工指定 metadata
前缀用于隔离 Backend；实际 metadata builder 的识别由这些现有测试覆盖，
仍未验证完整模型 Serving。调用计数、shape、storage 偏移、完整 kernel 名称、
grid、stream 及 trace 路径保存在
[h100_mixed_execution_audit.json](results/h100_mixed_execution_audit.json)。

这证明当前拆分分发按设计执行，不等于证明全部实现无缺陷或性能最优。decode
使用现有自适应 Split-KV 策略，prefill 保留本地 pad64 原生 kernel；没有针对
混合形状重新搜索 splits/tile，也没有做多 stream overlap。前面的 A/B/C 只
比较这套现有配置，不能推出所有分阶段实现都会出现同样的回退。

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/tmp/hy4-smallhead-bootstrap \
TRITON_CACHE_DIR=/tmp/hy4-plain-production-cache \
.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_mixed.py \
  --decodes 8 128 --prefills 128 512 \
  --audit-dir /tmp/hy4-mixed-execution-traces --output execution-audit.json

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/tmp/hy4-smallhead-bootstrap \
.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py \
  -k 'hy4_smallhead_backend or hy4_smallhead_metadata_phase' \
  -q --confcutdir=tests/kernels/attention
```

## 混合 batch A/B/C：区分拆批成本与 decode 收益

2026-11-02，在物理 GPU7（H100 80GB，UUID 尾号 `754aa34b024c`）完成。
生产源码为 `b9efac57c3`，本次只扩展已有 demo，不修改生产分发规则。
8 heads、TopK=2048、BF16 KV，QK=576、latent V=512。

- A：整个混合 batch 调用一次原生 FlashMLA。
- B：decode/prefill 分开，两段均用原生 FlashMLA。
- C：decode/prefill 分开，Triton decode + 原生 FlashMLA prefill。

B 与 C 共用相同索引转换方式及输出切片；两段在同一 stream 顺序执行。
B−A 衡量拆批的整体影响，包含额外调用、sink 准备和原生 kernel 形状变化，
不能全部解释成 launch 开销。C−B 衡量拆批结构内替换 decode 实现的收益，
也包含关联的 sink 准备差异；C/A−1 才是相对整批 baseline 的净延迟变化。

| Decode tokens | Prefill tokens | A µs | B µs | C µs | B−A µs | C−B µs | C/A−1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 129.61 | 142.50 | 93.22 | +12.90 | -49.28 | -28.08% |
| 8 | 512 | 272.05 | 330.08 | 281.02 | +58.03 | -49.06 | +3.29% |
| 8 | 2048 | 1035.45 | 1078.37 | 1042.47 | +42.92 | -35.90 | +0.68% |
| 32 | 128 | 136.24 | 146.85 | 113.62 | +10.61 | -33.23 | -16.60% |
| 32 | 512 | 311.99 | 335.60 | 303.36 | +23.61 | -32.25 | -2.77% |
| 32 | 2048 | 1049.35 | 1086.89 | 1063.49 | +37.55 | -23.40 | +1.35% |
| 128 | 128 | 191.36 | 197.26 | 188.78 | +5.90 | -8.49 | -1.35% |
| 128 | 512 | 383.94 | 388.77 | 377.71 | +4.83 | -11.06 | -1.62% |
| 128 | 2048 | 1143.67 | 1149.92 | 1131.77 | +6.25 | -18.14 | -1.04% |

九组中，B 的中位数都比 A 慢；C 在每组的全部 12 轮中都比 B 快。
因此这批输入里，Triton decode 确实改善了拆批路径。净结果有六组加速、三组
回退，各组 12 轮的 C/A 快慢方向一致。例如 8+512，拆批增加约 58.03 µs，
替换 decode 省约 49.06 µs，净增加约 8.96 µs（3.29%）。不能据此推出单一的
prefill 长度临界点：decode 数量、KV 访问与原生 kernel 形状都影响收益。
128 decode 的净收益仅约 1–1.6%，仍需目标硬件和真实请求分布复验。

复用原输入生成与 FP32 reference，每个形状的 A/B/C 都逐段比较最后一组输入
的全部 token，27 组形状/路径检查通过（`rtol=0.02, atol=0.01`）。
每个 decode 请求独立 16K 上下文，prefill 是一个共享 8K 前缀的请求；随机
TopK 按 query 位置限制为因果范围。四组独立输入轮换，未显式 flush L2。
每方案 12 轮、每轮 `rep=100ms`；六种执行顺序各重复两次，表中为各方案
中位数及其差值/比值。索引转换、sink 准备、attention、merge 计入 CUDA
Graph/event 时间；输入生成、Q head padding、Python/metadata builder 不计时。
原生仍为本地 pad64 版本，不代表支持 8 heads 的新版 FlashMLA。

GPU 进程监控 224 次采样仅见本次 benchmark PID `2090390`，未观察到其他
计算进程。Torch `2.13.0+cu129`、Triton `3.7.1`、CUDA `12.9`；原始样本、
执行顺序、seed、GPU UUID 和 demo 源码 SHA256 见
[h100_mixed_dispatch_abc.json](results/h100_mixed_dispatch_abc.json)。

保持 `hy4_smallhead_mixed=false` 为默认；此次没有新增按样本拟合的自动门控。
这是单卡 Attention backend 实验，不是 TP8 Serving，也没有测量 TTFT/TPOT。
H200 TP8 的最终启用判断留待真实 Serving A/B。

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/tmp/hy4-smallhead-bootstrap \
TRITON_CACHE_DIR=/tmp/hy4-plain-production-cache \
.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_mixed.py \
  --decodes 8 32 128 --prefills 128 512 2048 --rounds 12 --rep 100 \
  --output mixed-abc.json
```

## 首轮混合 batch A/C：Triton decode + 原生 prefill

实现基于 `d0665c674c`，新增 `hy4_smallhead_mixed` 实验选项，默认 false。
需同时开启 `hy4_smallhead_triton`，并使用 BF16 KV。现有纯 decode 路径保留。
metadata 识别真实单 token decode 前缀；混合路径转换一次物理索引，两段在同一
stream 顺序调用，写同一输出缓冲区。Triton merge 支持 padded token stride，
无需 `cat` 或输出复制。DCP/PCP、HiSparse、FP8 KV 继续沿用原有适用限制。

已有测试扩展覆盖 0/1/2/4 decode tokens 的 ON/OFF 分发、共享 prefill 请求的
真实页表、固定形状 CUDA Graph replay、短 prefill 和多 token decode 的路由。
首轮相关回归 **34 passed**；加入默认关闭的实验开关后，阶段/开关测试
**18 passed**（含此前阶段用例）。九组 benchmark 的双路径 FP32 对照全部通过。
全量模型、动态混合形状的 serving 与 H200 仍未验证。

2026-11-02，H100 GPU7；8 heads、TopK=2048、BF16。每个 decode 请求独立 16K
上下文，另有一个共享 8K 前缀的 prefill 请求，逐 query 限制随机 TopK 以保持因果。
轮换四组独立输入；先逐段校验 FP32，再做六轮 AB/BA 交替。计时包含 backend
中的索引转换、sink 准备、attention 和 merge；Q 的 head padding 在计时外，
双方共用同一 padded Q。CUDA Graph/event timing，不含 Python 分配与 metadata
builder 时间。基线是本地需要 pad 到 64 heads 的 FlashMLA，不代表新版 small-head
FlashMLA。采样时检查 GPU7 未出现其他计算任务。

| Decode tokens | Prefill tokens | 整批原生 µs | 分阶段 µs | 延迟变化 |
| --- | --- | --- | --- | --- |
| 8 | 128 | 130.04 | 93.50 | -28.10% |
| 8 | 512 | 267.03 | 279.43 | +4.64% |
| 8 | 2048 | 1022.21 | 1035.46 | +1.30% |
| 32 | 128 | 136.11 | 113.68 | -16.48% |
| 32 | 512 | 312.38 | 300.66 | -3.75% |
| 32 | 2048 | 1036.33 | 1065.17 | +2.78% |
| 128 | 128 | 189.72 | 189.06 | -0.35% |
| 128 | 512 | 378.99 | 376.28 | -0.72% |
| 128 | 2048 | 1140.98 | 1130.54 | -0.91% |

原始六轮样本：[h100_mixed_dispatch.json](results/h100_mixed_dispatch.json)。
每个结果为独立方案样本的中位数，变化为中位数之比。该 demo 显示收益取决于
混合比例，最高约降低 28.1%，也出现约 4.6% 回退。因此保留独立实验开关，不据
九个形状拟合自动选择规则。分批也改变原生 kernel 的形状，不能把全部收益归因于
替换 decode kernel；这些是 Attention 调用结果，不能直接当作 TTFT/TPOT 收益。

```bash
# Serving 的 HF override（其他模型与启动参数保持一致）
# --hf-overrides '{"hy4_smallhead_triton":true,"hy4_smallhead_mixed":true}'
.venv/bin/python benchmarks/kernels/benchmark_hy4_smallhead_mixed.py \
  --decodes 8 32 128 --prefills 128 512 2048 --rounds 6 --rep 60 \
  --output mixed.json
```

## 普通 Triton 生产迁移验证

生产 Split-KV 已采用之前测量的 `tl_n64_w4_s2` 实现：64-key tile、4 warps、
2 stages，使用 `tl.load` / `tl.dot`。公共 API、BF16 cache、Split-KV 策略、
FP32 partial、softmax merge、sink/LSE 语义和 FlashMLA 开关均保留。
生产内核移除了 Gluon 的布局类型、shared descriptors、显式 async-copy helper
和手写对齐分支。Gluon 只用于实验对照，不从生产内核调用。

本次验证：原有算子、Backend ON/OFF 和 FlashMLA 回归 **23 passed**。覆盖
8/16/32 heads、空行、长度/无效索引、未对齐/strided 视图和 CUDA Graph replay。
静态核对确认生产 Split-KV 与消融内核在固定 `BN=64` 后 AST 完全一致。

按用户选择，性能验收沿用此前已完成的五轮同口径测量；本次迁移没有新增 GPU
性能数据。以下是 **已有消融数据**，不是迁移后的重新计时。H100、8 heads、
TopK=2048、BF16 Q/KV、相同输入/splits/partial/merge，CUDA Graph/event timing。

| Decode tokens | 冻结 Gluon (µs) | 所采用的普通 Triton (µs) | 延迟变化 |
| --- | --- | --- | --- |
| 1 | 8.80 | 8.93 | +1.44% |
| 4 | 11.58 | 11.68 | +0.81% |
| 8 | 16.40 | 16.74 | +2.07% |
| 16 | 22.85 | 23.36 | +2.24% |
| 32 | 35.83 | 35.74 | -0.24% |
| 64 | 60.62 | 60.69 | +0.11% |
| 128 | 109.71 | 110.00 | +0.26% |
| 256 | 207.88 | 208.29 | +0.20% |
| 512 | 407.13 | 407.54 | +0.10% |

原始每轮样本见 [h100_gluon_ablation_final.json](results/h100_gluon_ablation_final.json)。
最大延迟差约 +2.24%；本机未观察到明显的大 batch 性能回退。这个结论只覆盖
上述 H100/编译器版本与算子范围，H200 TP8 Serving 仍由用户手动验证。

在可用 GPU 上，可直接复测当前生产内核与冻结 Gluon：

```bash
.venv/bin/python lab/hy4_smallhead/ablation.py \
  --tokens 1 4 8 16 32 64 128 256 512 --variants current production \
  --rounds 5 --rep 100 --output production-vs-gluon.json
```

`current` 为历史结果兼容名称，固定指向 Gluon 基线；`production` 为当前普通
Triton 生产内核。新输出包含生产源码 SHA256，避免混淆历史数据和后续复测。

## 历史 Gluon BF16 单算子结果与 SM 分析

2026-11-02；8 query heads，QK=576，latent V=512，TopK=2048，BF16 Q/KV。
GPU5 三轮交错测量的中位数，CUDA Graph/event timing，Triton 包含 split+merge。
原生 FlashMLA 使用本地基线要求的 64-head padding，padding 本身在计时外。
H100 SM90 的结果不能代替 H200 TP8 Serving 结论。

| Decode tokens | 自动 splits | Triton (µs) | 原生 FlashMLA (µs) | 加速比 |
| --- | --- | --- | --- | --- |
| 1 | 32 | 8.50 | 59.29 | 6.98× |
| 2 | 32 | 8.61 | 60.12 | 6.98× |
| 4 | 32 | 10.83 | 60.09 | 5.55× |
| 8 | 32 | 15.13 | 61.50 | 4.07× |
| 16 | 16 | 22.10 | 62.70 | 2.84× |
| 32 | 8 | 35.01 | 64.64 | 1.85× |
| 64 | 4 | 59.85 | 73.26 | 1.22× |
| 128 | 2 | 109.50 | 117.00 | 1.07× |
| 256 | 1 | 208.57 | 226.82 | 1.09× |
| 512 | 1 | 408.05 | 444.66 | 1.09× |

输入采用独立请求 KV 区域；1/2/4 tokens 轮换 64 组，8–128 tokens 轮换 4 组，
256/512 tokens 轮换 2 组。各点实际访问 footprint 均超过 50 MB L2，最小值
70.9 MB；未在每次调用前强制清空 L2。数据分别见
[小 batch](results/h100_bf16_small.json)、
[8–128 tokens](results/h100_bf16_decode.json)、
[256–512 tokens](results/h100_bf16_large.json)。JSON 保留环境和计时口径。

当前 H100 上没有观察到 128 tokens 后的明显回退，但 64 tokens 之后优势只有
约 7–22%。单算子仍需处理与 batch 近似成正比的 KV 字节数；大 batch 下两种
实现都接近内存带宽限制。不能据此推导 Serving 的 TTFT/TPOT、吞吐或延迟：
Serving 并发与 Decode batch 不同，还包含 prefill、Indexer、MoE、TP 通信和排队。

本机 `0.32.0+cu129` 的 BF16 FlashMLA 直接使用 8 heads 会报
`Unsupported h_q: 8`，只能按现有 HY4 路径 pad 到 64 heads。参考分支允许
原生 BF16 使用 8 heads。因此本结果证明对**本地稳定基线**的收益，不能宣称
优于同事使用的原生 FlashMLA 版本，也不能与此前 656B FP8 KV 数字横向比较。

### SM 是否打满

GPU2 使用 Nsight Compute 2025.2.1，batch=128，独立 KV 区域，自动 2 splits。
原始指标及单位见 [h100_bf16_ncu.json](results/h100_bf16_ncu.json)。

| Kernel | CTA 数 | Registers/thread | Shared/CTA | Achieved occupancy | Tensor active | DRAM throughput |
| --- | --- | --- | --- | --- | --- | --- |
| Triton split | 256 | 196 | 95.232 KB | 11.98% | 16.09% | 82.04% |
| Triton merge | 1024 | 29 | 2.048 KB | 41.59% | 0.19% | 24.64% |
| FlashMLA BF16 pad64 | 128 | 168 | 231.888 KB | 18.40% | 39.33% | 80.94% |

Shared 列包含 driver 保留空间，KB 为十进制。Tensor active 和 DRAM throughput
是 Nsight 的 peak-sustained elapsed 百分比；occupancy 是 active warps 指标。
profiler 使用 replay 和默认时钟/缓存控制，其耗时不混入上面的稳态 benchmark。

**计算 SM 没有打满，DRAM 已达到工具报告峰值的约 82%。** Split kernel 每 CTA
4 warps，寄存器与共享内存限制为每 SM 最多 2 CTA；理论 occupancy 约 12.5%，
实测约 12%，并不是只有 12% 的 SM 工作。256 个 CTA 可覆盖本机 132 个 SM。
原生 kernel 还计算 padding heads，较高 Tensor active 不等于同等有效工作更多。

参考设计减少了间接 KV gather 的寄存器中转，并随 batch 减少 splits。在 batch
128 时仅 2 splits，FP32 partial buffer 约 4.02 MiB；大 batch 不再保留 32 份
partial。后续若针对 H200 继续调优，应重测内存吞吐和延迟，而非仅追求 occupancy。

### 验证范围

相关算子及 Backend 回归 **23 passed**，涵盖 FP32/FlashMLA 对照、8/16/32 heads、
不同 splits、strided/未对齐输入、长度/无效项、空行、无穷 sink、TopK 尾块、
CUDA Graph replay、真实页表映射及 prefill fallback。使用 BF16 显式容差，
最大绝对误差记录在 JSON 中；不声称逐 bit 一致或已完成模型精度评估。

Backend 默认原生 FlashMLA；新路径仅在显式开启且满足参考设计适用条件时使用。
H200 TP8 Serving A/B、完整模型 Graph 与模型质量验证留给用户目标环境，
操作命令见 [README](README.md)。未合并 iHC，也未修改权重或 Scheduler。
