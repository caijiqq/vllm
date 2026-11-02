# 普通 Triton 生产迁移验证

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
