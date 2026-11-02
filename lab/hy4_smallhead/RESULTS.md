# H100 BF16 单算子结果与 SM 分析

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

## SM 是否打满

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

## 验证范围

相关算子及 Backend 回归 **23 passed**，涵盖 FP32/FlashMLA 对照、8/16/32 heads、
不同 splits、strided/未对齐输入、长度/无效项、空行、无穷 sink、TopK 尾块、
CUDA Graph replay、真实页表映射及 prefill fallback。使用 BF16 显式容差，
最大绝对误差记录在 JSON 中；不声称逐 bit 一致或已完成模型精度评估。

Backend 默认原生 FlashMLA；新路径仅在显式开启且满足参考设计适用条件时使用。
H200 TP8 Serving A/B、完整模型 Graph 与模型质量验证留给用户目标环境，
操作命令见 [README](README.md)。未合并 iHC，也未修改权重或 Scheduler。
