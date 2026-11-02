# H100 SM90 单算子结果

2026-11-02，在独立 worktree 的 H100 GPU2 完成。硬件不是 H200。
8 query heads，QK=576，latent V=512，TopK=2048，BF16 Q，生产算子写入的
656B FP8 KV。三轮交错测试，CUDA Graph/event timing，32 splits、32-key tile。

| Decode tokens | Triton (µs) | 原生 FlashMLA (µs) | 加速比 | ON 时选择 |
| --- | --- | --- | --- | --- |
| 1 | 15.17 | 20.39 | 1.34× | Triton |
| 2 | 15.38 | 21.40 | 1.39× | Triton |
| 4 | 16.11 | 20.91 | 1.30× | Triton |
| 8 | 19.76 | 23.87 | 1.21× | Triton |
| 16 | 33.06 | 32.10 | 0.97× | FlashMLA |
| 32 | 63.65 | 47.11 | 0.74× | FlashMLA |

每个点轮换 64 份缓存，最小访问 footprint 为 80.8 MB，超过 L2。
原始数字和环境版本见 [h100_rotating64.json](results/h100_rotating64.json)。
FlashMLA 的 padding 和 metadata 创建在计时外，Triton 包含 split + merge；
不包含模型其他计算、索引生成/转换和 collective。

前置参数探索比较了 8/16/32 splits 及 32/64-key tiles；
[32-key](results/h100_block32.json) 和 [64-key](results/h100_block64.json)
使用 8 缓存，可能复用 L2，不与最终 64 缓存数字混为同一口径。
32 splits 对 1–8 token 有收益，较大的 batch 回退。64-key tile 在最小 batch
略快但在 8 token 时优势较小，首版使用单一 32-key tile，避免复杂参数表。

相关回归 **25 passed**。FP32 reference、原生 FlashMLA、空行、无穷 sink、strided 输入、page size、
split 配置和 CUDA Graph replay 已验证；Backend OFF/ON 及 prefill fallback
另用真实 page table 和 index-group 转换验证。输出按 BF16 显式容差验收，
不声明逐 bit 相同。完整模型数值评估和 H200 TP8 Serving A/B 留给目标环境。

本机用新 worktree 源码加已安装 CUDA 扩展的 overlay，未重编 CUDA。
没有在缺少完整 HY4 权重、且 GPU 被其他任务占用时尝试 TP8 Serving。
该结果只能支持 H100 单算子小 batch 的结论；H200 是否保留相同阈值需实测。
