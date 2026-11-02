# HY4 混合批次实验与采用纯 decode 加速的决定

2026-11-02。当前方案：仅在显式开启 `hy4_smallhead_triton` 且满足适用条件时，
让纯单 token decode 批次使用普通 Triton；纯 prefill、混合批次、多 token decode
整体沿用原生 FlashMLA。混合分发开关和生产路径已经移除。

## 实验调用方式

实验保持 Scheduler 的混合 batch 不变，只在 Attention Backend 内按 token
切片。decode 请求在前，prefill 请求在后；共用 KV cache，一次转换整个 batch
的逻辑 TopK 到物理索引，两段直接写同一个输出缓冲区的不同切片。

```text
混合 batch
  → 整批索引转换
  → decode：Triton Split-KV → Softmax Merge
  → prefill：原生 FlashMLA
  → 返回完整输出
```

两阶段在同一 CUDA stream 上顺序执行，先 decode 后 prefill，没有并行重叠。
没有修改 CUDA FlashMLA、Scheduler、模型权重，也没有合并 iHC。

为了区分拆批影响与替换 decode 的收益，对比三种方式：

| 方案 | Decode | Prefill | Attention 执行方式 |
| --- | --- | --- | --- |
| A | 原生 FlashMLA | 原生 FlashMLA | 整批一次原生调用 |
| B | 原生 FlashMLA | 原生 FlashMLA | 分成两次原生调用 |
| C | Triton Split-KV + Merge | 原生 FlashMLA | 分阶段串行调用 |

## 实测效果

物理 GPU7，H100 80GB SM90；8 query heads（模拟 HY4 TP8 每卡 head 数）、
TopK=2048、BF16 KV、QK dim=576、latent V dim=512。原生 FlashMLA 是本地
需要将 Q pad 到 64 heads 的版本，不能代表支持 8 heads 的新版原生 kernel。
Torch 2.13.0+cu129、Triton 3.7.1、CUDA 12.9。

表中 decode 是本步单 token decode 请求数，每个请求有独立 16K 上下文；
prefill 是一个请求在本步计算的 query token 数，已有 8K 历史前缀。
例如 `8+512` 是 8 个 decode 请求加一个 512-token prefill chunk，
不是 512-token 总序列，也不是 520 个 Serving 并发请求。

| Decode tokens | Prefill query tokens | A 整批原生 µs | B 拆批原生 µs | C Triton + 原生 µs | C 相对 A 延迟变化 |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | 129.61 | 142.50 | 93.22 | -28.08% |
| 8 | 512 | 272.05 | 330.08 | 281.02 | +3.29% |
| 8 | 2048 | 1035.45 | 1078.37 | 1042.47 | +0.68% |
| 32 | 128 | 136.24 | 146.85 | 113.62 | -16.60% |
| 32 | 512 | 311.99 | 335.60 | 303.36 | -2.77% |
| 32 | 2048 | 1049.35 | 1086.89 | 1063.49 | +1.35% |
| 128 | 128 | 191.36 | 197.26 | 188.78 | -1.35% |
| 128 | 512 | 383.94 | 388.77 | 377.71 | -1.62% |
| 128 | 2048 | 1143.67 | 1149.92 | 1131.77 | -1.04% |

负数表示更快。每个方案测量 12 轮，六种 A/B/C 执行顺序各重复两次，取各方案
中位数。四组独立输入轮换，使用 CUDA Graph/GPU event 计时，包含索引转换、
sink 准备、attention 和 merge；不包含输入生成、Q padding 和 host metadata
构造。计时函数已除 graph 内重复次数，demo 再除以四组输入，没有按拆分段数
额外除以二。原始样本见 [A/B/C 数据](results/h100_mixed_dispatch_abc.json)。

九组中六组净加速、三组净回退，最大降幅 28.08%，最大回退 3.29%。在已经拆批
的结构里，C 在每组全部 12 轮中均快于 B；但 B 的中位数都比 A 慢。

例如 `8+512`：B−A 为 +58.03 µs，C−B 为 −49.06 µs，C−A 为 +8.96 µs。
替换 decode 确实改善了拆批路径，但未完全抵消拆批引入的整体成本。

## 执行计数验证

对 `8+128 / 8+512 / 128+128 / 128+512` 四组形状记录真实调用切片，
并追踪一次 CUDA Graph replay。每次 replay 含四组输入；下表归一到一个 batch。

| GPU 操作 | A 次数 | B 次数 | C 次数 |
| --- | --- | --- | --- |
| 物理索引转换 | 1 | 1 | 1 |
| 原生 FlashMLA attention | 1 | 2 | 1 |
| Triton Split-KV | 0 | 0 | 1 |
| Triton Softmax Merge | 0 | 0 | 1 |

`8+512` 的 C 实际 grid 为 Split-KV `(8,32,1)`、Merge `(8,8,1)`、
FlashMLA `(512,1,1)`，确认 split 数按拆分后的 8 个 decode tokens 选择。
Q 的 prefill 切片从 token 8 开始；两段输出共用 storage，写入范围互不重叠。
没有把整批误送进 Triton，也没有额外执行整批原生 attention。

索引转换、sink padding fill 和 32-byte sink copy 属于辅助操作，不能将全部
GPU 操作数等同于 attention 调用数。具体 kernel 名称、grid、stream、切片及
输出偏移见 [执行计数数据](results/h100_mixed_execution_audit.json)。其中 `/tmp`
trace 路径为测试机器上的临时文件位置；提取出的计数和 grid 已保存在该 JSON。

## 为什么不采用混合加速

当前实现不能保证在已测混合比例下取得净收益。decode 加速足够大时能覆盖拆批
成本；prefill 占主导或 decode 优势较小时，则可能小幅回退。为了保持当前分支
专注于纯 decode 优化，选择移除混合路径，而不是依据九个形状增加自动阈值。

拆分会改变 GPU 的调度机会：整批 native 调用可以在不同 SM 上同时调度不同
阶段的 CTA；同 stream 的两个调用则分别启动和收尾。再加上 Split-KV partial
读写、merge 及辅助操作，都可能影响净耗时。**这些是可能的机制，不是已量化的
耗时归因**：尚未完成各阶段稳态计时，B−A 不能全部解释为 launch 开销。

这个决定不表示所有混合分阶段实现都无效。实验没有针对混合形状重新搜索参数，
也没有验证多 stream overlap；但目前不继续这条优化路线。MiniMax 的分阶段
设计只提供结构参考，没有同口径的整批/拆批性能数据证明其所有混合形状都更快。

## 当前验证与适用边界

恢复后 Backend 和 Triton kernel 与纯 decode 基线 `d0665c674c` 完全一致。
复用已有测试，在 H100 GPU7 上 **34 项通过**，覆盖纯 decode 开关、prefill 和
混合批次整体回退、短 prefill 阶段识别、FP32/FlashMLA 对照和 Graph replay。
恢复时未重新计时，纯 decode 性能数据仍沿用此前验证，见 [RESULTS](RESULTS.md)。

混合实验仅覆盖上述本步 prefill query 长度，没有测试完整 8K/16K/32K prefill；
也没有完整模型精度、H200 TP8 Serving 或 TTFT/TPOT 结果。单卡模拟每卡局部
head 数，不包含 TP 通信和真实调度，不能把算子收益直接当作 Serving 收益。

## 历史版本

| 提交 | 内容 |
| --- | --- |
| `b9efac57c3` | 混合分发实验实现 |
| `833dbc24f5` | A/B/C 对照与性能数据 |
| `f775dcc1d3` | 实际切片与 GPU 执行计数审计，保留完整混合 demo |
| `7553704323` | 恢复纯 decode 加速，移除混合路径和 demo |

当前版本不再接受混合加速开关；旧配置中的 `hy4_smallhead_mixed` 不会开启此路径。
历史数据保留用于解释决定。需要复现历史实验时，应在独立 worktree 使用上述
实验提交及其环境说明，不要在当前纯 decode 版本中执行旧混合 benchmark 命令。
