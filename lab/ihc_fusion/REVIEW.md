# HY4 IHC 优化复盘与证据索引

归档日期：2026-11-09。最终实现基准为 `4f1fe69872`，核心调优提交为 `713fb9e7f8`；历史实验工作区 HEAD 为 `da9bc79158`。归档补充文件不修改模型或 kernel，不新增 GPU 测量。

## 阅读顺序

1. 本文：问题、演进、性能与面试边界；另见[分支迁移与重构专项复盘](BRANCH_REFACTOR.md)，补齐开发分支重构前后状态。
2. [方案总结](IHC_FUSION_SUMMARY.md)：Serving 收益及实现概览，旧描述以本文勘误为准。
3. [TP4 Forward 汇总](results/tp4_validation/20261022T021746Z/SUMMARY.md)与[完整报告](results/tp4_validation/20261022T021746Z/REPORT.md)：同层和跨层对照。
4. [Serving 原始证据](results/serving_h100_tp4_9rounds/README.md)：54组 JSON、日志、实际运行脚本、配对统计与离线重算。
5. [历史材料压缩包](archive/ihc_lab_da9bc79158_working_snapshot.tar.gz)：Lab 原始结果、源码、Golden、实验脚本、性能审计和各 rank trace。
6. [历史候选 Serving harness](archive/legacy_serving_harness.tar.gz)：保留原始未提交脚本，不视为54组Serving数据的生成入口。
7. [最终源码快照](archive/ihc_final_4f1fe69872_source.tar.gz)：对应最终提交的 HY4 模型目录和模型验证入口。

## 问题与方案演进

HY4 使用4路残差。子层边界的 Post 更新、下一子层 Pre 的 gate 投影与归约、残差加权混合和 RMSNorm 原本分多次执行。优化目标是减少中间张量的显存往返和 launch，同时保留正确的 gate 所属关系。

| 阶段 | 方案 | 观察与决定 |
| --- | --- | --- |
| 原路径 | Post、Pre、RMSNorm 分离 | 建立原始算子和 Golden 双对照 |
| Single Lab | 每 token 一个 CTA，融合 Post+Pre | 大 N 收益明显，小 N 并行度不足 |
| Two-stage Lab | Post+Pre1 多 CTA 分片，复用 Pre2 合并 | 改善小 N；冷热缓存、D、dtype 改变胜者，没有采用统一切换阈值 |
| 早期 Runtime | 同层 Attention Post→MLP Pre；跨层 MLP Post→下一层 Attention Pre | 初期为 H100、Eager、纯 Prefill、N=512～8192；开发分支随后多次调整 Graph/混合资格，详见重构专项 |
| 最终 Runtime | Post+Pre+RMSNorm 单 kernel；跨层携带 pending Post | 去除独立 RMSNorm；保留 PP 边界与 HCHead 前的 Post 完成 |
| 最终调参 | N≤128 用 BLOCK_K=2048，其余1024；autotune | 小 N 改善，大 N 基本持平，停止扩大参数搜索 |

开发分支在 `f7c538cc2e` 集中重构实现与验证，随后在 `713fb9e7f8` 单独调优。上述最终状态不能代表所有中间提交。

最终代码的 autotune key 是 `(D, BLOCK_K)`，候选 `(warps, stages)` 为 `(4,1)/(4,2)/(4,3)/(8,1)`。没有将 Lab Two-stage 接入生产自动分发。

## 关键结果与统计口径

| 证据 | 条件 | 结果 |
| --- | --- | --- |
| 中间算子 | H100、BF16、D6144、N8192、冷缓存 | Original 1299.924 µs → Single 848.939 µs，约1.53× |
| 三层 Forward | H100 TP4、真实三层 checkpoint、N8192、Eager | Original 44.143 ms；Intra 42.901 ms；Cross 41.807 ms |
| Forward 配对 | 9轮交错；不含 logits/采样 | 加速中位数1.0563×；按耗时中位数之比，延迟下降5.29% |
| Serving | H100 TP4、三层 FP8 checkpoint、Eager、并发4、输出512、关闭 Prefix Cache | 下表为9轮配对变化的中位数 |

| Serving 指标 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| TTFT P50 | -0.90% | -4.66% | -12.75% |
| TTFT P90 | -4.35% | -6.59% | -5.19% |
| TTFT P99 | -9.19% | -4.06% | -6.56% |
| TPOT P50 | -4.59% | -3.68% | -1.94% |
| 输出吞吐 | +3.95% | +3.54% | +1.77% |

Serving 为54组、3456条正式请求；每组预热16条、测量64条。16K/32K吞吐9/9轮改善；8K TTFT P50仅6/9轮改善；32K TPOT P99约上升1.17%。配对变化中位数不等于两组中位数的比值。

## 代码与旧文档的勘误

以最终源码快照及仓库 `vllm/models/hy_v4/nvidia/model.py`、`triton_ihc_post_pre.py` 为准：

- README 的 `(N,D)` autotune 描述过时，实际为 `(D,BLOCK_K)`。
- 最终代码移除了 enforce_eager、纯 Prefill、N 范围、compile、padding/ubatch、Graph mode 的显式门控。总结中的相关排除描述不能作为最终代码事实；路由放宽也不代表完整模型 Graph 已验收。
- 仍要求启用 iHC 与 PostPre 开关、D6144/C4、CUDA SM90、BF16 activation、相关 FP32 参数及连续同设备张量；排除 speculative config 与 batch-invariant。
- PostPre 开关允许同层融合；cross-layer 开关进一步允许跨层 pending 状态，不是所有融合都必须两个开关同时为真。
- 最终启用浮点融合并在 FP32 累加结果上直接计算 RMSNorm，改变舍入细节。数学公式保留不等于逐 bit 等价。
- 当前测试打印 max_abs/cosine，不设置自动通过阈值。历史严格模型验收保留161 passed / 1 failed，不能写成模型精度全部通过。

## 面试叙述与追问

可用表述：在 vLLM HY4 中用 Triton 实现同层及跨层 Post+Pre+RMSNorm 融合，处理 gate 归属、延迟 Post、PP 边界及 fallback；通过算子、三层 Forward 和交错 Serving 分层验证。H100 TP4 三层模型、Eager、并发4、32K输入/512输出的9轮测试中，TTFT P50下降约12.75%，吞吐提升约1.77%。

- 为什么跨层融合难：上一子层的 Post gate 与下一子层 Pre 参数不同；需要携带 residual/gates，首层和末层、PP 传输、HCHead 都必须正确完成状态。
- 为什么单 kernel 小 N 可能慢：CTA 数少且每 token 内部串行归约较长；Two-stage 增加并行度，但也引入 partial 读写和第二次 launch。
- 为什么算子1.53×而 Forward只有约1.056×：优化仅覆盖部分执行时间，Attention、MoE、通信等仍存在；不能直接推广算子加速比。
- 如何排除实验顺序偏差：每轮交换 A/B 顺序，先预热，单独采 profiler，使用同轮配对统计，并报告 P50/P90/P99 和回退。
- 精度闭环做到哪里：有算子和三层 hidden-state 对照；全量模型质量及严格验收尚待完成，不将请求成功率当作生成质量。

## 离线归档与复现

从仓库根目录校验本分组：

```bash
sha256sum -c lab/ihc_fusion/ARCHIVE_SHA256SUMS.txt
.venv/bin/python lab/ihc_fusion/results/serving_h100_tp4_9rounds/recompute.py
mkdir -p /tmp/hy4-ihc-history
 tar -xzf lab/ihc_fusion/archive/ihc_lab_da9bc79158_working_snapshot.tar.gz -C /tmp/hy4-ihc-history
```

压缩包保留历史相对路径，建议解压到空目录，不覆盖当前实现。每个包的 `.manifest.json` 记录成员 SHA256 和来源；Lab 包是工作区快照，包含未跟踪证据，不冒充 `da9bc79158` 的纯提交内容。`ihc_lab_status.txt` 和 `ihc_lab_tracked.diff` 记录其状态。清单列明排除的环境、编译缓存和临时 PID；正式 profiler PTX/SASS/cubin 与 trace 保留。

历史文档中的机器绝对路径、权重位置和旧命令保留为当时记录；不能原样假定适用于其他机器。完整 GPU 复跑需要匹配扩展、checkpoint 和硬件；离线阅读、解压、校验和 Serving 统计重算不需要 GPU。

## 尚待验证的项目

H200 全量78层质量与 Serving、完整模型 Graph、32K TPOT 尾延迟，以及 IHC+Attention 联合 A/B 尚未完成。两条分支没有合并，不能把两者收益相加。Attention 复盘位于另一个下载分组的 `lab/hy4_smallhead/REVIEW.md`，远端分支为 `feat/hy4-smallhead-triton`。

## 一次下载离线包

本分组的 `offline-review.tar.gz` 包含复盘文档、已归档结果、脚本和历史源码快照。下载它与 `OFFLINE_PACKAGE_SHA256SUMS.txt` 到同一目录后执行：

```bash
sha256sum -c OFFLINE_PACKAGE_SHA256SUMS.txt
mkdir -p hy4-review
tar -xzf offline-review.tar.gz -C hy4-review
cd hy4-review
```

随后执行上面的分组校验命令，并阅读本目录 REVIEW.md。清单按提交内容生成，本地另有修改时校验不匹配是预期行为；请在解压后的独立目录校验。离线包不包含自己及外部包校验文件。
