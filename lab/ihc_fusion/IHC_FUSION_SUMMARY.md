# HY4 iHC 融合方案总结

## 当前结论

当前 iHC 融合方案可以作为一个阶段性版本固定下来。H100 TP4 的交错 Serving 测试已经证明，融合收益能够从算子和 Forward 层面传递到真实请求；继续调整 Triton 的 warp、stage 和归约块参数，暂时没有为大输入带来额外收益。后续工作应以 H200 全量模型验证和数据集正确性评测为主，不再扩大 iHC 的生产改动范围。

这里的“固定”是指冻结当前实现和测试口径，不代表已经证明所有硬件、所有批次和 CUDA Graph 都有收益。当前最强的结论范围是：H100、TP4、三层 HY4 checkpoint、Eager、并发4、输出512、8K/16K/32K输入。

## 融合解决的问题

HY4 的 iHC 使用4个残差通道。每个子层边界包含以下逻辑：

1. 将上一阶段产生的残差和 Post gate 合成为4个更新后的残差通道。
2. 对4个通道做加权混合，得到下一层子模块的输入。
3. 计算下一组 Pre gate 和 Post gate。
4. 对混合结果执行下一层 RMSNorm。

原始路径会把这些步骤拆成多个 Python、PyTorch 或 Triton 操作，并把中间残差、gate 和混合结果写回显存，再由后续操作读回。跨层融合把“上一层 Post + 下一层 Pre + 下一层 RMSNorm”放进一个 Triton kernel，使中间结果尽量留在一次 kernel 执行的寄存器或片上缓存路径中，同时减少 kernel launch 和中间 Tensor 的全局内存往返。

融合不改变 iHC 数学语义。Kernel 仍然按照原有顺序完成残差更新、gate 投影、归一化、Pre/ Post gate 变换和输出写回；它只是改变这些步骤的执行边界。

## 当前执行路径

生产入口位于 [model.py](../../vllm/models/hy_v4/nvidia/model.py)。`HYV4DecoderLayer` 在初始化时读取两个配置字段：

```json
{
  "enable_ihc_post_pre_fusion": true,
  "enable_ihc_cross_layer_fusion": true
}
```

两个字段都需要为 `true`。模型本身还必须启用 iHC。`_can_fuse_ihc` 保留了生产所需的张量契约：

- CUDA、SM90、BF16 activation；
- `hidden_size=6144`、`hc_mult=4`；
- 输入、残差、gate 和权重的形状与 stride 正确；
- 相关 Tensor contiguous 且位于同一设备；
- gate 和 iHC 参数为 FP32；
- 没有 speculative config、batch invariant 模式或正在编译的路径。

满足条件时，`_ihc_post_pre` 调用 [triton_ihc_post_pre.py](../../vllm/models/hy_v4/nvidia/triton_ihc_post_pre.py) 中的 `triton_ihc_post_pre`。不满足条件时，继续走原有 Post、Pre 和 RMSNorm 路径。这个 fallback 保证了 padding、ubatch、特殊编译状态以及不支持的模型形状不会误用融合 kernel。

当前 kernel 的主要流程是：

```text
previous_post + x + residual
        │
        ├── 更新 4 路 residual
        ├── iHC gate 投影与归约
        ├── Pre/Post gate 变换
        ├── 4 路 residual 混合
        └── 下一层 RMSNorm
                │
             output
```

## Kernel 调优

Triton autotune 当前按 `(D, BLOCK_K)` 选择配置，候选包括4/8 warps和1/2/3 stages。`BLOCK_K` 对小批次使用2048，对较大批次使用1024；这是当前实现中唯一的简单 shape-aware 选择。RMSNorm 已包含在跨层 kernel 内，避免再启动独立的 RMSNorm 操作。

单 GPU 的 graph replay 测量显示，小 N 的融合 kernel 相比此前融合版本明显改善：N=1、4、64、128 的热缓存耗时约下降16%～17%，冷缓存耗时约下降30%～37%。但 N=1/4 的融合 kernel 仍没有稳定超过真实 vLLM 原生 RMSNorm 对照，因此没有把“小 N 全面领先”作为方案目标。

对 N=8192～32768，当前参数调整相对之前融合版本基本持平。这说明继续搜索 warp 或 stage 的边际收益已经较低；大输入的主要收益来自融合边界本身，而不是继续堆叠更多 autotune 配置。

## H100 Serving 证据

测试结果保存在 `/tmp/hy4-ihc-rerun-JOWKo8/paired_analysis.json`。测试配置如下：

- H100 TP4；
- 三层 HY4 FP8 checkpoint；
- Eager，关闭 Prefix Cache；
- 并发4，输入长度8K、16K、32K；
- 输出512 Token；
- 9轮交错，每轮交换 Original/Fused 先后顺序；
- 每组预热16条、正式测量64条；
- 54组结果、3456条正式请求全部成功。

每项数据先在同一轮、同一输入长度内计算 Fused 相对 Original 的变化，再取9轮中位数：

| 指标 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| TTFT P50 | -0.90%（6/9轮改善） | -4.66%（8/9） | **-12.75%（9/9）** |
| TTFT P90 | -4.35%（7/9） | **-6.59%（9/9）** | -5.19%（9/9） |
| TTFT P99 | -9.19%（7/9） | -4.06%（8/9） | **-6.56%（9/9）** |
| TPOT P50 | -4.59%（8/9） | -3.68%（9/9） | -1.94%（9/9） |
| 输出吞吐 | **+3.95%（8/9）** | **+3.54%（9/9）** | **+1.77%（9/9）** |

这组结果支持以下判断：

- 16K 和32K的吞吐在每一轮都提升；
- 32K 的 TTFT P50、P90、P99 每一轮都下降；
- 8K 的 TTFT P50 波动较大，不能宣称稳定改善；
- 32K 的 TPOT P99 中位数约上升1.17%，存在轻微尾延迟代价；
- 交错顺序后收益仍存在，不能简单归因于“某个模式总是后运行”。

因此，当前证据足以支持“在已测 H100 Serving 场景中融合有效”，还不足以支持“所有场景全面加速”。

## 正确性证据与边界

测试入口位于 [test_hyv4_ihc_fusion.py](../../tests/models/test_hyv4_ihc_fusion.py)。当前测试只输出最大绝对误差和余弦相似度，不设置自动通过阈值：

- 单算子测试比较融合 Post+Pre+RMSNorm 与原始路径的输出；
- 演示模型测试比较每层 hidden state；
- 全量模型入口通过 vLLM worker extension 读取每层最后一个 prompt Token 的 hidden state；
- 全量入口支持通过环境变量指定 checkpoint 和 TP 大小。

已有真实 checkpoint 的 TP2 演示结果为三层模型、输入128，逐层 cosine 约 `0.999956～0.999966`，最大绝对误差约 `0.00488～0.00537`。这证明真实 Runtime 路径能够运行并产生接近的数值结果，但它不是78层全量模型的最终验收，也没有覆盖所有 Token、所有 rank 和数据集生成质量。

H200 上仍需重新执行全量 hidden state 检查和 Serving 测试。H100 的 SM90 代码条件适用于 H200，但 H100 的性能数字不能直接当作 H200 性能承诺。

## 方案评价

从工程角度看，这个方案已经达到“可以固定并进入验证收尾”的程度：生产改动集中在 HY4 专属 Runtime 和一个 Triton kernel，保留原路径 fallback，沒有改公共 Scheduler、Attention Backend 或 CUDA Graph 管理逻辑。当前分支最新提交为 `713fb9e7f8`，本阶段相对 HEAD 的最后一轮改动规模为两个核心文件增加19行、删除13行；测试的类型声明和 CI 手动测试登记随后又增加了少量辅助改动。

从性能角度看，收益来自正确的融合边界，而不是只依靠人工挑选一个最快数字。大输入的 TTFT 和吞吐收益在交错测试中重复出现，尤其32K场景证据较强。小 N 仍有单 kernel 级别的并行度问题，但它没有在本次 H100 Serving 中造成整体吞吐回退。

从研究和工程简历角度看，项目有竞争力，理由是它覆盖了完整链路：

1. 识别 iHC 的跨层数据依赖，并设计 Post+Pre 融合边界；
2. 用 Triton 实现融合 kernel，并把 RMSNorm 纳入同一次执行；
3. 处理 Runtime dispatch、fallback、autotune、SM90 约束和真实 Serving；
4. 通过交错实验纠正“算子更快就必然 Serving 更快”的错误假设；
5. 用 P50/P90/P99、TPOT 和吞吐同时评估收益与尾延迟代价。

简历上可以写成：

> 基于 Triton 和 vLLM 实现 HY4 iHC 跨层 Post+Pre 融合，将 iHC gate 投影、4路残差更新、下一层 Pre 和 RMSNorm 合并到单 kernel；加入按形状选择的 autotune 配置和 SM90 Runtime dispatch。H100 TP4、三层 HY4 FP8、Eager、并发4、8K/16K/32K输入的9轮交错 Serving 测试中，32K TTFT P50/P90/P99 分别下降约12.8%/5.2%/6.6%，输出吞吐提升约1.8%；16K输出吞吐提升约3.5%。

简历表述必须保留测试范围，不能写成“全量78层、H200、所有请求场景稳定提升”。如果 H200 全量模型后续通过同样的交错测试，可以再把硬件、层数和 TP 配置更新进去。

## 建议冻结范围

当前建议冻结：

- 跨层 Post+Pre+RMSNorm kernel；
- 现有 `(D, BLOCK_K)` autotune 和小 N 配置；
- 当前 Runtime fallback 规则；
- 单算子、逐层 hidden state 和 vLLM benchmark 三类验证入口。

后续只做两件事：

1. 在 H200 上验证全量模型的数值和 Serving 结果；
2. 如果业务明确要求，单独调查32K TPOT P99尾延迟。

不建议在没有新 Trace 证据前继续增加 kernel 配置、改公共 Scheduler、适配 CUDA Graph 或重写 MoE。那些属于新的优化项目，应与当前 iHC 方案分开评估。
