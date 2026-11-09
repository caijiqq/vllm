# HY4 Attention 优化复盘与证据索引

归档日期：2026-11-09。最终源码基准 `7e3038e986`，生产实现恢复提交 `7553704323`；其 Backend/kernel 与 `d0665c674c` 一致。此轮归档不修改生产实现、不重新测量 GPU 性能。

## 阅读顺序

1. [当前方案](README.md)：契约、路由、正确性及目标环境复跑命令。
2. [性能与验证记录](RESULTS.md)：历史实验与最终状态分别标注。
3. [布局受控消融](ABLATION.md)：普通 Triton 与 Gluon 的同 dtype 对照。
4. [混合批次决定](MIXED_BATCH_DECISION.md)：A/B/C、实际调用计数及撤回理由。
5. [原始 JSON](results)：BF16、布局消融、混合 A/B/C、执行审计。
6. [离线历史快照](archive)：FP8、Gluon、混合实验和最终实现，各自带源码、相关测试、benchmark 与当时报告/结果。

## 方案演进

| 提交 | 方案 | 决策 |
| --- | --- | --- |
| 2a6c5bd721 | FP8 656B KV、32-key tile、32 splits；8 heads、1～8 tokens | 小 batch 1.21～1.39×；16/32 tokens回退，限制适用范围 |
| `daa54b83bb61f49954d8823a0a3197f448b6e480` | BF16 1152B KV、Gluon、64-key tile、自适应 splits | 适配参考 BF16 decode 数据流，扩大支持到8/16/32 heads；KV空间约为原FP8的1.76倍 |
| ff9abc98f7 | 同口径布局、搬运、语言与参数消融 | 普通 Triton 已能生成所需 MMA、swizzle、cp.async |
| d0665c674c | 普通 Triton，4 warps、2 stages | 生产移除 Gluon 专用布局与搬运代码，延迟基本持平 |
| b9efac57c3 至 f775dcc1d3 | 混合 batch 内拆 Decode/Prefill，同 stream 串行 | 分阶段局部收益不能覆盖所有形状的拆批成本 |
| 7553704323 / 7e3038e986 | 恢复纯 Decode，记录撤回理由 | 混合开关和生产拆分路径移除，相关回归34 passed |

## 最终适配契约

MLA absorption 后 Q/K维度576、输出latent维度512；scale沿用 Backend，本配置为0.0625。BF16 cache每token1152B，FP8模型权重与KV dtype独立。FP8 cache继续原生路径，不隐式转换。

开关 `hy4_smallhead_triton` 默认关闭。满足SM90、BF16、8/16/32 heads、PCP=DCP=1、无HiSparse、纯单token Decode、T×heads≤4096才使用新路径。纯Prefill、混合batch、多token Decode整批回退原生。用真实阶段标志排除短Prefill，不能只靠query长度猜阶段。

Split-KV根据T、SM数、K选择split数；batch越大split越少，降低FP32 partial读写。第二kernel合并，sink只加一次，返回LSE不含sink；保留无效slot、长度、空行、stride和Graph replay验证。

## 关键证据

| 对照 | 结果 | 边界 |
| --- | --- | --- |
| FP8首版 vs 原生FP8 | T=1/2/4/8 为1.34/1.39/1.30/1.21×；T=16/32为0.97/0.74× | 见FP8快照内原始JSON；不能与BF16直接比较 |
| BF16 Gluon vs 本地原生BF16 | T=1为6.98×，T=8为4.07×，T=128为1.07× | 本地原生要求8 heads pad到64；不是新版small-head原生对照 |
| 普通Triton vs Gluon | T=1～512延迟差-0.24%至+2.24% | 采用已有五轮消融结果，迁移/恢复时未重新计时 |
| 去掉异步KV搬运，T128 | 109.71→131.49 µs | 同口径独立消融 |
| 去掉shared swizzle，T128 | 109.71→274.34 µs | shared load conflict显著增加，不能仅用occupancy判断 |
| 混合A/B/C | 九组中六组净加速、三组回退 | 最快-28.08%，最慢+3.29%；不是Serving数据 |

混合A=整批原生，B=拆批全原生，C=Triton Decode+原生Prefill。8+512形状中，B−A约+58.03 µs，C−B约-49.06 µs，C−A约+8.96 µs。B−A包含形状与调度变化，不应全部归因为launch成本。计数审计确认一次索引转换、正确切片与输出共用storage，没有整批误进Triton或重复执行。

## 面试叙述与追问

可用表述：针对 HY4 TP8 每卡少量 attention heads 设计 Sparse MLA Decode 的 Triton 适配，保留物理索引、sink/LSE 与 fallback 契约；用受控消融验证数据搬运和 shared layout 的贡献，将 Gluon 实现简化为普通 Triton；通过混合批次 A/B/C 分离拆批成本和 Decode 收益，最终保留证据充分的纯 Decode 路径。

- 为什么不用FP8结果和BF16结果直接比较：KV字节数、解量化、原生kernel支持范围都改变；必须固定dtype和基线。
- 为什么小batch收益大：减少原生head padding的无效计算，并用Split-KV扩大并行度；大batch趋向带宽限制。
- 为什么普通Triton可以替代Gluon：同口径参数搜索、编译IR和性能消融显示编译器已生成需要的布局与异步搬运。
- 为什么occupancy低不等于SM没工作：它描述活跃warp比例；T128的split有256 CTA覆盖132 SM，DRAM指标约82%，不能把12% occupancy理解成12%的SM在工作。
- 为什么撤回混合优化：C在拆批结构内快于B，但部分形状无法抵消拆批成本；没有用少量样本硬拟合生产阈值。
- 还缺什么：全量模型质量、完整模型Graph、H200 TP8 Serving及联合IHC测试。单卡算子数据不包含真实TP通信与调度。

## 离线使用

```bash
sha256sum -c lab/hy4_smallhead/ARCHIVE_SHA256SUMS.txt
mkdir -p /tmp/hy4-attention-history
tar -xzf lab/hy4_smallhead/archive/attention_mixed_f775dcc1d3.tar.gz -C /tmp/hy4-attention-history
```

各包名称或manifest标注固定提交，内部带独立前缀，不依赖Git历史即可阅读被撤回的源码和demo。包是相关文件快照，不包含完整依赖/权重；实际运行需匹配版本的完整仓库与CUDA扩展。每包manifest列出提交与文件SHA256。

`archive/local_ablation_order_uncommitted.patch` 保存归档前本地未提交的执行顺序修改，原工作区文件保持原状。这份patch未用于宣称新的性能结果，也未合入本次归档提交的生产/benchmark源码。原始profiler临时trace未在本机归档源中收集到；已保存的提取计数、grid、指标和原始benchmark JSON见results，不能称所有临时文件均已归档。

IHC对应另一分组的 `lab/ihc_fusion/REVIEW.md`，分支 `feat/hy4-ihc-dev`。两条优化线独立，收益不可相加。

## 一次下载离线包

本分组的 `offline-review.tar.gz` 包含复盘文档、已归档结果、脚本和历史源码快照。下载它与 `OFFLINE_PACKAGE_SHA256SUMS.txt` 到同一目录后执行：

```bash
sha256sum -c OFFLINE_PACKAGE_SHA256SUMS.txt
mkdir -p hy4-review
tar -xzf offline-review.tar.gz -C hy4-review
cd hy4-review
```

随后执行上面的分组校验命令，并阅读本目录 REVIEW.md。清单按提交内容生成，本地另有修改时校验不匹配是预期行为；请在解压后的独立目录校验。离线包不包含自己及外部包校验文件。
