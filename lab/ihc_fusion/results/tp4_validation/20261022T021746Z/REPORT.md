# HY4 TP4 N8192 性能验证

**结论 B：TP4 放大了 iHC 占比及融合收益，但未达到 1.10×。模型严格数值验收未闭合，仅用于性能研究。**

实验：2026-10-22 UTC，HEAD `223ebbf62d906e8bb3a5c017d8ae9c3fcbd98868`，分支 `exp/hy4-ihc-lab`。4×H100 80GB（GPU1/2/3/5），三层真实 ModelOpt MXFP8 切片，TP4/PP1/mp/Eager，N8192，D6144，C4，BF16 iHC。同一模型实例、相同8192-token prompt，prefix cache关闭。

## 正式计时（Profiler关闭）

CUDA Event围住模型Forward，包含三层及embedding/HCHead，不含logits/sampling。每个样本取四rank最大值；计时前device barrier在区间外，模型本身通信保留。每模式6轮预热，切换后1轮settling，9轮平衡交错测量。完整命令和计时限制见 [REPRODUCE.md](REPRODUCE.md)。

| 指标 | Original | Intra | Cross |
|---|---:|---:|---:|
| Forward median ms | 44.143 | 42.901 | 41.807 |
| P10 / P90 ms | 44.088 / 44.443 | 42.659 / 42.996 | 41.740 / 42.025 |
| 相对Original（中位数之比） | 1.000× | 1.0290× | 1.0559× |
| 每rank Fused次数 | 0 | 3 | 5 |

Original/Cross配对中位数 **1.0563×**，P10/P90=1.0492/1.0651；9轮范围1.0447–1.0701×。中位数之比对应延迟下降 **5.29%**，不能称为10%下降。
Intra/Cross中位数之比 **1.0262×**，配对中位数 **1.0251×**，延迟下降2.55%。全部9轮Cross均快于Original和Intra。

原始逐rank、逐轮计时：[timing.csv](raw/timing.csv)；统计：[timing_summary.json](raw/timing_summary.json)。`generate_wall_ms`单独保存，不作为Forward或完整78层E2E指标。

## TP1 / TP4

| 指标 | 历史TP1 | 本次TP4 |
|---|---:|---:|
| Original ms | 78.185 | 44.143 |
| Intra ms | 76.972 | 42.901 |
| Cross ms | 76.234 | 41.807 |
| iHC Pre/Post占GPU活动 | 10.6% | 17.96–18.09% |
| 五个可融合边界占GPU活动 | 8.8% | 14.97–15.08% |
| Original/Cross 中位数比 | 1.0256× | 1.0559× |
| Intra/Cross 中位数比 | 1.0097× | 1.0262× |
| Cross每rank融合数 | 5 | 5 |

占比采用同一分母：单rank Forward内GPU activity duration之和；不是所有GPU duration求和后的wall time。TP1为历史同checkpoint、相同N/Eager数据，GPU/进程后端/同步测量条件不同，跨实验绝对耗时对比不作为严格TP scaling benchmark。TP4三模式比较在同一次运行内完成。

## Original逐rank GPU Activity（ms）

| 分类 | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| Attention | 18.780 | 18.760 | 18.896 | 18.774 |
| Dense MLP | 1.935 | 1.935 | 1.962 | 1.950 |
| Routed MoE + router/mix | 7.130 | 7.152 | 7.160 | 7.094 |
| Shared MoE | 0.562 | 0.558 | 0.557 | 0.554 |
| iHC Pre | 6.058 | 6.047 | 6.052 | 6.016 |
| iHC Post | 1.867 | 1.854 | 1.851 | 1.865 |
| HCHead | 2.961 | 2.966 | 2.972 | 2.959 |
| NCCL通信 | 3.501 | 3.497 | 3.415 | 3.656 |
| Other | 1.006 | 0.998 | 1.004 | 1.004 |
| Activity sum | 43.799 | 43.766 | 43.869 | 43.873 |
| Activity union | 43.799 | 43.765 | 43.868 | 43.872 |
| 首至末GPU activity span | 44.225 | 44.216 | 44.306 | 44.297 |
| 五边界activity | 6.605 | 6.591 | 6.586 | 6.568 |

Routed模块通过内部直接方法调用，未触发独立nn.Module hook；表中将MoE外层减去Shared和通信后的活动合并标为Routed+router/mix。原始逐kernel名称已保存，其中rank0 routed Humming两类GEMM合计5.482ms，另外包含top-k、token排序、量化、混合和router GEMM；不把这些都误称为纯GEMM。分类无重复计入。

Original每rank7次NCCL AllReduce，3.415–3.656ms，占约7.8–8.3%；GPU活动重叠仅约0.0005–0.0006ms，活动间隙约0.424–0.451ms。跨设备未建立因果关键路径图，Event最大值作为统一Forward延迟估计；union/span只描述各rank时间线。

## 融合覆盖及Profile限制

每rank实测：Original=6 Post+6 Pre1+6 Pre2；Intra=3 Fused+3 Post+3 Pre1+3 Pre2；Cross=5 Fused+1 Post+1 Pre1+1 Pre2。Cross融合所属layer顺序为0、1、1、2、2，结合调用链对应3处同层、2处跨层。首层Attention Pre和末层MLP Post保留，无静默回退。每个Fused的延迟、grid和block见activity CSV及JSON。

| Cross Fused顺序 | rank0 μs | rank1 μs | rank2 μs | rank3 μs |
|---|---:|---:|---:|---:|
| L0同层 | 863.509 | 857.591 | 857.203 | 861.240 |
| L0→L1跨层 | 859.989 | 856.311 | 857.202 | 857.624 |
| L1同层 | 862.197 | 859.799 | 857.267 | 859.320 |
| L1→L2跨层 | 855.381 | 860.151 | 855.667 | 860.024 |
| L2同层 | 859.957 | 857.848 | 855.155 | 859.416 |

五边界Original GPU活动6.568–6.605ms，Cross五Fused4.282–4.301ms，局部约1.54×；纯计时Forward节省2.336ms，与活动差值同量级。

**不将Profile wall time作为收益数据。** Cross profile首末GPU span46.85–46.93ms，高于纯计时41.807ms；rank0通信活动升至8.445ms，其他rank也出现额外等待/间隙。这一单次profile受跨rank观测扰动或调度等待影响，无法归因于融合；它只用于覆盖、单kernel时间和活动组成，不能用来宣称稳态通信增加。正式27个样本均在Profiler开启前完成。没有进行硬件counter测量。

## TP4放大机制与10%假设

- 四rank观测到完整 `[8192,4,6144]` iHC Residual及两次pending状态。iHC投影weight仍为 `[8,24576]`。
- Attention local heads=16；q_b/kv_b/o等投影、Dense gate_up/down实际按TP4分片，部分QKV/indexer投影仍复制。实际partition记录在initial_workers.json。
- MoE真实配置：`use_ep=False, tp_size=4, ep_size=1, num_local_experts=16`，每rank仍有16个experts，intermediate2048→512。是expert内部TP切分，不是把16experts分到4个rank。外层ep_group.size=4不能证明启用了EP。
- TP1 Attention活动34.835ms、MLP30.894ms；TP4 Attention约18.8ms、Dense+MoE约9.6ms，iHC仍约7.9ms（TP1约8.26ms）。加上通信约3.5ms、仍复制的HCHead约2.96ms等，Forward没有缩短到TP1的四分之一。iHC占比提高有实测活动支持。
- 通信及未分片工作使iHC相对占比低于“其余工作理想四等分”的预测；不能把所有通信时间都当作融合收益损失。稳态融合实际节省约2.34ms，放大为约1.056×，未达到1.10×。
- 以Original 44.143ms为基准，1.10×需要Cross≤40.130ms，当前还差1.677ms；10%延迟下降则需要≤39.729ms。这是条件计算，不是优化承诺。

## 正确性与范围

所有模式四rank开关一致；N8192每次只有一次Forward，warmup最终输出均有限，生成token均为93652。已知Original A/A差异、161 passed/1 failed和模型严格A/B未闭合的限制保留；本轮未研究、未修复、未放宽阈值，不宣称模型精度验收通过。

仅新增3个隔离实验脚本和本目录结果；没有修改已有Runtime/Kernel/Golden/Baseline/历史文件。SHA256和Git范围检查见raw/scope_check_final.json；新增交付文件见ARTIFACTS.txt。未提交、未推送、未切换分支。缓存仅位于本实验cache目录，保留但不作为需要审查的交付物。

原始证据：12份 `raw/{mode}_rank{rank}.trace.json`、12份activity CSV、activity_summary.json、逐模式profile观察、全部Worker开关验证、parallelism.json、版本信息及run.log。原始trace可用Chrome/Perfetto导入；不需要重跑即可复核。
