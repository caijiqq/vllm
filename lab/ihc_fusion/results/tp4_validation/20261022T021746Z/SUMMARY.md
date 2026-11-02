# HY4 iHC 三层 TP4：关键结果与开发历程

## 实验范围

2026-10-22，在 4 张 H100 80GB 上使用真实三层 HY4 ModelOpt MXFP8 切片，TP4/PP1、Eager、BF16 iHC。输入为 8192 token，隐藏维度 6144，4 路残差；关闭 Prefix Cache。计时覆盖模型 Forward（含三层、embedding 和 HCHead），不含 logits 与采样，因此不是 78 层全量模型或 Serving 指标。

## 关键性能数据

| 模式 | Forward 中位数 | 相对 Original 的中位数之比 | 每 rank 融合次数 |
| --- | ---: | ---: | ---: |
| Original | 44.143 ms | 1.0000× | 0 |
| Intra | 42.901 ms | 1.0290× | 3 |
| Cross | 41.807 ms | 1.0559× | 5 |

同一次运行完成 9 轮平衡交错测量，正式计时未开启 Profiler。Original/Cross **配对加速中位数为 1.0563×**，9 轮范围为 1.0447–1.0701×；以两组 Forward 中位数计算，延迟下降 5.29%，未达到 1.10× 加速目标。

Profiler 确认 Cross 每 rank 命中 5 次融合，包括 3 处同层和 2 处跨层边界；Original 为 0 次。五处可融合边界的 GPU 活动由 6.568–6.605 ms 降至 4.282–4.301 ms，局部约 1.54×。Original 的 iHC Pre/Post 占 Forward 内 GPU 活动的 17.96–18.09%。Profiler 只用于核实覆盖与活动组成，不用于计算上述 Forward 收益。

## 开发历程

1. 在独立 Lab 中冻结融合 kernel、原始路径对照、Golden 与算子测试，并加入三层 TP4 验证入口（`79447ce5f9`）。
2. 2026-10-22 先检查四 rank 的模式开关与预热输出，再完成 9 轮 Original/Intra/Cross 交错计时；保存全部 27 次正式测量后才采集 Profiler 证据。
3. 分析融合命中、Forward 延迟与 GPU 活动，并将完整 TP4 报告及汇总统计纳入 Git 历史（`aa72b73698`）。随后另行记录 H100 三层 Serving 验证结论（`e3e9cdee60`），其测试口径与本页离线 Forward 不混用。

## 结论与边界

TP4 下融合命中及约 5.6% 的配对 Forward 加速有实测证据；严格模型数值验收仍未闭合，历史结果为 161 passed / 1 failed，不能据此宣称精度通过。TP1 与 TP4 的设备和测量条件不同，也不能用本页绝对时延推导严格 TP 扩展效率。

详细方法与限制见 [完整 TP4 报告](REPORT.md)；可复核的汇总数据见 [计时统计](raw/timing_summary.json)和 [GPU 活动统计](raw/activity_summary.json)。
