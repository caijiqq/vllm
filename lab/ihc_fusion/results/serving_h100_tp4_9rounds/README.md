# H100 TP4 九轮 Serving 原始记录

本目录从历史 `/tmp/hy4-ihc-rerun-JOWKo8` 复制归档，保留54组官方 benchmark JSON、日志、实际 `run.sh` 与 `paired_analysis.json`。本次只重算统计，没有重新运行模型。文件级 SHA256 见分组根目录清单。

## 证据对应

- `r0` 至 `r8`：每轮 original/fused 两个模式；各含8192、16384、32768输入结果。
- `n*.json`：每组64条成功请求、0失败、输出512；汇总指标与 token 总数可校验。这些文件不是每条请求的完整原始时间序列。
- `server.log.gz`、`n*.log.gz`：启动、加载及 benchmark 运行记录。
- `run.sh`：实际测试脚本，含当时 checkpoint 和 bootstrap 绝对路径；用于审阅测试口径，迁移机器前需要改路径。
- `paired_analysis.json`：历史配对分析。
- `recompute.py`、`recomputed_summary.json`：归档时新增的纯标准库重算入口及结果，逐项比较9轮配对变化、胜出次数和54组请求计数。

从仓库根目录运行 `.venv/bin/python lab/ihc_fusion/results/serving_h100_tp4_9rounds/recompute.py`，无需 GPU。输出应包含 `completed_requests=3456` 和 `matches_archived_analysis=true`。

## 口径

H100 TP4、真实三层 HY4 FP8 checkpoint、Eager、关闭 Prefix Cache、并发4。每组预热16条、测量64条。偶数轮 Original→Fused，奇数轮相反。每轮先计算 `(fused/original-1)*100`，再取9轮中位数；延迟越低越好，吞吐越高越好。

`archive/legacy_serving_harness.tar.gz` 中的 `benchmarks/serve_ab.sh` 与 `serve_ab_results.py` 是另行保留的历史候选 harness，采用不同目录结构及部分不同指标/聚合规则，不是本目录54组结果的生产脚本，也不能用于替代本次配对中位数重算。复现本次运行应以本目录 `run.sh` 为口径依据。

编译缓存和进程 PID 未归档。原始文件未重写机器路径；checkpoint 权重和本机环境不包含在离线证据包中。记录支持三层测试范围，不支持宣称 H200/78层或所有请求场景均已验证。

日志使用 gzip 无损压缩；`compressed_logs_manifest.json` 记录解压后原文的大小与SHA256，可用 `gzip -dc` 阅读。压缩避免对随机请求ID执行拼写修正。
