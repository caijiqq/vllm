# HY4 iHC：Numerical / Runtime / Performance

默认验证只覆盖生产 BF16 / D6144 / C4。独立 PyTorch Golden、固定官方
Post + Pre1 + Pre2 Baseline、数值阈值和 Kernel 均保持原样。生产 Runtime 测试
不依赖 Lab；Lab 复用生产测试中原有的数据生成和数值验收 helper，避免维护两份规则。

## 本地算子与 Runtime

从仓库根目录运行，选择空闲 GPU 和新的输出目录：

```bash
export HY4_TP8_GPUS=1
export HY4_TP8_RUN=/tmp/hy4-local
bash lab/ihc_fusion/benchmarks/tp8_validation/run.sh validate.py --suite local
```

此入口按文件隔离执行现有 pytest，输出日志、JUnit、数值和实际 GPU launch 记录：

- Numerical：N128/513/8192；zero/tiny/large/cancel/saturated；Golden 与官方对照；
  预分配 Baseline 等价性；生产单 Kernel 与官方三 Kernel 的真实 launch。
- Runtime：路由上下界、dtype/layout/metadata 回退、实际 Dispatch、Pending 消费、
  首末层；N128/512/8192 三层 Original/Intra/Cross。Intra/Cross 独立验收。
- Graph：FULL N512、breakable N1024，分别有/无捕获 metadata；同图更新输入，
  回放 Prefill/Mixed/Padding/Decode，要求 ON/OFF 逐位一致且 GPU trace 无融合调用。
- CPU：快照有效性、严格失败保留、配置与配对 Rank-MAX 统计。

可直接使用 pytest 文件路径和 `-k` 选择子集，无新增配置框架。生产文件为
`tests/models/test_hyv4_ihc_fusion.py`，Golden 数值文件为
`lab/ihc_fusion/tests/test_post_pre.py`，CPU 检查为 `test_prefill_workload.py`。
实验性 Two-Stage 不进入默认集合；需要时显式运行
`pytest lab/ihc_fusion/tests/check_decode_two_stage.py`。通用 Shape/dtype 扩展矩阵
已移出日常验证；这不意味着所有 Lab 扩展均已验证。

## H200 完整权重 TP8

以下命令直接使用同一 Harness，不固定设备型号、checkpoint 或层数。
`run.sh` 检查选中 GPU 空闲，不清理其他进程。部署环境需安装匹配的 vLLM/CUDA
依赖；当前 checkout 的 `bootstrap/sitecustomize.py` 复用已安装 wheel 的扩展。

```bash
export HY4_TP8_GPUS=0,1,2,3,4,5,6,7  # 改成实际分配且空闲的 GPU
export HY4_TP8_CHECKPOINT=/path/to/full-HY4-checkpoint
export HY4_TP8_RUN=/path/to/results/hy4-tp8
H=lab/ihc_fusion/benchmarks/tp8_validation/run.sh

# Numerical：同实例 Original A/A + Original/Intra/Cross Hidden State A/B。
bash "$H" validate.py --tp-size 8 --shapes 128 512 8192 --aa-repeats 2 --warmup 2

# Runtime + E2E：每 rank 实际 Kernel trace 在计时结束后采集；无正确性 Gate。
bash "$H" benchmark.py --tp-size 8 --tokens 8192 --warmup 6 --rounds 9 --profile

# Serving Eager：原生 vllm bench serve 请求、流式指标和详细 JSON。
bash "$H" serving.py --tp-size 8 --host 127.0.0.1 --port 18080 \
  --execution eager --input-len 8192 --output-len 128 --concurrency 1 4 \
  --request-count 16 --request-rate inf --warmup 2 --rounds 3

# 完整模型 Graph 启动/捕获/请求 smoke；当前 Runtime 全部回退 Original。
bash "$H" serving.py --tp-size 8 --host 127.0.0.1 --port 18080 \
  --execution graph --input-len 512 --output-len 8 --concurrency 1 \
  --request-count 2 --warmup 1 --rounds 1
```

不同阶段写入 `validation/`、`performance/`、`serving/{eager,graph}/`。
同阶段再次运行需新的 `HY4_TP8_RUN`。TP 默认取 GPU 数；`--modes original intra cross`
可选择以 Original 为基准的子集。模型设置只通过 `--llm-overrides JSON` 覆盖，
例如 `'{"gpu_memory_utilization":0.85,"max_num_seqs":8}'`，TP/checkpoint 使用专用参数。
H200 全模型必须根据实际权重占用设置内存比例，以上命令不代表已完成 H200 验收。

模型数值采样限定 Eager/PP1，一次新鲜、完整、无 Padding 的 Forward。
快照编号、有效输入/Positions/Query 区间和参考生命周期必须匹配；陈旧、分块、
多次 Forward 或 Graph hook 结果记为 INVALID。物理 KV 地址不同不等于逻辑输入不同。
JSON 保留 PASS/FAIL/INVALID、逐位一致性、Max Abs、Relative L2、Cosine 和 ULP。
`--aa-repeats 6 --modes original` 可按需复现 A/A；不默认扩展逐层诊断。

Forward 计时仍为各 rank 的 CUDA event 时间逐轮取 MAX，再统计配对收益。
Barrier、模型初始化、生成输入和 profile 不计入该区间；另外保存 generate wall time。
这与 `vllm bench latency/throughput` 的指标不同，不互相替代。
`--profile` 只保留 Forward 范围和实际 Kernel 计数，删除模块分解/重叠统计。
未开启时明确 `kernel_hit_verified=false`。结果自动打印并保存原始 `timing.csv`、
配置和 `timing_summary.json`。历史 TP4/TP8 原始计时及 trace 仍可离线解析：

```bash
.venv/bin/python -S lab/ihc_fusion/benchmarks/tp8_validation/analyze_perf.py \
  /path/to/saved/performance --output-dir /tmp/hy4-analysis
```

Serving 复用 `RemoteOpenAIServer` 的启动等待和自有进程清理，只覆盖解释器启动方式。
请求与指标完全交给 `vllm bench serve`：TTFT/TPOT P50/P90、请求吞吐、Output tokens/s、
成功率、各轮相对加速比。每模式使用固定 seed、等长随机请求、greedy、ignore_eos，
保存实际启动参数、原始详细 JSON、`summary.json/csv`。失败、长度不足、模式缺失等
不产生正常性能结论。TPOT 为 `(latency - ttft)/(output_tokens - 1)`，ITL 是流式事件
间隔；不保证一个事件只有一个 Token。Graph 只报告 Original，不计算融合加速比。
启动开关不等于 Serving 实际 Kernel 命中，Mixed/小 N 仍可能回退。

## 一个 Kernel 微基准

旧 `bench_decode_two_stage.py`、`bench_prefill_selection.py` 合并到以下入口；
删除 `audit_baseline_scope.py`、`audit_postpre_timing.py`，保留其全部历史结果。
预分配、官方三次 launch、25ms Graph 预热和 FlashInfer 冷 L2 轮转口径保持不变。
CUPTI 不可用时采用完整轮转周期的 Graph/Event fallback。hot 单独使用 32 次 Graph
调用，与 cold 分开记录；不会混合比较。Cross 只在 Runtime/模型级测量。

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/lab/ihc_fusion/benchmarks/tp8_validation/bootstrap:$PWD"
.venv/bin/python -m lab.ihc_fusion.benchmarks.bench_post_pre \
  --tokens 128,513,8192 --hidden 6144 --dtypes bfloat16 \
  --magnitude 2 --hc-eps 1e-6 --warmup 30 --rounds 3 --repeat-ms 100 \
  --output /tmp/hy4-kernel.json
```

`--two-stage --two-warps 8 --cache both` 对应原三臂 Prefill 实验；
`--hidden 4096,7168 --tokens 1,8,32,128 --two-stage` 对应原 Decode 实验。
保留 `--warps`、dtype/shape、warmup/rounds/repeat-ms 配置，不默认进行参数搜索。
默认 magnitude=1.7、hc_eps=2e-6 沿用原主微基准；上述生产参数显式覆盖，JSON 保存配置。
每个计时臂必须先通过冻结 Golden/Official 比较；保存 FlashInfer 原始 samples 和汇总。
没有删改 `kernels/`、`reference/`、`baselines/` 中的实现。

## 尚未闭环的数值验收

Humming MXFP8 Stream-K 的模型级 Original A/A 差异仍未解决；固定输入重放中关闭
Stream-K 曾恢复逐位一致，但本 Harness 不强制更改模型计算。不能将所有 A/B 差异
归因于融合。N8192 三层链的严格数值失败继续返回 FAIL；没有 skip/xfail 或放宽阈值。
算子 Golden 通过、Runtime 正常或性能正收益都不等于全模型严格数值通过。
历史证据位于 `results/`；已有 `decode_profile.py` 保留为按需工具，非默认流程。
`H200_TP8_MANUAL_TEST.md`、`RUNTIME_GRAPH_VALIDATION.md` 为历史记录，当前命令以本页为准。
