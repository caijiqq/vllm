# HY4 iHC Decode Two-Stage — H100 实验报告

结论：1024 元素分片、8 warps 的 Post+Pre1 真融合解决了测量范围内小 N 的退化。N=1/8/32、三个 dtype 均超过官方三阶段；保留单 Kernel Prefill，不实现 Runtime 或 Lab 自动 Dispatch。

## 实现与范围

仅新增 `kernels/decode_two_stage.py`（125 行，含约 40 行第一阶段 Triton）、`tests/test_decode_two_stage.py`（60 行）、`benchmarks/bench_decode_two_stage.py`（239 行）及本目录结果。实际行数以文件为准。运行时代码集中在一个文件。原 Golden、Baseline、单 Kernel、测试预期、容差和历史结果均未改动。`before.json`、`scope.json` 保存原文件哈希与审计；`implementation.patch` 保存三个新源码文件的完整补丁。

第一阶段 Grid `[N, ceil(4D/1024)]`，8 warps。D=4096/7168 分别有 16/28 CTA/token。输入 x[N,D]、R[N,4,D]（FP32/BF16/FP16），previous_post[N,4]、W[8,4D]、scale[2]、base[8] 为 FP32。每个 Residual 元素只有一个 CTA 写入。

Partials 是 FP32 `[N, splits, 9]`，顺序为平方和、4 路 Pre 投影、4 路 Post 投影。第二阶段直接调用冻结官方 `_ihc_pre_stage2`，Grid `[N, ceil(D/1024)]`，4 warps；分别 4/7 CTA/token。返回单流 [N,D]、更新 Residual [N,4,D]、FP32 下一子块 Post Gates [N,4]。两个 Kernel 使用同一 CUDA stream 的执行顺序完成依赖，无跨 CTA 自旋、原子完成计数或额外同步 Kernel。

保持 `r32=previous_post*x+R` 的 FP32 分离乘加（enable_fp_fusion=False）；先 cast 到输入 dtype，再转 FP32 计算投影/平方和。RMS 使用 sumsq/(4D)+norm_eps；Scale、Base、Magnitude、Pre/Post 的 additive hc_eps、顺序通道归约完全由官方 Pre2 执行。不包含 mHC comb/Sinkhorn。FP32 是已有 Lab 直接调用扩展，并非宣称官方 Runtime 支持 FP32。

参考的是当前仓库 mHC 的多 CTA 投影和 Kernel 边界归约思路，没有移植其 GB300 阈值或未舍入 FP32 residual 的投影语义。

## 正确性

最终 `pytest_final.txt`：99 passed（原 53 + 新 46），无失败、无警告。新测试覆盖三 dtype、N=1/8/32/128、D=4096/7168、非整 tile D=257、offset view、零/tiny/large/cancel/saturated、N=0、改变输入后多次 CUDA Graph Replay。Projection/Square Partials 与官方逐位相等；与 Golden、官方、单 Kernel 使用原误差规则比较。独立 Profiler 测试确认仅两个 Launch；源码 Grid 和逐元素部分和比较确认分片布局。

36 组默认完整 Benchmark 均先检查 Golden 和官方；三项输出对官方 max_abs 全为 0。对 Golden 的最大误差（跨对应 dtype 全部 Shape，非单案例）：

| dtype | output max_abs | output relative L2 | gates max_abs | gates relative L2 |
|---|---:|---:|---:|---:|
| bfloat16 | 0.0625 | 2.7991009e-05 | 9.5367432e-07 | 2.1749912e-07 |
| float16 | 0.0078125 | 1.0650575e-05 | 8.9406967e-07 | 2.1655528e-07 |
| float32 | 8.5830688e-06 | 2.2124395e-07 | 8.9406967e-07 | 2.1422892e-07 |

Residual 对 Golden 全部逐位一致。低精度 output max_abs 包含最终 cast 边界差异；没有扩大任何容差。

## 测量方法与完整数据

H100 80GB HBM3，物理 GPU 3；132 SM；driver 575.57.08。PyTorch 2.13.0+cu129，Triton 3.7.1，CUDA 12.9。当前仓库 commit `4fc3e26c5430fb0b12e887370a731f52819c5d2c`，官方源码固定 `5df63a7203cdb9328d824a27b682e77c7b9928be`，实际导入文件及哈希在 JSON metadata。GPU UUID、环境命令、源码哈希均已记录。

复用原 `time_call`：FlashInfer CUDA Graph + CUDA events，冷 L2 rotating buffers；30 次预热、25ms graph warmup、每轮 100ms、3 轮，取各轮中位数的中位数，交替执行顺序。每图覆盖完整轮换且至少 32 次调用，排除分配/JIT。Two-Stage 总耗时同时包含两个 Kernel。热缓存另测相同图方法、关闭 rotation。未使用缺失的 CUPTI Python 包、ncu/nsys；不报告硬件 stall 百分比。

以下为默认 8-warps 冷缓存完整路径，单位 μs。Speedup 均为官方 / 对应候选。原始每轮 p10/p90、误差、热缓存、阶段采样和资源见 `h100_final.json`，便于分析的全量表见 `h100_final.csv`。

| dtype | N | D | 官方 | Single | Two-stage | Single speedup | Two speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| bfloat16 | 1 | 4096 | 8.933 | 13.966 | 5.761 | 0.640 | 1.551 |
| bfloat16 | 8 | 4096 | 9.339 | 13.827 | 5.947 | 0.675 | 1.570 |
| bfloat16 | 32 | 4096 | 10.693 | 14.308 | 7.320 | 0.747 | 1.461 |
| bfloat16 | 128 | 4096 | 19.370 | 15.704 | 15.862 | 1.233 | 1.221 |
| bfloat16 | 1024 | 4096 | 106.014 | 78.520 | 91.225 | 1.350 | 1.162 |
| bfloat16 | 4096 | 4096 | 431.479 | 290.070 | 356.356 | 1.488 | 1.211 |
| bfloat16 | 1 | 7168 | 9.219 | 31.451 | 6.420 | 0.293 | 1.436 |
| bfloat16 | 8 | 7168 | 9.635 | 29.578 | 6.578 | 0.326 | 1.465 |
| bfloat16 | 32 | 7168 | 12.699 | 29.180 | 9.754 | 0.435 | 1.302 |
| bfloat16 | 128 | 7168 | 28.910 | 32.430 | 24.032 | 0.891 | 1.203 |
| bfloat16 | 1024 | 7168 | 188.631 | 141.462 | 160.016 | 1.333 | 1.179 |
| bfloat16 | 4096 | 7168 | 758.532 | 523.444 | 623.655 | 1.449 | 1.216 |
| float32 | 1 | 4096 | 9.280 | 13.659 | 6.000 | 0.679 | 1.547 |
| float32 | 8 | 4096 | 9.843 | 13.749 | 6.268 | 0.716 | 1.570 |
| float32 | 32 | 4096 | 11.581 | 14.704 | 8.254 | 0.788 | 1.403 |
| float32 | 128 | 4096 | 23.125 | 17.902 | 18.761 | 1.292 | 1.233 |
| float32 | 1024 | 4096 | 149.831 | 130.101 | 118.561 | 1.152 | 1.264 |
| float32 | 4096 | 4096 | 593.458 | 497.542 | 437.862 | 1.193 | 1.355 |
| float32 | 1 | 7168 | 9.271 | 27.053 | 6.285 | 0.343 | 1.475 |
| float32 | 8 | 7168 | 9.950 | 26.995 | 6.706 | 0.369 | 1.484 |
| float32 | 32 | 7168 | 13.977 | 28.313 | 11.058 | 0.494 | 1.264 |
| float32 | 128 | 7168 | 34.914 | 33.933 | 28.354 | 1.029 | 1.231 |
| float32 | 1024 | 7168 | 260.889 | 250.198 | 199.440 | 1.043 | 1.308 |
| float32 | 4096 | 7168 | 1040.057 | 997.136 | 760.368 | 1.043 | 1.368 |
| float16 | 1 | 4096 | 8.946 | 13.554 | 5.766 | 0.660 | 1.552 |
| float16 | 8 | 4096 | 9.366 | 13.409 | 5.938 | 0.698 | 1.577 |
| float16 | 32 | 4096 | 10.658 | 13.843 | 7.307 | 0.770 | 1.458 |
| float16 | 128 | 4096 | 19.391 | 15.258 | 15.851 | 1.271 | 1.223 |
| float16 | 1024 | 4096 | 106.108 | 76.475 | 91.074 | 1.387 | 1.165 |
| float16 | 4096 | 4096 | 431.447 | 280.037 | 355.873 | 1.541 | 1.212 |
| float16 | 1 | 7168 | 9.144 | 30.938 | 6.343 | 0.296 | 1.442 |
| float16 | 8 | 7168 | 9.622 | 29.009 | 6.552 | 0.332 | 1.468 |
| float16 | 32 | 7168 | 12.664 | 28.568 | 9.706 | 0.443 | 1.305 |
| float16 | 128 | 7168 | 28.768 | 31.769 | 23.879 | 0.906 | 1.205 |
| float16 | 1024 | 7168 | 188.596 | 136.826 | 159.735 | 1.378 | 1.181 |
| float16 | 4096 | 7168 | 758.505 | 507.840 | 622.962 | 1.494 | 1.218 |

## 阶段时间、缓存与资源

下表为 BF16 N=1；阶段时间来自独立热缓存 Profiler，不能与冷缓存完整路径相加比较。

| D | 官方冷/热 | Single 冷/热 | Two 冷/热 | Stage1 热 trace | Stage2 热 trace |
|---|---:|---:|---:|---:|---:|
| 4096 | 8.933/6.275 | 13.966/10.277 | 5.761/4.221 | 2.176 | 1.760 |
| 7168 | 9.219/6.368 | 31.451/19.768 | 6.420/4.684 | 2.208 | 2.080 |

第一阶段 8 warps：29–30 registers/thread、128 B shared、spill=0；PTX local load/store=0、FP32 FMA=0，静态 bar.sync 12 条，没有原完整融合的 16/28 次 chunk 串行循环。每 dtype/Hidden 的 PTX 已保存，JSON 按 kernel hash 对应资源。官方 Pre2 为 40/48 registers/thread、shared=0、spill=0。无证据支持寄存器溢出归因。

结构改善是将同一 Token 投影铺到 16/28 个 CTA，而非仅一个 CTA。N=1 仍不能填满 132 SM；阶段2含部分和归约和输出写入，仍有不可消除的成本。大 N 时第一阶段执行时间占比明显提高；新的重复 x 读取、Partials 及第二次 Launch 使其在 BF16/FP16 高 N 不能超越现有 Single。此为数据流和时间证据，不是未经测量的硬件 stall 分解。

首次实验 `h100.json/h100.txt` 在 FP32 N=4096 D=7168 辅助 Profiler 的严格 40-event 断言中断，保留 23 行部分结果，未用于最终表。完整重跑发现有时 Stage1 只采到19条而 Stage2有20条；现在记录实际采样数，要求两个阶段各至少10条有效样本。独立双 Launch 正确性测试仍严格要求2次，未放宽。最终表使用独立重跑 `h100_final.*`。首次运行期间仅格式化新代码；最终重跑对应最终源码哈希。

## B4：只改变第一阶段 warps

在 BF16 N=1/8/32/128 追加 4-warps 对照，其余布局不变。每个配置均先通过原 Golden 容差。完整原始数据在 `h100_w4.json/csv`。

| N | D | Two w8 μs | Two w4 μs | 官方 / w4 speedup |
|---|---:|---:|---:|---:|
| 1 | 4096 | 5.761 | 6.084 | 1.468 |
| 8 | 4096 | 5.947 | 6.302 | 1.483 |
| 32 | 4096 | 7.320 | 6.973 | 1.533 |
| 128 | 4096 | 15.862 | 13.951 | 1.390 |
| 1 | 7168 | 6.420 | 6.722 | 1.364 |
| 8 | 7168 | 6.578 | 6.881 | 1.400 |
| 32 | 7168 | 9.754 | 8.737 | 1.455 |
| 128 | 7168 | 24.032 | 21.038 | 1.372 |

w4 为 56–58 registers/thread、4096 B shared、spill=0。N=32/128 改善但 N=1/8 退化约5%，故不替换默认8 warps，不推导分发阈值。保留显式 num_warps 实验参数，不做 Output Tile 拆分。

## 决策、限制与复现

- 小 N：三个 dtype 的 N=1/8/32，默认 Two 均超过官方，主要目标达到。BF16 N=32/128 可显式使用已测 w4 候选。
- BF16/FP16 N=1024/4096：已有 Single 更好，原实现和历史结果完整保留；本轮重测 Single 与历史收益基本一致。
- N=128：默认 w8 下 D=4096 Single 略快，D=7168 Two 更快；BF16 w4 则在两者均更好。没有统一 N=32 阈值。
- FP32 大 N：本轮 Two 快于 Single，因此适用范围还与 dtype、D 有关。
- 下一项最值得验证的单个方向：仅把 w4 对照扩展到 FP32/FP16 的 N=32/128，确认当前 BF16 改善是否可迁移；不急于引入 Output Tile 或自动 Dispatch。
- 不宣称未经测试的形状、机器或运行时收益。不需要修改白名单外文件。

从仓库根目录运行（已有 uv 虚拟环境）：

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$PWD/lab/ihc_fusion/results/.cache/triton"
lab/ihc_fusion/results/.venv/bin/python -m pytest lab/ihc_fusion/tests -q -p no:cacheprovider
lab/ihc_fusion/results/.venv/bin/python -m lab.ihc_fusion.benchmarks.bench_decode_two_stage --output lab/ihc_fusion/results/decode_two_stage/reproduction.json
# 局部对照：追加 --dtypes bfloat16 --tokens 1,8,32,128 --warps 4，使用新的 output 路径
```

Benchmark 拒绝覆盖已有 JSON，完整路径数据包括所有三个实现。源码 Ruff 检查通过。未提交、推送、切换分支或修改 Runtime。
