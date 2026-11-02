# Gluon 布局必要性与优化空间：受控消融

2026-11-02，生产基线 `daa54b83bb61f49954d8823a0a3197f448b6e480`，H100 SXM 80GB / SM90 / 132 SM。
本轮只增加实验脚本与证据，生产 Kernel、Attention Backend、Scheduler、权重和 iHC
均未修改。H200 仍需用户手动复测。

## 结论

**本负载需要合适的布局与数据搬运，但不要求必须用 Gluon。** 调优后的普通
Triton 在 1–512 decode tokens 内与当前 Gluon 相差约 -0.24% 至 +2.24%。
编译后的 IR 证明，普通 Triton 自动选择了相同的 MMA warp 排布 `[1,4]`、
shared swizzle `vec=8, perPhase=1, maxPhase=8`，并生成 `cp.async`。
Gluon 提供显式控制；这些优化的性能也能由当前版本的普通 Triton 编译器实现。

当前布局有实际作用：固定其他因素，batch=128 去掉 KV 异步拷贝使延迟增加
约 20%，去掉 shared swizzle 则使延迟约为原来的 2.50 倍。它们的收益不能相加，
也不能解释成旧 FP8 版本到新 BF16 版本的完整收益比例。

本轮扫描没有找到稳定超过当前默认策略的参数组合，因此保留生产实现。
这不证明所有可能算法已经最优；只说明本轮可调参数范围内没有可落地的新收益。

## 口径与验证

固定 BF16 Q/KV、8 heads、QK=576、latent V=512、TopK=2048、scale=0.0625 和 sink。
核心消融固定 64-key tile、自适应 splits、FP32 `[T,S,H,514]` partial 和同一个
生产 softmax merge。只改变 KV 加载方式、shared layout 或 split-kernel 实现。
普通 Triton 的对照直接处理 BF16，不混入旧版 FP8 的解量化开销。

每个请求有独立 16K KV 区域，轮换 `max(2, ceil(32/T))` 份输入。实际被索引的 KV
footprint 最小约 71 MB，超过 50 MB L2；不是每次调用前强制清空 L2。
编译、分配、输入生成和 FP32 reference 均在计时外。所有方案共用预分配的
partial/output 缓冲区；测量包括 split+merge，并另外记录 split-only 延迟。
CUDA Graph/event timing，按轮次轮换和反转方案顺序；JSON 保留每轮样本。
本页数字只与本轮同口径对照比较，不混用历史 wrapper benchmark 数字。

- 布局粗扫：GPU5，22 个方案 × 8/128 tokens，3 轮；记录在
  [sweep JSON](results/h100_gluon_ablation_sweep.json)。
- 正式复测：GPU5，5 个方案 × 9 个 batch 点，5 轮；记录在
  [final JSON](results/h100_gluon_ablation_final.json)。
- Split/tile 扫描：GPU4，32/64-key tile × 1/2/4/8/16/32/64 splits ×
  1/8/32/128 tokens，3 轮；记录在
  [splits JSON](results/h100_gluon_ablation_splits.json)。同组比较始终在同卡完成。
- 上述 145 个配置/形状组合均先通过 FP32 output/LSE 对照再计时。
  容差为 output `rtol=0.02, atol=0.01`，LSE `rtol=atol=1e-4`。
- 普通 Triton 另复用已有算子测试：**13 passed**，覆盖 8/16/32 heads、split
  配置、strided/未对齐视图、无效索引/长度、空行、sink、尾块和 CUDA Graph replay。
  adapter 只替换该测试进程中的 Python 入口，不修改生产文件。

首轮 GPU2 测量期间出现其他进程，样本波动接近两倍，已弃用；这里不包含其计时。

## 普通 Triton 能否复现性能

选择的普通 Triton 配置是 `BLOCK_N=64, num_warps=4, num_stages=2`。
选型前比较了 32/64/128-key tile、4/8 warps、1/2/3 stages；避免用未调优对照
推断语言优劣。粗扫中 batch=128 的普通 Triton stages=1 约 199.67 µs，
stages=2 约 110.31 µs；只改编译参数就消除了大部分差距。

| Decode tokens | 当前 Gluon (µs) | 普通 Triton stages=2 (µs) | 普通 Triton 延迟变化 |
| --- | --- | --- | --- |
| 1 | 8.80 | 8.93 | +1.44% |
| 4 | 11.58 | 11.68 | +0.81% |
| 8 | 16.40 | 16.74 | +2.07% |
| 16 | 22.85 | 23.36 | +2.24% |
| 32 | 35.83 | 35.74 | -0.24% |
| 64 | 60.62 | 60.69 | +0.11% |
| 128 | 109.71 | 110.00 | +0.26% |
| 256 | 207.88 | 208.29 | +0.20% |
| 512 | 407.13 | 407.54 | +0.10% |

batch=128 的编译资源也接近：Gluon 196 registers/thread、94,208 bytes shared；
普通 Triton 197 registers/thread、94,464 bytes shared，均无 register spills。
IR 摘录和静态 PTX 指令计数保存在 [ncu JSON](results/h100_gluon_ablation_ncu.json)。
PTX 指令计数是编译代码中的静态出现次数，不是运行时执行次数。

## 哪些底层优化确实有用

以下仅替换对应因素，batch=128、2 splits：

| 方案 | split+merge (µs) | 相对当前耗时 |
| --- | --- | --- |
| 当前 Gluon | 109.71 | 1.00× |
| Gluon KV 普通 load/store | 131.49 | 1.20× |
| Gluon 无 shared swizzle | 274.34 | 2.50× |
| 普通 Triton，stages=2 | 110.00 | 1.00× |

使用相同 seed 和输入形状采集 Nsight Compute 2025.2.1；此处不使用 profiler
耗时作为 benchmark 延迟。资源和带宽指标为硬件计数器结果：

| 方案 | Registers/thread | Achieved occupancy | Shared load conflict 占 wavefronts | DRAM peak-sustained |
| --- | --- | --- | --- | --- |
| 当前 Gluon | 196 | 12.00% | 0.12% | 80.65% |
| Gluon KV 普通 load/store | 254 | 11.95% | 0.07% | 63.86% |
| Gluon 无 shared swizzle | 186 | 12.22% | 86.87% | 26.22% |
| 普通 Triton，stages=2 | 197 | 11.99% | 0.61% | 79.02% |

Shared load conflict 比例按 Nsight 官方规则计算：
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum /
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`。
它不是全局访存冲突率，也不是等待时间百分比。

去掉 swizzle 后寄存器更少、occupancy 仍约 12%，延迟却大幅增加，并有大量 shared
load bank conflicts。单看寄存器数或 occupancy 会误判。KV 改普通 load/store
后则增加寄存器中转，registers 从 196 增至 254，DRAM 吞吐下降。

## 还有多少优化空间

布局粗扫覆盖 shared swizzle、MMA warps `[1,4]/[2,2]/[4,1]`、copy warp 排布、
4/8 warps、QK chunk 64/128/256/512 和 key tile 32/64/128。没有稳定超过当前
Gluon 的方案；chunk 大小的差异落在约 1% 的测量波动内。

单独扫描 split 数，以下只比较 GPU4 同一轮数据：

| Decode tokens | 当前 tile / splits | 当前 (µs) | 扫描最佳 tile / splits | 最佳 (µs) |
| --- | --- | --- | --- | --- |
| 1 | 64 / 32 | 8.79 | 64 / 32 | 8.79 |
| 8 | 64 / 32 | 16.41 | 64 / 16 | 16.36 |
| 32 | 64 / 8 | 35.77 | 64 / 8 | 35.77 |
| 128 | 64 / 2 | 110.12 | 64 / 2 | 110.12 |

8 tokens 的 16 splits 只有约 0.3% 差异，不足以支持增加新路由规则。
128 tokens 的 32-key tile 即使调到 4 splits，也慢于当前 64-key/2-split 配置。

因此，本轮没有测出可承诺的额外加速。更深的数据流优化、跨 tile 的加载/计算重叠、
单 split 时融合输出等仍是待测方向；约 81% 的 DRAM 指标不能直接换算成可获得的
提速，也不能证明已经达到硬件上限。本轮没有为未经验证的方向增加生产代码。

## 复现

环境沿用 [README](README.md)。本机在命令前加
`CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/tmp/hy4-smallhead-bootstrap`；
H200 使用本分支和匹配扩展。粗扫保留原 seed 顺序，正式测量按 `seed+tokens`
固定每个形状的输入，使 split 扫描也复用相同随机数。

```bash
.venv/bin/python lab/hy4_smallhead/ablation.py \
  --tokens 8 128 --seed-once --rounds 3 --rep 50 --output sweep.json

.venv/bin/python lab/hy4_smallhead/ablation.py \
  --tokens 1 4 8 16 32 64 128 256 512 \
  --variants current gl_loadstore gl_no_swizzle tl_n64_w4_s2 tl_n32_w4_s2 \
  --rounds 5 --rep 100 --output final.json

.venv/bin/python lab/hy4_smallhead/ablation.py \
  --tokens 1 8 32 128 --splits 1 2 4 8 16 32 64 \
  --variants current gl_n32 --rounds 3 --rep 50 --output splits.json

.venv/bin/python lab/hy4_smallhead/check_plain_triton.py

ncu --profile-from-start off \
  --kernel-name 'regex:(_split_kv|gluon_split|triton_split)' \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --section MemoryWorkloadAnalysis_Tables --export ablation \
  .venv/bin/python lab/hy4_smallhead/ablation.py --tokens 128 \
  --variants current gl_loadstore gl_no_swizzle tl_n64_w4_s2 \
  --profile --output codegen.json
```

正式 profiler 实测只额外采集 Tables 中的四个 shared load/store bank-conflict 和
wavefront 指标，以减少 replay passes；上面的 Tables 命令提供同类指标及更多细节。
`current` 指此报告基线的生产内核；从其他提交重跑时，应同时记录实际源码版本。
实验内核中 `gl_control` 是 `daa54b83bb61f49954d8823a0a3197f448b6e480` 的冻结等价版本，粗扫已确认两者一致。
