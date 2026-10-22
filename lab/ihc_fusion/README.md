# HY4 iHC Fused PostPre Lab

独立算子实验，不接入 vLLM Runtime。入口为
`kernels/fused_post_pre.py::fused_post_pre`，返回
`(next_x, next_residual, next_post_gates)`。

- 上一子块提供 `x[N,D]`、`residual[N,C,D]`、`previous_post[N,C]`。
- 下一子块提供 FP32 `weight[2C,C*D]`、`scale[2]`、`base[2C]`、
  `magnitude`、`hc_eps`、`norm_eps`。权重前 C 行为 Pre，后 C 行为 Post。
- 激活支持 BF16/FP16；FP32 是 Lab 对官方底层函数的直接调用扩展。
  所有输入要求同一 CUDA device、contiguous；支持非零 storage offset。
- 输出激活和 Residual 与输入 dtype 一致，Post gates 为 FP32。
  空 Token 输入不启动 kernel。可用 `out=(...)` 传入三个独立预分配输出。
  输出缓冲不得与输入或彼此重叠；本实验均使用独立缓冲。
- 非空输入只启动一个 Triton kernel。保留上一 Post 的低精度舍入；
  所有投影、归一化、门控和输出累加使用 FP32，禁用 FP fusion。
- C=4 使用固定 1024 lane 投影，先跨 chunk 累加再树形归约。
  该树与官方不同，数值误差按冻结规则验证；不保证输出逐位一致。
  Residual 在所有测试中逐位一致。
- 为控制寄存器，C=4 在最终输出阶段从原始输入重算舍入后的 Post，
  不回读已写出的新 Residual；必需的 Residual 输出仅写一次。
  C≠4 保留通用单 kernel 实现，未测其性能。
- H100 粗调后 C=4、N≤128 默认 16 warps，其余 4 warps。
  `num_warps=` 可显式覆盖。此阈值不代表所有形状的最优配置。

完整结果、误差、退化分析和阶段验收见 [results/REPORT.md](results/REPORT.md)。
小 Token Decode 仍有明显性能退化；本实验不构成 Runtime 集成建议。

## 环境与复现

从仓库根目录运行。所有输出、环境和缓存均放在白名单内：

```bash
export IHC_LAB="$PWD/lab/ihc_fusion"
export UV_CACHE_DIR="$IHC_LAB/results/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$IHC_LAB/results/.cache/python"
export UV_COMPILE_BYTECODE=0
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=3
export TRITON_CACHE_DIR="$IHC_LAB/results/.cache/triton"
export CUDA_CACHE_PATH="$IHC_LAB/results/.cache/cuda"
export XDG_CACHE_HOME="$IHC_LAB/results/.cache"
export FLASHINFER_WORKSPACE_BASE="$IHC_LAB/results"

# 本机复用已安装的 torch 2.13.0+cu129、triton 3.7.1、FlashInfer，保持只读。
# --python 选择解释器；所有实际 Python 命令均经过 uv 创建的 .venv。
uv venv "$IHC_LAB/results/.venv" --python /usr/bin/python3.12 --system-site-packages
uv pip install --python "$IHC_LAB/results/.venv/bin/python" \
  -r requirements/lint.txt pytest==9.1.1 ruff==0.14.0

IHC_CORRECTNESS_JSON="$IHC_LAB/results/reproduced_correctness.json" \
IHC_LAUNCH_JSON="$IHC_LAB/results/reproduced_launches.json" \
  "$IHC_LAB/results/.venv/bin/python" -m pytest -c /dev/null \
  --confcutdir=lab/ihc_fusion lab/ihc_fusion/tests/test_post_pre.py \
  -q -p no:cacheprovider

"$IHC_LAB/results/.venv/bin/python" -m lab.ihc_fusion.benchmarks.bench_post_pre \
  --rounds 3 --repeat-ms 100 \
  --output "$IHC_LAB/results/reproduced_benchmark.json"

"$IHC_LAB/results/.venv/bin/python" -m ruff check lab/ihc_fusion
"$IHC_LAB/results/.venv/bin/python" -m ruff format --check lab/ihc_fusion
```

Golden/Baseline 单独复现时，为 pytest 设置 `IHC_TEST_BASELINE_ONLY=1` 并选
`-k 'shapes or numerical_boundaries or preallocated_baseline'`，对应最初 43 项。
`-c /dev/null` 和 `--confcutdir` 避免加载 vLLM Runtime 的顶层测试配置。
未安装 git hooks 或 editable vLLM，因为会写出白名单。

在其他环境需自行在该 Lab 虚拟环境内准备上述 PyTorch/Triton/FlashInfer
版本与兼容 CUDA driver；不需要编译或安装本源码树中的 vLLM。

## 固定参考与官方来源

`reference/torch_post_pre.py` 是独立 PyTorch Golden，完全不调用 Triton。
`baselines/official.py` 的官方文件 SHA256 与 commit 通过双重校验：
`5df63a7203cdb9328d824a27b682e77c7b9928be`，
`vllm/models/hy_v4/nvidia/triton_ihc.py`。

此 checkout 缺少已编译 vLLM 扩展，直接 import 会失败。Baseline 从原始 AST
加载未改写的 kernel、函数和常量，省略 runtime 导入与支持判断，不走模型分发，
不替换官方计算实现。实际源文件与 callable code 路径在结果 JSON 中记录。
正式计时使用三个原始官方 kernel、原始 launch 配置与预分配缓冲；此路径已验证
与官方 Post→Pre 包装逐位一致。计时不包含分配、编译、输入生成或主机拷贝。

Golden、Baseline、误差规则在 Phase 2 后冻结，哈希见
`results/frozen_sha256.json`。保留了首次 Baseline/Golden 验证时发现并修正
固定绝对阈值问题的日志；后续所有 fused 版本均未修改接受规则。
