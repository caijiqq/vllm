# HY4 iHC SM90 Runtime 与 CUDA Graph 验证

2026-10-23，分支 `feat/hy4-ihc-dev`，起点 `aa72b73698`。
**PARTIAL：Runtime 兼容性修复已完成，严格链级和模型级数值验收未闭合。**
未修改 Kernel、Golden、Baseline、Attention/MoE、Scheduler 或通用 Graph 框架。
没有 Push 或创建 PR。原始日志、trace、数值样本和临时驱动位于
`/tmp/hy4-runtime-validation`，不纳入提交。

## 只读审计

- 唯一融合设备名称门禁位于 `HYV4DecoderLayer.__init__`：精确匹配
  `NVIDIA H100 80GB HBM3`，因此 H200 即使打开配置也无法融合。
  改为现有 `current_platform.is_device_capability(90)`，不开放 SM80/SM100。
- 另外两道门禁是初始化的 `model_config.enforce_eager` 和
  `_can_fuse_ihc` 的 `cudagraph_runtime_mode == NONE`。
  没有额外的 `is_capturing` 判断。`torch.compiler.is_compiling()` 回退保留。
- 模型前向依次调用各 DecoderLayer。首层 Attention Pre 保留；Attention
  Post 与 MLP Pre 同层融合；MLP Post 可延迟到下一层 Attention Pre。
  pending 为局部 `(residual, post_gates)`，Pre 参数属于消费者，Post gates
  属于生产者。消费者不能融合时先物化 Post，再执行普通 Pre。
- `_ihc_defer_post` 只在本 PP stage 最后一层之前启用，stage 尾部总是完成
  Post。PP 只传 flatten 后的 hidden states，接收 stage 从 pending=None 开始。
  最后一个 stage 完成 Post 后才进入 HCHead 和最终 norm。
- V1 入口是 `GPUModelRunner` 包裹的 `CUDAGraphWrapper` 或
  `BreakableCUDAGraphWrapper`。本机实际默认 V2，入口为
  `gpu/model_runner.py::execute_model` → `cudagraph_manager.run_pw_graph`
  → `breakable_cg_runner`。FULL 则走 manager 的 `run_fullgraph`。
- HY4 Attention 的 `_indexer_and_attn` 已提供分段图的 eager break，使用
  预分配输出；本轮不修改该边界。

## 局部适配与限制

Eager 继续要求明确的、无 padding 的纯 Prefill metadata。
Graph 的 batch key 不包含实际 Prefill/Decode 数量；capture 可能没有 metadata。
因此 Graph 路由只使用静态 tensor 条件，允许符合条件的混合/Decode/padded 批次。
iHC 运算逐 token 独立，不读取 attention metadata；不能把捕获时的批次判断
固化到会服务其他批次的图里。小 N 等静态不适格条件仍回退。
这里 N 指 tensor 的物理行数，包括 Graph padding；实际请求 token 数不用于图内路由。

保留：实验开关、iHC、D6144/C4、BF16 输入/residual、FP32 gates/参数、同设备、
连续 layout、512≤N≤8192、无 speculative/batch-invariant/ubatch。
PIECEWISE 只允许现有 breakable 路径；非 breakable 编译路径继续回退。

Python 分支不读取 tensor 数值。`empty_like` 输出在 capture 时由 graph pool
分配，replay 使用捕获的地址；pending 本身不跨请求保存，不增加模型全局 buffer。
图和其输入/输出存储必须由现有 wrapper 管理。配置在捕获后应保持不变；改变
融合配置需要重新建图，不能只修改 Python flag 后复用旧图。

## 测试设计及证据

复用 `tests/models/test_hyv4_ihc_fusion.py` 和冻结 Lab helpers。
模块契约是将 `[N,D]` 输入、`[N,4,D]` residual 与 gates 变换到下一边界；
风险是 gate/weight 所有权错误、漏掉首尾 Post/Pre、图重放读取陈旧地址。
最低成本覆盖为现有三层合成 Decoder 链及真实 graph wrappers；真实 Attention/MoE
和 TP 通信另用原 MXFP8 三层权重验证。

- 修改前：53 passed / 1 failed，失败为已有 N8192 严格三层链用例。
- 最终 Runtime 测试：74 passed / 13 failed。13 项均为严格数值失败：原 N8192
  1 项，加上更新输入的 12 项 Graph ON/OFF 比较；详见 `acceptance.log`。
  18 种 Graph 场景的 72 组更新输入均完成 replay，各自 Graph/Eager 逐位一致。
- 冻结 Lab 独立运行：108 passed。与多个 Profiler 混跑时曾有两个 GPU launch
  计数失败，独立进程原样重跑通过；未修改 Lab 测试来绕过。
- H100、Mock H200 构造及实际融合 Dispatch 通过；Mock SM80、SM100、未知架构回退。
  Mock 在 H100 上执行，只证明门禁/Dispatch，不证明 H200 硬件兼容。
- Graph 覆盖 N128/512/1024、FULL/breakable、正常链/消费者关闭/模拟 PP 边界，
  从无 metadata 的 capture 开始，再以 Prefill/mixed/padding/Decode metadata
  重放四组更新输入。OFF/ON 都与各自非捕获执行逐位比较，并检查重复 replay。
- 正常三层 N512/1024：每次 replay 的 GPU trace 为 5 Fused + 1 Pre1 + 1 Pre2
    - 1 Post，Python 融合调用数在 replay 时不增加。N128 为 0 Fused + 各 6 个原
  Kernel；模拟 PP 断点为 4 Fused + 各 2 个原 Kernel；关闭中间消费者为
  2 Fused + 各 4 个原 Kernel。首层 Pre、末层 Post 保留。
- 合成链测试没有非法访存或地址生命周期错误。真实多 GPU PP 传输尚未测试。

## 严格数值失败：不放宽阈值

冻结边界规则：Residual exact；gates rtol=2e-5、atol=2e-6、relative L2≤2e-6；
BF16 output 使用既有 2 eps 相对误差及 FP32 cancellation budget，relative L2≤eps。
三层链仍采用原 `rtol=2*BF16 eps, atol=2e-5`，没有 xfail、skip 或宽松替代规则。

九个生产 Kernel/Golden 用例全部通过：output 最大 absolute error=0.03125，
relative L2≤2.03997e-5，cosine≥0.9999999997919。

N8192 原始输入下，旧/新 Runtime 的 OFF、同层、跨层结果分别逐位一致，重复
A/A 也逐位一致。跨层对 OFF：max abs=1.0、relative L2=5.12420e-6、
cosine=0.9999999999869；201326592 个元素中 103 个不等，仅 1 个违反严格规则。
失败坐标 `(8162,3,4517)`，OFF=-0.037353515625，ON=-0.042724609375。
误差分布与实际小型输入样本见 `numerical_audit.json` 和 `rounding_*.pt`。

更新输入的第 2 步 `x = -roll(x,1,0)*0.91`，N512/N1024 均出现 4 个超阈值元素；
同样在旧 Runtime 的 Eager 同层/跨层路径复现，旧/新逐位一致且 A/A exact。
N512 最终 `(225,2,2384)` 为 ON=-0.37109375、OFF=-0.341796875。
同 token/column 的第一处融合边界为 0.011474609375 对 0.01153564453125；
BF16 中点 0.011505126953125，独立 FP64 诊断值距中点约 9.02e-9。
这些证据支持归约顺序造成 BF16 舍入分叉、在后续层传播并在抵消区域放大的原因。
FP64 仅用于诊断，未替代既有 FP32 Golden 或改变验收规则。

因此，新测试发现了旧实现已有的输入敏感精度问题，Graph 本身的逐位 replay
对齐不能替代严格 Fusion ON/OFF 验收。修复该精度问题涉及冻结 Kernel 的数值语义，
本轮不扩大到 Kernel 修改。

## 真实 H100 TP4 三层模型

权重：`/data/chenhao/Hy4-preview-FP8-3-layers`，BF16 激活、ModelOpt MXFP8、
TP4/PP1，GPU2/3/5/6。不加载完整 HY4。GPU0 原有任务未触碰。
使用当前工作树 Python 源码及已有安装包的二进制扩展；uv 创建当前 `.venv`，
临时 `sitecustomize.py` 只补充扩展库搜索路径。Torch 2.13.0+cu129。

最初同进程动态切换并额外创建 OFF 图的实验在 N512 Graph OFF 出现 IMA。
该次性能结果作废。随后标准固定配置的 Graph OFF 在三个 N 上完成，未复现 IMA。
后续每个模式独立模型进程，启动前固定两个融合开关，不创建第二套实验图。
这不是通过禁用 Graph 规避问题；Graph 模式仍走原生 V2 breakable replay。
额外多图实验的 IMA 原因未闭合，不声称其生命周期问题已经修复。

原路径模型 A/A 严格数值仍失败，甚至 N128（没有融合）也失败。
模型 A/B 结果只作为诊断，不把生成 token 一致或有限输出当作精度验收。

## 真实模型结果与性能口径

四模式均完成 N128/512/8192、原输入 A/A、新输入 A/A。Graph ON/OFF 每个 rank
均捕获 4 个图段、3 个 Attention eager breaks；标准独立进程均无 IMA。
Graph ON N8192 replay 的每个 rank 实测 5 Fused + 各 1 个 Pre1/Pre2/Post；
Python 融合 spy 次数为 0（图重放不执行 Python），GPU trace 中融合次数为 5。
Eager ON 的 Python 和 GPU 次数均为 5。OFF 均为 0 Fused + 各 6 个原 Kernel。

N8192、原输入、全部 rank 的最差数值指标：

| 对照 | max abs | relative L2 | 最低 cosine | 严格判定 |
| --- | ---: | ---: | ---: | --- |
| Eager ON / OFF | 0.343750 | 0.00900183 | 0.999959484 | FAIL |
| Graph ON / OFF | 0.351563 | 0.00911502 | 0.999958458 | FAIL |
| Graph ON / Eager ON | 0.343750 | 0.00854383 | 0.999963501 | FAIL |
| Graph OFF / Eager OFF | 0.339844 | 0.00864883 | 0.999962599 | FAIL |

上述模型级 FAIL 不能仅凭原路径 A/A 噪声解释完毕；没有证明融合造成了新的误差，
也没有证明全部差异都来自原有非确定性。完整 96 组跨模式/输入/rank 对照以及
各模式 A/A 指标存于 `fixed/summary.json` 和 `fixed/*_numerical.json`。

以下性能为**数值尚未验收的诊断数据**。三层真实模型，N8192/D6144/C4/BF16，
相同 prompt `100+i%197`、temperature=0、seed=7、max_tokens=1，prefix cache 关闭。
每模式独立进程，6 次 warmup，15 次测量。CUDA Event 围住模型 Forward 或整个
分段图 replay；包含 Attention eager breaks、embedding/HCHead 和 TP 通信，
不含 logits/sampling。每个样本先在 device group barrier 同步（计时外），取四
rank 耗时最大值，然后求 median/P10/P90。Profiler 在计时后单独执行。

| 模式 | median ms | P10 ms | P90 ms | min–max ms |
| --- | ---: | ---: | ---: | --- |
| Eager OFF | 44.273 | 44.149 | 44.493 | 44.057–45.026 |
| Eager ON | 41.990 | 41.772 | 42.166 | 41.664–45.558 |
| Graph OFF | 43.895 | 43.681 | 44.046 | 43.627–44.589 |
| Graph ON | 41.676 | 41.555 | 41.925 | 41.451–42.022 |

Eager 融合延迟下降 5.16%，Graph 融合下降 5.06%；Graph ON 比 Eager ON 低约
0.75%。没有观察到明显的融合性能退化。独立进程顺序测量未交错，未锁定时钟、
未显式 flush L2，因此收益幅度仍可能受到漂移影响，不当作完整模型 E2E 结论。
最后一轮标准 capture 的日志粒度为约 1 秒（3 个 shape 的总量），不计入 replay；
OFF/ON graph memory 分别为每 rank 7.29/7.61 GiB，ON 增加约 0.32 GiB。

## 修改与检查

- `300bc325c6`：SM90 架构门禁，Runtime +1/-1。
- `ebf4432ff3`：Graph 稳定路由，Runtime +12/-6。
- 测试/本文档独立提交；Runtime 合计 +13/-7，没有新 Kernel 或公共框架修改。
- 测试文件复用三层构造，增加架构 mock、Graph wrappers、新输入、消费者关闭、
  PP 物化、trace 及静态 fallback 覆盖；报告列表补齐类型并使用 Transformers v5
  名称和通用 accelerator 同步 API，通过相关项目 lint。
- 完整 pre-commit 仅 CI 登记检查失败：原有
  `tests/models/test_hyv4_ihc_fusion.py` 未由任何 Buildkite job 执行。
  本轮没有修改 `.buildkite`，避免越过 Runtime/架构/测试范围；手工执行结果如上。
- 仍未验证：H200 真机、完整 HY4 TP8、真实 PP>1 传输、真实模型 FULL graph、
  speculative/batch-invariant/ubatch/torch.compile 融合（这些保守回退）。

## 复现与 H200 后续验收

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/hy4-runtime-validation/bootstrap:$PWD
export TRITON_CACHE_DIR=/tmp/hy4-runtime-validation/cache/triton
.venv/bin/python -m pytest tests/models/test_hyv4_ihc_fusion.py -q -p no:cacheprovider
.venv/bin/python -m pytest lab/ihc_fusion/tests -q -p no:cacheprovider
```

原始模型四模式驱动为临时 `fixed_run.py` / `fixed_workers.py`；每个模式以
`ARM=eager_off/eager_on/graph_off/graph_on` 独立启动，配置与采样规则保存在脚本中。
日志/trace/缓存/权重快照不提交。Ruff check/format 使用项目 v0.14.0 pre-commit
hooks 执行；不向与其他工作树共享的 `.git/hooks` 安装 hook。

H200 TP8 下一轮最小步骤：

1. 在隔离 checkout 应用三个提交，核对真实 SM90、依赖、权重 revision、TP8/PP1。
   使用现有 full-model 配置，两个 `hf_overrides` 均设为 True。
2. Graph 设 `enforce_eager=False`、`VLLM_USE_BREAKABLE_CUDAGRAPH=1`、
   `compilation_config.mode=0`、`cudagraph_mode=PIECEWISE`，显式包含测试 Prefill N
   的 capture sizes。若图只覆盖小 Decode shape，即使配置开启也可能没有融合。
3. OFF/ON 各自独立进程并重新 capture。先做小 N fallback、适格 N、混合批次、
   padding、新输入多次 replay；检查每个 rank 的初始化 flag 和实际 GPU trace。
   L 个本地层预期 2L-1 个融合（无额外 PP 断点），不能只检查编译缓存。
4. 保存真实 hidden states/logits，重复 OFF A/A、ON A/A、OFF/ON、Eager/Graph；
   使用当前严格阈值和固定输入。先隔离原路径 Attention/MoE 非确定性，再判断
   融合误差；目前 H100 严格验收未闭合，H200 只适合受控诊断，不应作为发布验收通过。
5. 精度与稳定性通过后再做 TP8 完整模型性能测试：同配置 warmup、交错重复、
   每次取全部 rank 最大值，报告 median/P10/P90 和 E2E 指标；单独记录 capture。
   不将本次三层数据外推成完整模型 TP8 加速比。
