# IHC 分支迁移与重构复盘

本页补齐第一次复盘中没有单独展开的开发阶段。先前材料覆盖了最终变化，但 Lab 工作区快照与最终源码之间缺少开发分支重构前的完整验证框架；现补入迁移、重构前、重构后三份固定提交快照。

## 分支关系

`exp/hy4-ihc-lab` 与 `feat/hy4-ihc-dev` 的共同祖先是 `5df63a7203`。Lab 的最终提交 `da9bc79158` 不在开发分支最终版本的祖先链上。开发分支通过新提交接入已经验证的方案，而非直接 merge 整个 Lab 分支。

```text
共同基线 5df63a7203
  ├─ exp/hy4-ihc-lab
  │    Lab 算子 → Runtime 接入 → 同层/跨层 → TP4与Prefill选型
  │    结束于 da9bc79158
  └─ feat/hy4-ihc-dev
       990a299dee 生产接入 → 79447ce5f9 冻结Lab与Golden
       → Graph路由调整 → TP8验证框架 → baabc917e5 混合批次资格调整
       → f7c538cc2e 实现与验证重构 → 713fb9e7f8 后续调优
       → 4f1fe69872 最终文档基准
```

两个 Runtime 核心文件 `model.py`、`triton_ihc_post_pre.py` 在 Lab HEAD 与开发分支首次接入提交中 Git blob 完全一致。这支持“迁移已有实现”的判断，不能将分支迁移本身当作一项新增性能优化。[机器可核验记录](archive/branch_refactor_evidence.json)包含共同祖先、文件blob和重构统计。

## 关键阶段与变化

| 阶段 | 提交 | 变化与含义 |
| --- | --- | --- |
| Lab 研发 | `4fc3e26c54` 至 `da9bc79158` | Golden/官方对照、Single/Two-stage、同层/跨层集成、TP4与Prefill选型 |
| 开发分支接入 | `990a299dee`、`79447ce5f9` | Runtime核心实现迁移；冻结Lab、Golden及官方基线SHA，拆分生产与实验职责 |
| Graph路由探索 | `ebf4432ff3` | 尝试以稳定张量资格保证capture/replay路径一致 |
| TP8框架与回退 | `e393f5c92e`、`244dfa61d9` | 增加TP8 harness、Serving/cache/decode profiling；后者重新拒绝Graph路由，并整理旧Lab/TP4工具 |
| 重构前最后调整 | `baabc917e5` | 去掉纯Prefill中“无Decode”的限制，允许满足条件且含Prefill的混合批次；仍要求actual token匹配、N范围等。不是Attention Backend拆批 |
| 集中重构 | `f7c538cc2e` | 25文件、增加253行/删除3162行；RMSNorm融合、autotune、张量资格路由、精简验证与移除自定义harness |
| 后续优化 | `713fb9e7f8` | 调整BLOCK_K与autotune key、增加8-warps候选、开启FMA，修复全量hidden-state探针配置与传输 |

仅存在TP8工具不代表全量TP8测试已通过。Graph路由曾放开、收紧、再移除显式过滤；不能把“中间版”概括成始终相同的一套门控。

## 集中重构前后

| 项目 | 重构前 baabc917e5 | 重构后 f7c538cc2e |
| --- | --- | --- |
| Kernel边界 | Post+Pre，RMSNorm单独执行 | Post+Pre+RMSNorm，减少独立launch与中间输出落地 |
| Runtime资格 | 依赖ForwardContext/Attention metadata；N=512～8192，要求含Prefill、actual token匹配，排除compile/ubatch/Graph | 去除这些显式条件，按启用开关、配置、dtype、设备与张量形状/布局判断 |
| Triton调优 | BLOCK_K固定1024、4 warps | 仍固定1024；新增4 warps、1/2/3 stages的autotune，key为(N,D) |
| 数值细节 | Pre的BF16输出落地后执行RMSNorm | FP32累加结果直接RMSNorm；此提交仍为enable_fp_fusion=False |
| 验证入口 | Lab多类严格测试、metadata/Graph路由测试、自定义TP4/TP8与Serving harness | 主要集中到tests/models/test_hyv4_ihc_fusion.py，输出max_abs与cosine；README改用vLLM标准Serving工具 |
| 验收含义 | 保留较多自动断言和边界回归 | 更简洁、便于真实权重逐层观测，但没有自动数值阈值；不能等同于“原严格验收失败已修复” |

减少重复工具、使用标准Serving入口是可解释的工程收益；删除测试和简化指标也收窄了自动回归保障。这次重构既有生产行为变化，又有验证策略变化，不是纯代码清理，也不能仅凭删除行数评价质量。

## 不要混淆重构与后续调优

`713fb9e7f8` 才把autotune key从(N,D)改成(D,BLOCK_K)，N≤128使用2048、其他使用1024，新增(8 warps,1 stage)，并将enable_fp_fusion改为True。

同一提交还让真实模型hidden-state检查同时开启PostPre/cross-layer开关，调整runner的block size/token budget，并把worker返回值转为可传输的列表再恢复Tensor。因此重构后验证入口的完善与kernel调优发生在同一提交，不能把测试口径变化解释为模型误差自然变小。

现有原始统计支持各自实验条件下的性能结论；没有单独控制变量的实验把RMSNorm融合、路由放宽、FMA和autotune的所有端到端贡献逐一拆开。三层Forward数据不能自动归属于重构后的最终kernel；Serving结果也不能全部归因于某一个重构项。

## 离线证据入口

- [开发分支迁移快照](archive/ihc_development_migration.tar.gz)：对应79447ce5f9，包含生产模型文件和冻结Lab。
- [重构前快照](archive/ihc_before_refactor.tar.gz)：对应baabc917e5，包含后来移除的TP8/Serving harness和当时的模型测试。
- [重构后快照](archive/ihc_after_refactor.tar.gz)：对应f7c538cc2e，保留第一版RMSNorm/autotune及精简验证。
- [重构完整差异](archive/refactor_changes.patch)、[开发分支提交顺序](archive/development_history.txt)、[Lab提交顺序](archive/lab_history.txt)。
- 最终713fb9e7f8之后的实现见既有 `ihc_final_4f1fe69872_source.tar.gz`；Lab未提交实验数据仍见原工作区快照。

每份源码包的manifest记录完整提交与逐文件SHA256。解压到独立目录审阅，不覆盖当前代码。分组 `offline-review.tar.gz` 和校验清单已更新，旧提交下载链接仍指向旧包。

## 面试如何讲这次重构

可以说：先在隔离Lab验证融合边界和数值契约，再把经过验证的实现迁移到生产开发分支，随后将RMSNorm纳入融合，并将分散的验证工具收敛到算子/逐层hidden-state与标准Serving入口。重构减少重复脚手架，也改变了路由和浮点舍入，因此重新区分性能验证与严格质量验收；后续另做小N调优和探针修复。

被问“这是重构还是优化”时，应回答：两者都有。生产融合与路由改变属于优化和行为调整；验证工具重组属于工程重构。仅迁移分支没有带来新的算法收益，测试简化也不能替代未完成的精度验收。
