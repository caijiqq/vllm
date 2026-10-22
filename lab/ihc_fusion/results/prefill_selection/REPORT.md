# HY4 PostPre Prefill 收尾与选型证据

结论：保留冻结 Single 与 Two-Stage；本轮不新增 Selection，不修改 Runtime。公开 HY4 BF16 D=6144 在已测 N=768/1024/2048/4096/8192 上 Single 稳定领先；小 Prefill 有可能存在，Single 不能被证明覆盖所有 Prefill。中间区的胜者受缓存状态影响，不能仅凭合成 N 给出生产阈值。Kernel 层已具备开展受控模型集成验证的基础，尚不具备直接启用生产替换的证据。

## A. 实际调用与证据等级

工作分支 `exp/hy4-ihc-lab`，HEAD `078c5c6fa01258079ea508d526535b4a7cd15429`。开始时已有 `bench_decode_two_stage.py` 一行修改及未跟踪 `results/prefill_6144/`，均原样保留；初始差异在 `initial.diff`。

**源码事实**（以下路径相对仓库根）：

- `vllm/models/hy_v4/nvidia/model.py:158,187`：enable_ihc 为真进入 `_forward_ihc`，没有 Prefill/Decode 判断。当前 Runtime 没有调用 Lab 的 PostPre。第一处可融合边界在同层 Attention 后 Post → MLP Pre（约201–204行）；第二处为本层 MLP Post → 下一层 Attention Pre，需要跨层状态调度。第一层首次 Pre 与最终 Post→HCHead 不属于相同 PostPre 算子。
- `vllm/models/hy_v4/nvidia/hc.py:330,365,380`：prepare_input 接受 `[N,D]`（repeat）、`[N,4D]`（reshape）、`[N,4,D]`（不变）；pre 返回 reduced、post_gates、原 residual；post 将前一子块输出和它自己的 gate 加到 residual。融合必须使用上一子块 Post Gates 和下一子块的 hc_fn/scale/base。
- `hc.py:67` 的 hc_fn 是 ReplicatedLinear，FP32 权重 `[8,4D]`，Scale/Base FP32；D来自 config.hidden_size，不是 D/TP。Attention/MLP 输出恢复完整 hidden dim，MoE forward 在 `moe.py:180` 恢复 orig_shape。TP/EP 内部通信不等于将 iHC 权重的 Hidden 维分片。
- `hpc/hpc_ihc.py:43` 只支持 SM100/103、C=4、D=4096/6144，H100 SM90 不进入 HPC。`triton_ihc.py:19` 的官方路径还要求 CUDA、BF16/FP16、非 VLLM_BATCH_INVARIANT；FP32 仅为 Lab 直接调用扩展。不能在 Runtime 绕过这些原始回退条件。
- `vllm/v1/core/sched/scheduler.py:594,703,1129`：按 request 剩余 token、输入 budget、long_prefill_token_threshold 和 Chunked Prefill 条件分配 token；没有固定 Prefill N。Continuous Batching 的本步 N 是所有本地 scheduled request token 数之和，可含混合 Prefill/Decode。不是 prompt_length，也不是 batch_size，不能根据 N 判断阶段。
- 旧 Runner `vllm/v1/worker/gpu_model_runner.py:4019,4052,4164,3400`：scheduled token sum 经 CUDA Graph/DP padding、可能的 ubatching，形成传入模型的 token 维。新 Runner `vllm/v1/worker/gpu/model_runner.py:1280,1298,1537` 同样有 num_tokens_after_padding，并显式保存 is_prefilling_np/has_prefill。Runner 由 `gpu_worker.py:563` 选择。应记录每次边界实际张量 N，而不是只记录 scheduler 的未 padding 总数。
- 短 prompt、chunk 尾部、prefix cache 命中后剩余少量 token 都可以产生小 Prefill；batch 混合又可能使总 N 很大。源码证明可达，不证明高频。MTP 在 `nvidia/mtp.py:354` 显式禁用 iHC，不纳入本算子目标。

**配置事实**：仓库 HYV4Config 默认 hidden_size=2816 (`vllm/transformers_utils/configs/hy_v4.py:86`)，不能当作部署模型的 D。仓库支持表列出的公开 [tencent/Hy4-preview config](https://huggingface.co/tencent/Hy4-preview/blob/705d81ee51566a186d645b74c974d642ef2828fe/config.json) 则为 D=6144、C=4、BF16、magnitude=2、hc_eps=1e-6、rms_norm_eps=1e-5。已保存原 JSON、响应 X-Repo-Commit 和 SHA256，见 public_config*.json。模型加载 dtype 由 checkpoint config 与用户 dtype 选项共同决定（`vllm/config/model.py:2450`）。4096/7168 是已测 Lab 点；4096 也在 HPC 支持集合中，但不能据此声称它们是当前部署的常见 D。

**未知条件**：未提供目标 checkpoint/启动命令/真实请求 trace；本机已检查的模型目录未挂载 HY4 权重。本轮只下载公开配置，未下载或运行大模型。尚不清楚实际 TP/PP/DP、Runner、chunk budget、prefix cache 命中率、Prefill N 频率、混合 batch 比例和层间缓存热度；已请求用户补充。公开配置是可验证参考，不等于确认用户部署。测试输入和权重仍为 seeded synthetic，没有伪造模型 shape 分布。

当前两种 Lab Kernel 覆盖 C=4、连续输入和三种 dtype 的数学契约，已补齐 D=6144 主要点；不宣称覆盖所有真实模型 stride/alias、padding、并行模式或 batch-invariant 配置。

## B. 实验与性能结论

读取历史 Single 36组与三路对照36组后，主比较采用同场测量的 `decode_two_stage/h100_final.json`。历史官方源码校验仍是 commit 5df63a72。`comparison.csv` 汇总旧36组、新9组及独立重复4组，每组列出三路完整延迟、相对官方 speedup 和冷/热胜者。没有重复全部旧 Benchmark，也没有继续 N=1 专项调优。

本轮只针对公开模型的 BF16 D=6144，N=64/128/256/512/768/1024/2048/4096/8192。Synthetic case 的参数改为该公开配置的 magnitude/epsilon；Golden 与容差未变。配置和执行与旧数据相比并非同一组 scalar，故重复测试独立记录，不混并旧样本。N=128/256/512/768 另起进程再测，每次3轮，检验交叉区稳定性。

H100 80GB GPU3，PyTorch2.13.0+cu129、Triton3.7.1、CUDA12.9；复用 time_call 的 cold L2 rotation CUDA Graph/events，30次预热、25ms graph warmup、每轮100ms、3轮。三路轮流先测；热缓存同样测完整路径，关闭rotation。无分配、编译和正确性检查计入延迟。未更改warps或冻结Kernel。每个被测形状均先比较 Golden/官方、验证 Graph Replay。

已有用户 `prefill_6144/h100_6144.json` 仅6行至N=1024；记录的 --hidden 参数当前脚本不接收，现有未提交修改还引用未定义 hidden_values，不能视为可直接复现的完整Sweep。本轮不改该文件，新增独立入口避免依赖其 main。仅复用安全的 two_call 函数。

三路结果（μs；冷/热各为3轮中位数的中位数）：

| N | Official 冷 | Single 冷 | Two 冷 | Single speedup | Two speedup | Official 热 | Single 热 | Two 热 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 16.463 | 26.505 | 13.114 | 0.621 | 1.255 | 13.316 | 17.847 | 11.607 |
| 128 | 25.811 | 28.335 | 21.318 | 0.911 | 1.211 | 21.726 | 18.064 | 19.634 |
| 256 | 43.310 | 37.172 | 36.517 | 1.165 | 1.186 | 40.329 | 25.337 | 35.335 |
| 512 | 81.682 | 73.568 | 69.617 | 1.110 | 1.173 | 80.703 | 69.670 | 69.623 |
| 768 | 120.266 | 82.815 | 103.551 | 1.452 | 1.161 | 120.496 | 82.655 | 103.481 |
| 1024 | 161.298 | 119.496 | 137.438 | 1.350 | 1.174 | 161.344 | 119.122 | 137.386 |
| 2048 | 323.815 | 234.502 | 269.289 | 1.381 | 1.202 | 323.611 | 231.002 | 269.391 |
| 4096 | 651.667 | 436.172 | 534.694 | 1.494 | 1.219 | 651.342 | 436.087 | 534.854 |
| 8192 | 1299.924 | 848.939 | 1062.789 | 1.531 | 1.223 | 1300.031 | 849.245 | 1062.948 |

独立重复：

| N | Single 冷 | Two 冷 | Single 热 | Two 热 |
|---|---:|---:|---:|---:|
| 128 | 28.342 | 21.321 | 18.085 | 19.639 |
| 256 | 37.250 | 36.563 | 27.051 | 35.506 |
| 512 | 73.543 | 69.655 | 69.428 | 69.619 |
| 768 | 82.848 | 103.547 | 83.005 | 103.374 |

D=6144 判断：N=64 Two 在两种缓存状态均胜；N=128 Single 冷缓存慢于官方，但热缓存胜过Two；N=256 冷缓存Two仅约2%领先，而热缓存Single明显胜；N=512 冷缓存Two约6%领先，热缓存基本持平；N>=768 的上述已测点Single在冷/热均胜。768只是第一个测试到的稳定领先点，不是已证明的连续区间阈值。尤其不能用N=512与768之间插值写死规则。

旧36组默认配置选型摘要（具体三路延迟与加速比见comparison.csv）：

| dtype | D | N=1/8/32 | N=128 | N=1024/4096 |
|---|---:|---|---|---|
| BF16/FP16 | 4096 | Two 冷热均胜 | Single，冷缓存仅小幅优势 | Single |
| BF16/FP16 | 7168 | Two 冷热均胜 | 冷Two、热Single | Single |
| FP32 | 4096 | Two 冷热均胜 | Single | Two |
| FP32 | 7168 | Two 冷热均胜 | 冷Two、热Single | Two |

所有旧36组与新测点均至少有一种新方案超过官方（冷、热需分别查看）；没有发现两种新方案同时劣于官方的已测区域。该结论不能推广到未验证Shape或硬件。Single在小N的显著退化仍存在；Two在大N不如Single不等于比官方慢。

**Selection 决策**：不新增可执行Dispatch。真实部署的N分布未证实，中间区的冷热胜者又不同；只有元信息的静态规则不能捕获缓存状态。本轮只能确认：若实际Prefill集中在已测大N点，Single已足够且没有必要继续调Two；若真实trace包含小N Prefill，Two有保留价值。未验证Shape/硬件/布局及batch-invariant模式应保留原Runtime路径，不能绕过官方predicate硬调用三阶段Triton。尚不能用这些合成实验宣布“一律Single”或固定N阈值。

## C. 修改和验证

新增源码仅 `benchmarks/bench_prefill_selection.py`（138行）与 `tests/test_prefill_workload.py`（32行）。前者复用旧计时工具，只运行必要配置；后者补充原测试未覆盖的D=6144、公开配置参数、FP32/BF16/FP16、N=64/256/1024及更新输入后的Graph Replay。没有新增Kernel或Selection，没有改冻结代码、容差或测试预期。结果仅写本目录。

最终 pytest：108 passed（原99+新9），无失败；Ruff检查通过。所有9个新Benchmark形状和4个重复点的三路Golden/官方精度检查通过，误差指标在JSON逐项记录。FP32舍入、非FMA语义沿用已冻结Kernel；原53/99项也重新运行。

复现：

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$PWD/lab/ihc_fusion/results/.cache/triton"
lab/ihc_fusion/results/.venv/bin/python -m lab.ihc_fusion.benchmarks.bench_prefill_selection --output lab/ihc_fusion/results/prefill_selection/reproduction.json
# 独立重复追加 --tokens 128,256,512,768，并换新的 output 文件名
lab/ihc_fusion/results/.venv/bin/python -m pytest lab/ihc_fusion/tests -q -p no:cacheprovider
```

## D. 下一轮 Runtime 最小验证方案（本轮未实施）

可以进入受控、默认关闭的模型级集成验证，但不能直接启用生产替换。当前不需要更多盲目Kernel调参，最重要的下一步是真实模型的边界trace和同层端到端A/B。

1. 先提供目标checkpoint revision、dtype、TP/PP/DP/EP、Runner、编译/CUDA Graph模式、chunk budget、prefix caching配置及代表性请求集。在每个iHC边界记录N/D/C/dtype/stride、layer/subblock、scheduled与padded tokens、纯Prefill或混合batch、ubatch/rank。CPU元信息采样即可，不需要每次GPU同步。用真实layer参数/输入补验数值和选型；单纯冷/热两端实验不代表实际层间缓存。
2. 最小首轮接入仅替换同层 Attention Post → MLP Pre：在 `nvidia/hc.py` 增加局部适配，读取 MLP hc_pre 参数，接收 Attention 输出、原Residual、Attention Post Gates；在 `nvidia/model.py:_forward_ihc` 替换相邻调用。注意Kernel返回顺序为 `(output,residual,post)`，旧pre为 `(output,post,residual)`。不融合后续RMSNorm，不改Attention/MLP，第一层Pre和MLP最终Post照旧，不改PP传输协议。
3. 仅在已有批次元信息明确允许的Prefill调用启用；N绝不能当Prefill标志。混合批次和无法判定的调用先保留原路径。继续遵守HPC、batch-invariant、dtype/layout和硬件条件；不新增Scheduler调度。用已有forward context/批次metadata验证是否能无API变化获得资格条件，不能假设当前hc层已经拥有这个信息。
4. 对比真实模型logits/生成、first/last layer、chunk尾部、prefix cache、mixed batches、TP/PP、CUDA Graph capture/replay和torch.compile。测TTFT/Prefill吞吐及真实三路完整边界时间；这一步决定是否需要仅按已验证元信息的Selection。
5. 跨层MLP Post→下一层Attention Pre留到独立步骤：它需要延迟Post并携带单流/Residual/Post Gates，明确首尾和PP stage的flush，不能通过只替换一个Kernel调用完成。本轮不把这项较大改动混入最小接入。

未提交、推送、切换分支；无需白名单外修改。缺少真实部署/trace是运行时选型证据的缺口，不是Kernel正确性失败。
