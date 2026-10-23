# HY4 iHC Fusion：H200 TP8 人工部署与测试手册

审计日期：2026-10-23。本文仅由源码、Git 历史和已有报告得出；编写时未启动模型、GPU 测试或安装依赖。以下命令由操作者在标明的机器上执行。所有现场数据写到独立目录，不写回源码仓库。

使用 Bash；每个终端先执行 `set -o pipefail`，使 `命令 | tee` 保留上游失败状态。任何门禁命令失败立即停止该步骤，不能继续复制后续启动命令。后文生成的脚本会检查失败；交互终端仍需人工确认退出码和输出。路径占位符应填绝对路径，示例环境文件要求路径不含单引号或换行。

## 1. 部署就绪结论：Conditional

**具备开展 H200 TP8 诊断性测试的代码条件，尚不具备正式正确性通过、性能发布或可合入结论。** 未发现融合路径专门排斥 TP8 或 H200 的静态条件；这不等于 H200 完整模型已经验证。

| 审查项 | 结论与证据 |
|---|---|
| 当前代码 | `feat/hy4-ihc-dev`，HEAD `58fd65e9bd7c61975c9fbb929d61f18160bd3368`；审计开始时工作区干净 |
| 三个目标提交 | `300bc325c6` SM90 门禁、`ebf4432ff3` Graph 稳定条件 Dispatch、`58fd65e9bd` 测试与报告，均在 HEAD 祖先链上 |
| 基线 | 本地 `v0.32.0` 指向 `5df63a7203cdb9328d824a27b682e77c7b9928be`；不要仅凭镜像版本字符串认定源码和 ABI 一致 |
| 基线差异 | 总计 91 个文件，+12774/-12，主要是 lab、测试及文档；生产路径只有 `model.py` +120/-12 和新增 `triton_ihc_post_pre.py` +111；无 C++/CUDA 扩展、构建配置或依赖清单差异 |
| 代码完整性 | Kernel 在更早的 `990a299dee` 集成；**不能仅把上述三个提交 cherry-pick 到任意 0.32.0 镜像源码上**。下文传输完整分支 |
| 架构 | 使用精确 capability 9.0 判断，保留 SM80、SM100、未知架构 fallback；H200 必须现场验证 8 张卡实际返回 `(9,0)` |
| Graph | 已有 H100 真正 breakable capture/replay 和 GPU fused trace；完整 H200 模型的图内存、地址生命周期和所有 rank 仍待验证 |
| 数值 | **未闭合**。已有严格三层链失败，完整度有限的模型 A/A 也失败；性能一律先标记“诊断性” |
| Docker | 生产改动没有新增 C++ 编译需求；源码需要找到镜像已有扩展及打包的 FlashMLA 接口。仅设置 PYTHONPATH 未必足够；第 4 节提供有门禁的补充加载方案 |
| 全量部署 | 完整权重格式、层数、模型配置、8 卡空闲状态、显存容量及镜像匹配情况尚未现场确认 |

已有证据见 [RUNTIME_GRAPH_VALIDATION.md](RUNTIME_GRAPH_VALIDATION.md)。H100 TP4 **三层切片** N8192：Eager OFF/ON 中位数 44.273/41.990 ms，Graph OFF/ON 43.895/41.676 ms，各 15 次；约 5.16%/5.06%。这些是 Forward 诊断数据，不能外推为 H200 TP8 完整模型 E2E 收益。原始证据在开发机 `/tmp/hy4-runtime-validation`，不是部署必需品，也不是 H200 路径。

现存问题的最小处理建议：镜像不兼容时换成与基线匹配的已有镜像/扩展；数值问题另行审查冻结 Kernel 的数值语义；图内存不足时记录原始失败后，为 OFF/ON 同步选择较小 capture 集合并重新启动。本文不执行这些修复，不改 Runtime，也不以 Eager 替代 Graph 验收。

## 2. 源码与配置调用链

以下行号固定于上述 HEAD，路径相对于仓库根目录。

| 位置 | 行号与含义 |
|---|---|
| `vllm/model_executor/models/registry.py` | 133：注册名称为 `HYV4ForCausalLM`；现场检查权重 architectures，不自行改名 |
| `vllm/models/hy_v4/__init__.py` | 31–34：CUDA 进入 NVIDIA 实现 |
| `vllm/models/hy_v4/nvidia/model.py` | 111–124：初始化 iHC 与融合总门禁 |
| 同上 | 204–256：运行时 eligibility；258–278：融合/物化 Post 后普通 Pre 的 fallback |
| 同上 | 280–311：Attention Pre → Attention → 同层 PostPre → MLP → pending/末尾 Post |
| 同上 | 363–368：仅本 PP stage 的非末层允许延迟 Post；473–501：首层 pending=None、逐层传递、PP flatten、HCHead 与 norm |
| `vllm/models/hy_v4/nvidia/hc.py` | 101–155：原始 iHC Pre；157 起为 Post |
| `vllm/models/hy_v4/nvidia/triton_ihc_post_pre.py` | 11：GPU Kernel 名 `_fused_post_pre`；74–111：分配输出并以 grid `(N,)` 启动 |
| `vllm/platforms/interface.py` | 554–574：`is_device_capability(90)` 是精确匹配，不是“至少 SM90” |
| `vllm/v1/worker/gpu/model_runner.py` | 1999 起 FULL、2035 起 PIECEWISE 调度；本手册选择后者 |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | 565–576：`BreakableCUDAGraphWrapper` 与 `run_pw_graph` |
| `vllm/compilation/breakable_cudagraph.py` | 324 起按 descriptor 取图；367 起 capture；414 起 replay 与 debug 地址检查 |
| `vllm/models/hy_v4/nvidia/attention.py` | 709 起预分配 Attention 输出；745 起已有 eager break |
| `vllm/config/vllm.py` | 703–741：V2 runner 选择；812–840：breakable 开启时 compilation mode 设为 NONE |
| `vllm/engine/arg_utils.py` | 975 `--enforce-eager`；1010 `--hf-overrides`；1853 `--compilation-config`；1870 `--profiler-config` |
| `vllm/config/profiler.py` | 69 起 profiler、目录和活动；`vllm/entrypoints/serve/profile/api_router.py:35,48` 为 start/stop；`gpu_worker.py:1386` 起按 TP rank 命名 |
| `vllm/benchmarks/serve.py` | 1592 起 `bench serve` 参数；1733 warmups、1739 profile、1744 保存；`datasets/datasets.py:1888` 起 random 参数、2430 起 custom JSONL |

### 2.1 真正的 ON/OFF

没有本任务专用的 `VLLM_IHC_FUSION` 环境变量。实际开关是 HF 配置字段，初始化时读取，**必须是 JSON 布尔值**：

```json
{"enable_ihc_post_pre_fusion": true, "enable_ihc_cross_layer_fusion": true}
```

两个字段缺省均为 false。第一个控制融合总开关，第二个允许跨层延迟 Post；只开第二个不会产生融合。OFF 显式同时设 false。模型自身还必须启用 `enable_ihc`，不得为了命中而把非 iHC 权重强制改成 iHC。

其他条件：D=6144、C=4、BF16 x/residual，FP32 post gates 和 Pre weight/scale/base，相关 Tensor 同 CUDA 设备且 contiguous；`512 ≤ N ≤ 8192`；无 speculative config、无 `VLLM_BATCH_INVARIANT`、无 ubatch，且不在 `torch.compiler.is_compiling()` 中。

- **Eager**：还需可识别的 MLA metadata、纯 Prefill、没有 Decode token、实际 token 数等于 Tensor N。混合批次、未知 metadata、padding 等会回退。
- **Graph**：FULL 或 breakable PIECEWISE 可进入；本轮只测已有模型级证据的 breakable PIECEWISE。基于静态 Tensor 条件，不再依据 capture 时的 Prefill/Decode 数量。混合、Decode、padding 批次只要物理 N 合格，也可能保留融合；不是错误命中。
- **小 N**：判断的是物理 N。128 的图保留 128 capture bucket 才能验证小 N fallback；如果被 pad 到 512，则可能融合。
- 首层 Attention Pre 和末层 MLP Post 必须保留。pending 是单次模型 forward 的局部 `(residual, post_gates)`，不是模型持久状态。消费者不合格时先完成生产者 Post，再普通 Pre；stage 末层不传 pending 到 PP。
- `empty_like` 在 capture 时分配，replay 复用图持有的地址；不能捕获后动态切开关。已知同进程实验性额外建图曾出现 IMA，原因未闭合，因此四个模式各自新进程。
- 没有 TP=4 的融合门禁；TP8 的通信、Attention 头划分、权重加载和内存仍由完整模型验证。PP>1 仅审查边界逻辑，本手册固定 PP=1，不声称实测 PP。

## 3. Phase 0A：现场环境与离线同步

### 3.1 H100 开发机：导出固定分支

**目的/前提**：只读确认所测版本，导出已有 Git 对象。当前 HEAD 应等于审计 HEAD；不打包权重、venv、缓存和 `/tmp` 测试数据。以下操作由人执行，当前文档编写未执行导出。

```bash
cd /data/qiuyuhao/projects/vllm-hy4-dev
pwd
git branch --show-current
git status --short
git log -5 --oneline
git worktree list
git rev-parse HEAD v0.32.0
git merge-base --is-ancestor 300bc325c6 HEAD
git merge-base --is-ancestor ebf4432ff3 HEAD
git merge-base --is-ancestor 58fd65e9bd HEAD
git diff v0.32.0..HEAD --stat
git diff --exit-code HEAD -- vllm
test "$(git rev-parse HEAD)" = 58fd65e9bd7c61975c9fbb929d61f18160bd3368

EXPORT_DIR=$(mktemp -d /tmp/hy4-h200-export.XXXXXX)
git bundle create "$EXPORT_DIR/hy4.bundle" feat/hy4-ihc-dev v0.32.0
git bundle verify "$EXPORT_DIR/hy4.bundle"
cp lab/ihc_fusion/H200_TP8_MANUAL_TEST.md "$EXPORT_DIR/"
cd "$EXPORT_DIR"
sha256sum hy4.bundle H200_TP8_MANUAL_TEST.md > SHA256SUMS
ls -lh
# 将下面两个占位符替换为现场值；也可经内网中转/移动介质复制整个目录。
scp hy4.bundle H200_TP8_MANUAL_TEST.md SHA256SUMS \
  '<H200_SSH_HOST>:<H200_TRANSFER_DIRECTORY>/'
```

**为什么**：bundle 包含完整祖先链和基线，不依赖 H200 访问 GitHub；单独复制未提交手册。SHA256 保证传输一致。**PASS**：三个祖先检查及 HEAD 检查成功，生产代码无未提交修改；手册是唯一预期新文件。**FAIL**：版本不符或源码脏，停止导出，不 checkout/reset。保存上述 Git 输出和 SHA256SUMS；传输错误先核对目录权限和两端校验值。

### 3.2 H200 宿主机：硬件、镜像、权重和解包

先填现场变量。尖括号是占位符，不能原样执行；源码目标必须是新目录，模型路径必须是**本地完整权重**，不是三层切片。

```bash
export TRANSFER='<H200_TRANSFER_DIRECTORY>'
export HOST_SRC='<NEW_H200_SOURCE_DIRECTORY>'
export HOST_MODEL='<EXISTING_FULL_HY4_WEIGHTS_DIRECTORY>'
export HOST_RUN='<NEW_H200_RESULTS_DIRECTORY>'
export IMAGE='<EXISTING_VLLM_0_32_0_IMAGE_TAG_OR_DIGEST>'
export CONTAINER='<NEW_UNIQUE_CONTAINER_NAME>'
mkdir -p "$HOST_RUN"
git --version | tee "$HOST_RUN/git-version.txt"
nvidia-smi -L | tee "$HOST_RUN/gpus.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv \
  | tee "$HOST_RUN/gpu-state-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
  | tee "$HOST_RUN/gpu-processes-before.csv"
nvidia-smi topo -m | tee "$HOST_RUN/topology.txt"
docker version | tee "$HOST_RUN/docker-version.txt"
docker ps --no-trunc | tee "$HOST_RUN/docker-processes.txt"
docker image inspect "$IMAGE" > "$HOST_RUN/image.json"
test -r "$HOST_MODEL/config.json"
ls -lh "$HOST_MODEL" | tee "$HOST_RUN/model-files.txt"
df -h "$HOST_MODEL" "$HOST_RUN" | tee "$HOST_RUN/disk.txt"
cd "$TRANSFER"
sha256sum -c SHA256SUMS
test ! -e "$HOST_SRC"
git clone --branch feat/hy4-ihc-dev "$TRANSFER/hy4.bundle" "$HOST_SRC"
git -C "$HOST_SRC" bundle verify "$TRANSFER/hy4.bundle"
git -C "$HOST_SRC" rev-parse HEAD v0.32.0 | tee "$HOST_RUN/source-head.txt"
git -C "$HOST_SRC" status --short | tee "$HOST_RUN/source-status.txt"
test "$(git -C "$HOST_SRC" rev-parse HEAD)" = 58fd65e9bd7c61975c9fbb929d61f18160bd3368
cp "$TRANSFER/H200_TP8_MANUAL_TEST.md" "$HOST_RUN/"
```

**目的**：确认可独占的 8 张 H200、驱动和互联、已有镜像及完整权重。**PASS 门禁**：8 张目标卡可用，无其他任务争用；镜像确实本地存在；权重目录可读；bundle/hash/HEAD 正确。**FAIL**：卡被占用则另约窗口，不杀其他任务；路径或版本不符则先查挂载/传输。保存所有输出。`nvidia-smi` 的 CUDA Version 是驱动支持上限，不是容器 PyTorch 编译 CUDA 版本。

确认这 8 张卡允许使用后进入容器。下面是新容器示例，不覆盖已有容器，也不拉取镜像：

```bash
docker run --pull=never --name "$CONTAINER" --gpus all \
  --ipc=host --network host \
  -v "$HOST_SRC:/work/hy4:ro" \
  -v "$HOST_MODEL:/models/hy4:ro" \
  -v "$HOST_RUN:/results" \
  --entrypoint /bin/bash -it "$IMAGE"
```

只在宿主机恰有这 8 张目标卡时使用 `--gpus all`；多于 8 卡时改为经确认的 Docker device 列表，并记录 UUID 映射。`--ipc=host` 供 TP 通信，`--network host` 便于同机客户端；服务后面仅绑定 `127.0.0.1`。若现场已有合适容器，可使用已有挂载，但须把下文路径相应统一，且保留 `docker inspect`。不要传模型权重。

## 4. Phase 0B：容器 Python、源码和扩展加载

### 4.1 创建使用镜像现有包的隔离执行目录

**H200 Docker 容器内**。不安装/升级包，不执行 editable install。需要镜像已有 `uv` 和 Python；缺失时记录环境阻塞，使用已有合适环境，不临时联网安装。

```bash
export RUN_ROOT=/results/manual
export HY4_SOURCE=/work/hy4
export MODEL=/models/hy4
mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"
command -v uv
command -v python
# 只创建隔离解释器入口，继承镜像现有 site-packages；禁止下载解释器。
UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never uv venv --system-site-packages \
  --python "$(command -v python)" "$RUN_ROOT/.venv"
env -u PYTHONPATH .venv/bin/python - <<'PY' | tee image-python.txt
import importlib.metadata as m
import platform
import torch, triton, vllm
print('Python', platform.python_version())
for p in ('torch', 'triton', 'vllm', 'transformers', 'flashinfer-python'):
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, 'NOT INSTALLED')
print('image vllm', vllm.__file__)
print('torch CUDA build', torch.version.cuda)
print('CUDA available / count', torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
for i in range(8):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    assert torch.cuda.get_device_capability(i) == (9, 0)
PY
export IMAGE_VLLM_DIR=$(env -u PYTHONPATH .venv/bin/python -c \
  'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')
nvidia-smi | tee container-nvidia-smi.txt
git -C "$HY4_SOURCE" rev-parse HEAD v0.32.0 | tee container-source.txt
```

**PASS**：版本/路径清楚，镜像 package 与后续 source 不同，8 卡都是 H200/SM90。**FAIL**：停止启动；先查 Docker GPU 暴露、镜像 Python、驱动兼容。历史 H100 环境是 Torch `2.13.0+cu129`、Triton `3.7.1`、vLLM `0.32.0+cu129`，这些是参考证据，不应自动安装这些版本。CUDA Runtime/ABI 还须由下一步扩展导入和 OFF 模型启动验证。

### 4.2 补充镜像二进制搜索路径，确保所有 worker 使用分支源码

**为什么不直接 PYTHONPATH 或重装**：本 checkout 的普通 `vllm` package 会遮住镜像 package，Python 不会自动合并其中的 `.so`；checkout 还只有 FlashMLA package 占位文件。已有 H100 验证使用临时 `sitecustomize.py` 补充加载。下方沿用此方式，动态定位镜像路径，文件只在结果目录。Python 主体仍来自只读 checkout；打包扩展/缺失第三方接口来自镜像，必须记录。

```bash
cd "$RUN_ROOT"
mkdir -p bootstrap
cat > bootstrap/sitecustomize.py <<'PY'
import importlib.util
import os
import sys
import traceback
from pathlib import Path

try:
    import torch
    import vllm
    source = Path(os.environ['HY4_SOURCE']).resolve()
    installed = Path(os.environ['IMAGE_VLLM_DIR']).resolve()
    assert Path(vllm.__file__).resolve() == source / 'vllm/__init__.py'
    assert installed != source / 'vllm'
    vllm.__path__.append(str(installed))
    loaded = []
    for name in ('_vllm_fa2_C', '_vllm_fa3_C'):
        candidates = sorted((installed / 'vllm_flash_attn').glob(name + '*.so'))
        if not candidates:
            continue
        assert len(candidates) == 1, candidates
        qualified = 'vllm.vllm_flash_attn.' + name
        spec = importlib.util.spec_from_file_location(qualified, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded.append(qualified)
    assert loaded, 'No packaged FlashAttention extension found'
    import vllm.third_party.flashmla
    vllm.third_party.flashmla.__path__.append(str(installed / 'third_party/flashmla'))
    vllm.third_party.__path__.append(str(installed / 'third_party'))
except Exception:
    traceback.print_exc()
    # Python ordinarily swallows a sitecustomize failure; fail closed here.
    os._exit(78)
PY
cat > env.sh <<EOF
set -o pipefail
export RUN_ROOT='$RUN_ROOT'
export HY4_SOURCE='$HY4_SOURCE'
export MODEL='$MODEL'
export IMAGE_VLLM_DIR='$IMAGE_VLLM_DIR'
export PYTHONPATH='$RUN_ROOT/bootstrap:$HY4_SOURCE'
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME='$RUN_ROOT/hf-cache'
export TRITON_CACHE_DIR='$RUN_ROOT/triton-cache'
export VLLM_CACHE_ROOT='$RUN_ROOT/vllm-cache'
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=0
export PORT=8000
export GPU_MEMORY_UTILIZATION=0.85
EOF
source env.sh
.venv/bin/python - <<'PY' | tee source-imports.txt
import importlib
import importlib.metadata as meta
import inspect
import os
import sys
from pathlib import Path
import vllm
root = Path(os.environ['HY4_SOURCE']).resolve()
print('distribution version', meta.version('vllm'))
print('source package', vllm.__file__, list(vllm.__path__))
for name in (
    'vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch',
    'vllm._flashmla_C', 'vllm._flashmla_extension_C',
    'vllm.third_party.flashmla.flash_mla_interface',
    'vllm.models.hy_v4.nvidia.model',
    'vllm.models.hy_v4.nvidia.triton_ihc_post_pre',
    'vllm.entrypoints.cli.main',
):
    module = importlib.import_module(name)
    print(name, module.__file__)
    if name.startswith('vllm.models.hy_v4') or name == 'vllm.entrypoints.cli.main':
        assert Path(module.__file__).resolve().is_relative_to(root)
from vllm.models.hy_v4.nvidia.model import HYV4DecoderLayer
assert 'is_device_capability(90)' in inspect.getsource(HYV4DecoderLayer.__init__)
from vllm.platforms import current_platform
assert current_platform.is_cuda() and current_platform.is_device_capability(90)
for name, module in sorted(sys.modules.copy().items()):
    path = getattr(module, '__file__', '') or ''
    if name.startswith('vllm.') and path.endswith('.so'):
        print('LOADED_EXTENSION', name, path)
PY
.venv/bin/python -m vllm.entrypoints.cli.main serve --help > serve-help.txt
.venv/bin/python -m vllm.entrypoints.cli.main bench serve --help > bench-help.txt
sha256sum "$HY4_SOURCE/vllm/models/hy_v4/nvidia/model.py" \
  "$HY4_SOURCE/vllm/models/hy_v4/nvidia/triton_ihc_post_pre.py" \
  bootstrap/sitecustomize.py env.sh > deployment.sha256
```

**门禁**：每条 Python 命令退出码必须为 0。检查 `source-imports.txt` 中 Runtime、Kernel、CLI 均来自 `/work/hy4`，扩展来自记录的镜像路径；`serve-help.txt`/`bench-help.txt` 确实有本文参数。若 `.so` 缺失、undefined symbol、FlashMLA Python 接口缺失、依赖 API 不匹配：**Blocked**，先保存完整报错、镜像 digest、Torch/CUDA/扩展路径，换匹配环境后重验，不临时复制未知 `.so` 或升级包。导入通过只是必要条件；量化、Attention/MoE 算子的运行兼容性由 Phase 1 OFF 验证。

本分支生产差异不要求重新构建 vLLM CUDA/C++ Extension；融合 Kernel 需要 Triton 首次 JIT。**这不保证任意标称 v0.32.0 的镜像都能复用其扩展。** 源码 `_version` 缺失可能显示开发版本，不能据此判断加载错误；以文件路径、HEAD、hash 为准。此方案无需 editable install，也不向镜像包目录写文件。

### 4.3 检查完整权重与有效配置

**目的/前提**：扩展门禁通过；确认完整模型是本融合支持的 iHC D6144/C4，量化元数据与现有加载器匹配。不要猜权重为 MXFP8，不强行传 `--quantization`。

```bash
cd "$RUN_ROOT"
source env.sh
.venv/bin/python - <<'PY' | tee weights-check.txt
import json
import os
from pathlib import Path
from vllm.transformers_utils.config import get_config
p = Path(os.environ['MODEL'])
raw = json.loads((p / 'config.json').read_text())
print('config.json', json.dumps(raw, ensure_ascii=False, indent=2))
cfg = get_config(str(p), trust_remote_code=True)
for key in ('architectures', 'model_type', 'num_hidden_layers', 'hidden_size',
            'hc_mult', 'enable_ihc', 'torch_dtype', 'quantization_config',
            'num_attention_heads', 'vocab_size', 'max_position_embeddings'):
    print('EFFECTIVE', key, getattr(cfg, key, None))
assert cfg.hidden_size == 6144 and cfg.hc_mult == 4 and cfg.enable_ihc
assert cfg.num_hidden_layers > 3, 'This is not the intended full model'
for index in p.glob('*.index.json'):
    data = json.loads(index.read_text())
    if 'weight_map' in data:
        shards = sorted(set(data['weight_map'].values()))
        missing = [s for s in shards if not (p / s).is_file()]
        print('INDEX', index.name, 'shards', len(shards), 'missing', missing)
        assert not missing
print('WEIGHT_FILES', [(x.name, x.stat().st_size) for x in p.iterdir()
                       if x.suffix in ('.safetensors', '.bin', '.pt')])
PY
sha256sum "$MODEL/config.json" > model-config.sha256
```

**PASS**：再由操作者对照模型发布清单确认层数、分片总量和 tokenizer 是完整 HY4；`>3` 不能独自证明完整。有权重 manifest 时现场校验，避免为首轮测试反复全量 hash 大权重。记录 ModelOpt/FP8/MXFP8 等实际元数据、加载日志中的检测结果；二者不一致就停止。模型支持的上下文应至少 8448；否则本手册长度配置不适用，记录阻塞，不能偷偷截断输入。`trust_remote_code=True` 仅用于经确认的本地模型包，与后续服务保持一致。

## 5. 公共准备：固定请求、四个独立进程

以下均为 **H200 Docker 容器内**。每个终端先 `cd /results/manual; source env.sh`。第二终端可从宿主机执行 `docker exec -it "$CONTAINER" /bin/bash`。

### 5.1 一次性生成请求和 benchmark 输入

**目的**：直接 token IDs 验证精确 N；单独保存 benchmark 文本数据，避免不同模式重新随机生成或 chat template 改变输入。此处不启动模型。

```bash
cd "$RUN_ROOT"
source env.sh
mkdir -p inputs
.venv/bin/python - <<'PY' | tee inputs/manifest.txt
import hashlib
import json
from pathlib import Path
from vllm.benchmarks.datasets.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer
import os
tok = get_tokenizer(os.environ['MODEL'], trust_remote_code=True)
allowed = [i for i in range(tok.vocab_size) if i not in set(tok.all_special_ids)]
assert len(allowed) > 512
p = Path('inputs')
for n in (128, 512, 2048, 8192):
    for variant, offset in (('a', 0), ('b', 17)):
        ids = [allowed[100 + (i + offset) % 197] for i in range(n)]
        request = dict(model='hy4-test', prompt=ids, max_tokens=1,
                       temperature=0, seed=7, ignore_eos=True,
                       add_special_tokens=False, stream=False)
        (p / f'n{n}_{variant}.json').write_text(json.dumps(request))
        request.update(logprobs=1, logprob_token_ids=allowed[100:116],
                       return_tokens_as_token_ids=True)
        (p / f'n{n}_{variant}_numerical.json').write_text(json.dumps(request))
    rows = RandomDataset(random_seed=7).sample(
        tokenizer=tok, num_requests=64, prefix_len=0,
        range_ratio=0.0, input_len=n, output_len=32)
    target = p / f'bench_n{n}.jsonl'
    target.write_text(''.join(json.dumps({'prompt': r.prompt, 'output_tokens': 32},
                                       ensure_ascii=False) + '\n' for r in rows))
    lengths = [len(tok(r.prompt).input_ids) for r in rows]
    print(target, 'requested', n, 'actual_min_max', min(lengths), max(lengths),
          'sha256', hashlib.sha256(target.read_bytes()).hexdigest())
    assert max(lengths) + 32 <= 8448
PY
sha256sum inputs/*.json inputs/*.jsonl > inputs/SHA256SUMS
```

**PASS**：文件和 hash 固定；直接 IDs 请求长度精确；benchmark 实际重编码长度已记录。随机 token 的 decode/re-encode 可能不能精确达到目标，不能把文件名 N8192 当成实际 N。**FAIL**：tokenizer/模型配置不匹配先解决路径，不重新改数据直到“看起来有收益”。保存整个 inputs 目录。当前 CLI 中 `--random-range-ratio 0.0` 才是固定长度分布，1.0 不合法；上面直接调用同一实现并冻结结果。

### 5.2 创建唯一启动入口

**目的**：四模式除开关和 Graph 外配置一致；每次运行目录和服务进程独立，进程内不切换图。`GPU_MEMORY_UTILIZATION=0.85` 是待现场容量验证的起始值，不是 H200 容量保证。若需调整，四模式一致并记录新试验批次。

```bash
cat > "$RUN_ROOT/start-server.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /results/manual
source env.sh
: "${ARM:?choose eager_off/eager_on/graph_off/graph_on}"
: "${RUN_ID:?unique result directory name}"
: "${PURPOSE:?profile or perf}"
case "$ARM" in
  eager_off) FUSED=false; GRAPH=false ;;
  eager_on)  FUSED=true;  GRAPH=false ;;
  graph_off) FUSED=false; GRAPH=true ;;
  graph_on)  FUSED=true;  GRAPH=true ;;
  *) exit 2 ;;
esac
CASE_DIR="$RUN_ROOT/$RUN_ID"
mkdir "$CASE_DIR"
ARGS=(serve "$MODEL" --served-model-name hy4-test --host 127.0.0.1
  --port "$PORT" --tensor-parallel-size 8 --pipeline-parallel-size 1
  --distributed-executor-backend mp --dtype bfloat16 --trust-remote-code
  --seed 7 --generation-config vllm --max-model-len 8448
  --max-num-batched-tokens 8192 --max-num-seqs 4
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --no-enable-prefix-caching --enable-chunked-prefill --no-enable-dbo
  --hf-overrides "{\"enable_ihc_post_pre_fusion\":$FUSED,\"enable_ihc_cross_layer_fusion\":$FUSED}")
if "$GRAPH"; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  ARGS+=(--compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[128,512,2048,8192]}')
else
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  ARGS+=(--enforce-eager --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}')
fi
case "$PURPOSE" in
  profile)
    export VLLM_LOGGING_LEVEL=DEBUG
    ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$CASE_DIR/traces\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}") ;;
  perf) export VLLM_LOGGING_LEVEL=INFO ;;
  *) exit 2 ;;
esac
printf '%q ' .venv/bin/python -m vllm.entrypoints.cli.main "${ARGS[@]}" > "$CASE_DIR/command.txt"
printf '\n' >> "$CASE_DIR/command.txt"
env | sort | grep -E '^(VLLM_|CUDA_|NCCL_|PYTHONPATH|TRITON_|HF_|ARM=|PURPOSE=|RUN_ID=)' > "$CASE_DIR/environment.txt"
date -u +%FT%TZ > "$CASE_DIR/start-time.txt"
.venv/bin/python -m vllm.entrypoints.cli.main "${ARGS[@]}" 2>&1 | tee "$CASE_DIR/server.log"
SH
bash -n "$RUN_ROOT/start-server.sh"
```

`--dtype bfloat16` 固定激活，量化权重仍按元数据加载；TP8/PP1/mp 固定拓扑；不传 speculative、HiSparse 或其他优化参数。关闭 prefix cache 避免重复输入直接 cache hit；开启 chunked prefill 并固定 token budget。`mode:0` 禁用 torch.compile，**不禁用 CUDA Graph**；Graph 由 `cudagraph_mode` 和 breakable wrapper 控制。性能进程不启用 profiler，不使用 DEBUG 日志。

**PASS**：脚本语法检查成功；启动时日志实际配置一致。**FAIL**：参数不识别先查导入路径和保存的 help，不删除参数凑启动。启动入口只存在结果目录，不修改 Runtime。

## 6. Phase 1：完整模型 Eager TP8 OFF / ON

### 6.1 启动 A，再独立启动 B

**前提**：Phase 0 全部门禁通过，8 卡独占，无旧服务占用端口。以下四种启动都是前台运行；每次在另一个终端执行客户端检查。

```bash
# 终端 S：A。等待 ready，然后在终端 C 执行 6.2–6.4。
ARM=eager_off RUN_ID=eager_off_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"
```

完成 A 后在终端 S 按 Ctrl-C 正常退出；确认其 worker 退出和端口释放，再执行：

```bash
# 终端 S：B。不能与 A 并行占用 8 张卡。
ARM=eager_on RUN_ID=eager_on_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"
```

**目的/预期**：先证明完整模型 OFF 可加载和推理，再验证 ON；日志应为 TP8/PP1、8 rank 正常、`enforce_eager=True` 和 Graph NONE，无模型图 capture。检查加载层数、量化类型、权重加载警告。**PASS**：完整权重加载、服务 ready、没有 rank/NCCL/缺权重错误；6.3 中模型图 launch 为零。**FAIL**：OFF 已失败属于部署/原路径阻塞，暂不归因融合；保存完整 server.log，查显存、分片、扩展和互联。所有 rank 成功以日志拓扑加 8 份 worker GPU trace 证明，单看 HTTP 200 不够。

### 6.2 客户端请求、预热和更新输入

**终端 C，容器内**。每次启动后将 RUN_ID 设成该服务目录；以下流程 A/B/C/D 相同。

```bash
cd /results/manual
source env.sh
export RUN_ID=eager_off_profile   # B/C/D 分别改为对应 *_profile
export CASE_DIR="$RUN_ROOT/$RUN_ID"
curl --fail-with-body -sS "http://127.0.0.1:$PORT/health"
curl --fail-with-body -sS "http://127.0.0.1:$PORT/v1/models" | tee "$CASE_DIR/models.json"
export N=8192
# 先做 8 次相同形状预热；不在 profiler 或性能计时窗口内。
for i in $(seq 1 8); do
  curl --fail-with-body -sS "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' --data-binary @"inputs/n${N}_a.json" \
    -o "$CASE_DIR/warmup_n${N}_${i}.json" || break
done
# 新输入与重复新输入，均在同一个已捕获 shape 下测试。
for label in a b a b; do
  curl --fail-with-body -sS "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' --data-binary @"inputs/n${N}_${label}.json" \
    | tee -a "$CASE_DIR/updated_n${N}.jsonl"
  printf '\n' >> "$CASE_DIR/updated_n${N}.jsonl"
done
```

**参数解释**：直接 token IDs、`add_special_tokens:false`、max_tokens=1，减少混合 Decode 干扰，单请求 C=1；temperature=0、seed=7 固定。N8192 主测试后依次 `export N=512`、`128`、`2048` 重复本步骤及下一步，记录各自文件。**PASS**：全部返回成功，usage.prompt_tokens=N、completion_tokens=1，无 NaN/错误、无 rank 退出。a/b 改变输入但相同长度；相同输出 token 本身不能证明或否定陈旧状态，结合 Phase 4 的数值诊断。**FAIL**：任一预热失败就停，不继续计时。保存 JSON、服务日志和当时 GPU 状态。

### 6.3 独立 GPU trace：验证真实执行

**目的**：把真实融合与仅开配置区分开。前提：上述 shape 已预热，当前没有其他客户端请求。一次 profile 会话只包含一个 O=1 请求，便于推导调用次数。

```bash
curl --fail-with-body -sS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'Content-Type: application/json' \
  -d "{\"profile_prefix\":\"n${N}_a\"}"
curl --fail-with-body -sS "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' --data-binary @"inputs/n${N}_a.json" \
  -o "$CASE_DIR/profile_n${N}.json"
curl --fail-with-body -sS -X POST "http://127.0.0.1:$PORT/stop_profile"
ls -lh "$CASE_DIR/traces"
```

如果中间请求失败，也应执行 stop_profile 导出诊断；该次不能记 PASS。每个 rank 的 trace 文件名含 `tp0`…`tp7`、`rank0`…`rank7`。以下解析只读已导出的 GPU kernel events，不统计 CPU Python 函数或 JIT 缓存。

```bash
cat > "$RUN_ROOT/read-traces.py" <<'PY'
import collections
import gzip
import json
import os
from pathlib import Path

root = Path(os.environ['CASE_DIR']) / 'traces'
prefix = 'n' + os.environ['N'] + '_a_'
names = ('_fused_post_pre', '_ihc_pre_stage1', '_ihc_pre_stage2', '_ihc_post_kernel')
files = sorted(p for p in root.rglob('*') if p.is_file()
               and p.name.startswith(prefix)
               and (p.name.endswith('.json') or p.name.endswith('.json.gz')))
assert files, 'No worker traces for this request'
for path in files:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        events = json.load(f)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    counts = {name: sum(name in e.get('name', '') for e in kernels) for name in names}
    graph_launches = sum('cudaGraphLaunch' in e.get('name', '') for e in events)
    grids = collections.Counter(
        (name, str(e.get('args', {}).get('grid', 'MISSING')))
        for e in kernels for name in names if name in e.get('name', ''))
    print(path.name, 'gpu_kernel_events', len(kernels), 'counts', counts,
          'cudaGraphLaunch_events', graph_launches, 'grids', dict(grids))
    for e in kernels:
        if '_fused_post_pre' in e.get('name', ''):
            print('FUSED_FIRST_EVENT', e)
            break
print('TRACE_FILES', len(files), '(verify ranks 0..7 individually)')
PY
.venv/bin/python read-traces.py | tee "$CASE_DIR/trace_n${N}_summary.txt"
```

**判定（L 为现场完整模型实际层数，PP=1）**：单次无分块的纯 Prefill、所有层条件合格时，每个 rank：

| 模式/物理 N | fused | 原 Pre stage1 | 原 Pre stage2 | 原 Post |
|---|---:|---:|---:|---:|
| OFF | 0 | 2L | 2L | 2L |
| ON，512…8192 | 2L−1 | 1 | 1 | 1 |
| ON，物理 N128 | 0 | 2L | 2L | 2L |

**PASS**：8 个 rank 均有 GPU kernel events，ON 命中 `_fused_post_pre` 且计数和原路径减少符合预期；OFF 为零。融合 grid.x 即物理 N（源码 grid `(N,)`），必须保存。若额外分块、重复 forward 或 trace 范围不干净，不直接套表，先按时间线分离每次 forward 再核对。HCHead 等其他 kernel 不计入表。

Eager 的 HY4 forward 不应有模型 `cudaGraphLaunch`。Graph 则还须满足第 7 节，**仅看到 fused GPU event 不足以证明它在 replay 中**。若 trace 的 GPU 活动缺失，标记可观测性 BLOCKED；不能用编译日志、返回文本或更快的耗时替代。

### 6.4 保存退出状态

**目的**：防止下一模式与旧 worker 重叠。完成该模式的请求、trace、Phase 4 诊断后，在终端 S Ctrl-C；不要用宽泛 `pkill`。

```bash
# 容器内：服务退出后检查；预期 curl 连接失败。
curl -sS "http://127.0.0.1:$PORT/health"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
  | tee "$CASE_DIR/gpu-processes-after.csv"
```

**PASS**：该次 API server 和 8 个 worker 正常退出，显存释放；其他现场进程不受影响。若 worker 未退出，先保存 PID 和日志，处理明确属于本次运行的进程后再启动下一模式。保存 shell 退出状态和时间。

## 7. Phase 2：完整模型 breakable CUDA Graph TP8

### 7.1 C/D 独立启动

```bash
# 终端 S：C。完成 6.2–6.4、下面的 Graph 验收和 Phase 4 后退出。
ARM=graph_off RUN_ID=graph_off_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"
```

```bash
# 终端 S：D。确认 C 进程退出后执行。
ARM=graph_on RUN_ID=graph_on_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"
```

**前提**：Eager OFF/ON 启动和执行证据齐全；不可将 C/D 改成 `--enforce-eager`。客户端 RUN_ID 分别改为 `graph_off_profile`、`graph_on_profile`，重复第 6 节 N8192/512/128/2048 的预热、请求和 trace。

**配置解释**：`VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_USE_BREAKABLE_CUDAGRAPH=1`、`mode:0`、`cudagraph_mode:PIECEWISE`。不是默认主要针对 decode 的 capture sizes，也不是普通 torch.compile PIECEWISE。显式 `[128,512,2048,8192]` 覆盖四个目标物理 shape；中间长度可能 pad 到更大 bucket，必须看 trace，不能只看请求长度。8192 是本配置最大调度 token budget，捕获最大图可能消耗较多显存。

### 7.2 capture/replay 验收

```bash
# 终端 C：检查启动/预热日志，不把 capture 时长当成 replay 时长。
grep -nE 'Captured breakable|cudagraph|CUDA graph|enforce_eager|compilation_config|rank|Error|Traceback' \
  "$CASE_DIR/server.log" > "$CASE_DIR/graph-log-excerpts.txt"
cat "$CASE_DIR/graph-log-excerpts.txt"
```

**PASS 必须同时满足**：

1. 配置显示 Eager 关闭、PIECEWISE 开启；DEBUG 日志有相应 batch descriptor 的 `Captured breakable cudagraph`，捕获完成后服务可运行。首轮 JIT/模型加载/capture 时间单列为启动成本。
2. 对同一 shape 多次请求、a/b 更新输入均成功；日志无非法访存、地址检查失败、rank 退出或新增持续 capture。捕获入口含地址检查，但不能因此宣称所有生命周期问题已被证明不存在。
3. **预热后的**每个 rank trace 有 `cudaGraphLaunch`；在 Perfetto/Chrome trace 本地查看时间线，将 fused kernel 的 correlation/graph node 信息关联到 graph replay。Attention 的 eager break 仍会执行，这是设计行为，不要求整个 forward 只有一个 graph。
4. D 在适格 shape 的 replay 中有 `2L−1` fused，原始 Pre/Post 各保留首尾边界；C 的 fused 为 0；物理 N128 两者均回退。不能只凭整段 trace 同时存在 graph launch 和 fused 就断定后者属于图，必须看节点/相关 launch。
5. 8 个 rank 全部完成，a/b 数值记录没有可明确识别的新陈旧输出/跨请求污染；严格数值状态仍按 Phase 4 单独标记。

如果 PyTorch trace 缺失 graph node 关联或无法区分 eager/captured fused，记录“Graph 运行可见、融合归属未确认”，验收 **PARTIAL**，保留原 trace 供 GPU profiler 分析，不凭日志填 PASS。不临时部署新 profiler 框架。若 capture OOM，保存失败和峰值显存，再把 **C/D 同时**缩小为 `[128,512,2048]`、相应限制测试范围并重新启动；N8192 Graph 验收仍记未完成，不能沿用原性能对比。

## 8. Phase 3：完整模型 E2E 性能（人工执行）

### 8.1 输入含义与最小矩阵

- 请求 ISL 是每个请求的 token 数；`max-num-batched-tokens=8192` 限制一次调度的总 token budget，不等于每个请求的长度。
- chunked prefill 可能把一个请求拆成多个 forward；并发请求可在同一 forward 合并。Tensor N 是本次 forward 的物理行数，Graph 还可能 padding。
- C=4、ISL8192 不意味着 Tensor N=32768；本配置每步预算最多 8192，再按 graph bucket 选图。C=4、OSL32 时 Eager 可能形成混合 Prefill/Decode，触发合法 fallback，而 Graph 静态合格 shape 可能仍融合。
- 先用单请求精确 IDs trace 建立命中证据，再对所测 benchmark 工作负载做短 profile，确认其实际 N 分布。单请求 trace 不能代替并发 workload 命中证据。

首轮固定下表，作为输入规模覆盖，不声称它代表未知的线上流量分布。后续若有实际 HY4 请求分布再据此选取，本文不扩展调优。

| ISL 目标 | 最大并发 C | OSL | 目的 |
|---:|---:|---:|---|
| 8192 | 1 | 1 | 主 Prefill 延迟/吞吐诊断 |
| 8192 | 4 | 1 | 并发 Prefill、合并/分块影响 |
| 2048 | 1 | 32 | 中等 Prefill，并记录 ITL/TPOT/E2E |
| 8192 | 4 | 32 | 长 Prefill 与混合调度下的 E2E |

128/512 已用于功能与 fallback，不要求扩成性能矩阵。OSL=1 时 ITL/TPOT 无有效定义，填 N/A，不能报告为 0 ms 加速。

### 8.2 先补对应 workload 的命中 trace

**前提**：仍使用对应模式的 profile 服务；这一步的耗时不纳入性能结果。下面以 ISL8192/C4/OSL32 为例，其余矩阵行相应替换 N/C/O。

```bash
export N=8192 C=4 O=32
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions \
  --model hy4-test --tokenizer "$MODEL" --trust-remote-code \
  --dataset-name custom --dataset-path "$RUN_ROOT/inputs/bench_n${N}.jsonl" \
  --skip-chat-template --custom-output-len "$O" --num-prompts 8 --num-warmups 8 \
  --max-concurrency "$C" --request-rate inf --seed 7 --temperature 0 --ignore-eos \
  --profile --save-result --save-detailed --result-dir "$CASE_DIR" \
  --result-filename "profile_workload_n${N}_c${C}_o${O}.json" \
  2>&1 | tee "$CASE_DIR/profile_workload_n${N}_c${C}_o${O}.log"
```

**为什么**：`bench serve --profile` 在初始检查和 warmup 后调用已配置的 start/stop_profile。此时 trace 可能不带 `n..._a` 前缀，按新文件时间和 rank 逐一检查，不能用前面单请求计数直接验算。保存各类 kernel 的 grid.x 分布、分块/混合情况和 graph launch 关联。**PASS**：所有请求成功，8 rank 有证据，能够解释融合/回退；**FAIL/PARTIAL**：GPU trace 缺失或 ON 为零，先查实际 N/混合批次/导入路径，不把后续整体收益归因 Fusion。

### 8.3 新进程测量，不带 profiler

每个 `(模式, N, C, O, 重复编号)` 是一个新服务进程；客户端也是新进程。完成对应 profile 证据后关闭 profile 服务。至少 3 次独立重复，推荐 OFF/ON 交替顺序，不能同时跑两组。下面是一次完整测量示例；A/B/C/D 均按此执行，将 ARM 和 RUN_ID 显式替换。

```bash
# 终端 S：例，第一轮 Eager OFF / N8192 / C1 / O1。
ARM=eager_off RUN_ID=eager_off_n8192_c1_o1_r1 PURPOSE=perf \
  bash "$RUN_ROOT/start-server.sh"
```

```bash
# 终端 C：同一 CASE，待 health 正常后测量。
cd /results/manual
source env.sh
export N=8192 C=1 O=1 REP=1
export RUN_ID=eager_off_n8192_c1_o1_r1
export CASE_DIR="$RUN_ROOT/$RUN_ID"
curl --fail-with-body -sS "http://127.0.0.1:$PORT/health"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv \
  | tee "$CASE_DIR/gpu-before.csv"
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions \
  --model hy4-test --tokenizer "$MODEL" --trust-remote-code \
  --dataset-name custom --dataset-path "$RUN_ROOT/inputs/bench_n${N}.jsonl" \
  --skip-chat-template --custom-output-len "$O" --num-prompts 64 --num-warmups 16 \
  --max-concurrency "$C" --request-rate inf --seed 7 --temperature 0 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir "$CASE_DIR" --result-filename result.json \
  2>&1 | tee "$CASE_DIR/benchmark.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv \
  | tee "$CASE_DIR/gpu-after.csv"
.venv/bin/python - <<'PY' | tee "$CASE_DIR/result-check.txt"
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['CASE_DIR']) / 'result.json').read_text())
assert r['completed'] == 64 and r['failed'] == 0, (r['completed'], r['failed'])
assert not any(r['errors']), r['errors']
assert all(n == int(os.environ['O']) for n in r['output_lens']), r['output_lens']
print('actual input lengths', sorted(set(r['input_lens'])))
print('input tokens / benchmark second', r['total_input_tokens'] / r['duration'])
print('requests/s', r['request_throughput'])
print('output tokens/s', r['output_throughput'])
print('total tokens/s', r['total_token_throughput'])
PY
```

**参数说明**：custom 数据集、seed、无 chat template 固定输入及顺序；采样温度 0、忽略 EOS 固定输出长度；request-rate inf 是饱和 closed-loop 最大并发测试，不是生产到达率模拟。64 个计时请求，16 个 warmups 和初始 test request 不进入报告计时。若仍有首次编译/新 capture 发生于测量窗口，该次作废，增加相同预热后重启重测，OFF/ON 使用相同规则。

**PASS（测量有效性）**：64/64 成功、输入/输出长度及配置一致，无别的 GPU 任务、无 profiler、无首次 JIT/capture 落入计时；至少 3 个独立重复。**FAIL**：先查 JSON errors、服务日志和 GPU 干扰，不挑选最快一轮。每次完成后按 6.4 停服务，再手动开始下一模式/重复/工作负载。将 r1/r2/r3 各自的原始 result.json、日志和 GPU 状态完整保留。

### 8.4 指标与汇总规则

保存每次 TTFT、ITL、TPOT、E2EL 的 median/P90/P99；跨 3 次重复报告各次 median 的中位数和 min–max（重复数较少，不虚构置信区间）。吞吐同样报告中位数/min–max。E2EL 是单请求服务观测延迟，client_queue_time 是客户端并发限流等待，若需要用户排队全时长应另外保存两者之和，不混合口径。

- Prefill 吞吐代理 = `total_input_tokens / duration`，单位 input tok/s。OSL=1 时更接近 Prefill 主导服务吞吐，仍包含服务/调度/首 token 开销，**不是纯 Prefill Kernel 吞吐**。
- E2E 吞吐：`request_throughput` req/s、`output_throughput` output tok/s、`total_token_throughput` input+output tok/s。
- TTFT 包含实际服务等待及 Prefill；ITL 是连续输出间隔；TPOT 是首 token 之后每个输出 token 平均耗时。OSL32 才报告后两项。
- 延迟改善 `% = (OFF − ON) / OFF × 100`；吞吐改善 `% = (ON − OFF) / OFF × 100`；负值即退化。
- Eager 仅与 Eager 比，Graph 仅与 Graph 比。Graph capture/模型加载耗时单独记启动成本，不与 replay/E2E 稳态混算。

若 ON 明显更慢，先看 3 次离散范围、实际 N、图 padding、合法 fallback 和显存压力；只记录，不展开 Kernel/MoE 调优。没有该 workload 的真实命中证据时，结论只能写“配置 A/B 整体差异，融合贡献未确认”。所有性能仍附“严格数值验收未闭合，诊断性结果”。

## 9. Phase 4：数值回归记录，非阻塞诊断

### 9.1 固定输入 A/A 与 OFF/ON 的输出和 logprob 子向量

**前提**：在每个 profile 服务退出前执行，不在性能窗口中。现有服务 API 不直接返回全部 hidden states/raw logits；不修改 Runtime 插 hook。下面复用公开 `logprobs`/`logprob_token_ids`/`return_tokens_as_token_ids`，采集固定 16 个 vocab ID 的第一输出位置 log-probability。它是诊断子向量，**不是完整 logits，也不是现有 Golden 验收**。

```bash
export N=8192
for variant in a b; do
  for repeat in 1 2; do
    curl --fail-with-body -sS "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      --data-binary @"inputs/n${N}_${variant}_numerical.json" \
      -o "$CASE_DIR/n${N}_${variant}_numerical_${repeat}.json"
  done
done
```

按 N128/512/2048/8192 分别执行。**记录条件**：JSON 无错误，字段存在，usage 正确；缺失就记录 unavailable，不临时改变采样/阈值掩盖差异。a/a、b/b 用于衡量原路径和 ON 各自非确定性；a/b 看输入更新，但相同 argmax token 不能证明陈旧状态。API 对极小 logprobs 可能以 -9999 截断，命中这种值的指标要标明受截断影响。

四模式完成后计算：

```bash
cat > "$RUN_ROOT/compare-logprobs.py" <<'PY'
import json
import math
from pathlib import Path
import numpy as np

def vector(path):
    data = json.loads(path.read_text())
    choice = data['choices'][0]
    return choice['logprobs']['top_logprobs'][0], choice['logprobs']['tokens']

def compare(a, b, label):
    va, ta = vector(a)
    vb, tb = vector(b)
    # Request also returns the sampled token; compare the fixed requested IDs.
    request = json.loads(Path('inputs/n8192_a_numerical.json').read_text())
    keys = ['token_id:' + str(i) for i in request['logprob_token_ids']]
    assert all(k in va and k in vb for k in keys), (a, b, va.keys(), vb.keys())
    x = np.array([va[k] for k in keys], dtype=np.float64)
    y = np.array([vb[k] for k in keys], dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    print(label, 'max_abs', abs(y-x).max(),
          'relative_L2', np.linalg.norm(y-x) / nx if nx else math.nan,
          'cosine', x @ y / (nx*ny) if nx and ny else math.nan,
          'sampled_tokens', ta, tb, 'clipped', bool((x <= -9999).any() or (y <= -9999).any()))

arms = ('eager_off', 'eager_on', 'graph_off', 'graph_on')
for n in (128, 512, 2048, 8192):
    for variant in ('a', 'b'):
        def file(arm, rep):
            return Path(arm + '_profile') / f'n{n}_{variant}_numerical_{rep}.json'
        for arm in arms:
            compare(file(arm, 1), file(arm, 2), f'{n}/{variant}/{arm}/AA')
        for off, on in (('eager_off', 'eager_on'), ('graph_off', 'graph_on'),
                        ('eager_on', 'graph_on'), ('eager_off', 'graph_off')):
            compare(file(off, 1), file(on, 1), f'{n}/{variant}/{off}:{on}')
PY
.venv/bin/python compare-logprobs.py | tee numerical-summary.txt
```

**判定**：本子向量没有既定严格阈值，只记录指标，不新增容差或写数值 PASS。OFF A/A 已不稳定时不能把所有 A/B 差异归因 Fusion；ON 特有异常、NaN/Inf、明显跨请求复用旧响应仍应单列回归候选并保留输入。missing keys/接口不可用先核对导入版本与协议，不把缺数据当作对齐成功。完整 hidden states/raw logits 未采集就填“未采集”，不以此子向量替代。

### 9.2 复用已有严格验收证据

已有 `tests/models/test_hyv4_ihc_fusion.py` 和冻结 `lab/ihc_fusion` Golden，不改动、不提高容差：Residual exact；gates rtol=2e-5、atol=2e-6、relative L2≤2e-6；BF16 输出使用既有 cancellation budget；三层链 `rtol=2*BF16 eps, atol=2e-5`。历史 Runtime 74 passed / 13 strict failures、冻结 Lab 108 passed；详细分布与输入见既有报告。

首轮完整模型测试不额外启动合成 Kernel suite。若后续明确需要重跑已有用例、且镜像已经具备测试依赖，可在所有模型退出后人工执行既有命令（不安装依赖）：

```bash
cd /results/manual
source env.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  "$HY4_SOURCE/tests/models/test_hyv4_ihc_fusion.py" -v -p no:cacheprovider \
  --basetemp "$RUN_ROOT/pytest-tmp" 2>&1 | tee "$RUN_ROOT/existing-runtime-tests.log"
```

这不是 TP8 完整模型数值验收。记录逐项 PASS/FAIL、输入 shape 和历史是否已失败；缺依赖就标记未执行，不联网安装。原模型 A/A 失败和旧 Kernel BF16 舍入问题均不在本轮修复范围。即使 H200 服务运行和融合命中全通过，严格数值栏仍为 **PARTIAL**，不能据此给出可合入结论。

## 10. 一页最小执行清单

以下是首次执行顺序索引；先照相应节建立变量/脚本，不能在未准备环境时仅复制启动行。每一步检查 PASS 才继续。

| 顺序/位置 | 操作与原因 |
|---|---|
| 1 / H100 开发机 | 执行 3.1 Git 检查、bundle、SHA256、传输；固定完整分支，无需 GitHub 或权重传输 |
| 2 / H200 宿主机 | 填 3.2 的 6 个现场变量，检查 8 卡/进程/镜像/权重，校验并 clone 新目录；确保隔离 |
| 3 / H200 宿主机 | 运行 3.2 `docker run`，只读挂载源码与模型，结果独立挂载 |
| 4 / 容器 | 执行 4.1–4.3；uv 仅建环境入口，确认 source/extension 路径、SM90、完整权重与配置 |
| 5 / 容器 | 执行 5.1/5.2；生成固定输入、env.sh 和唯一启动脚本 |
| 6 / 容器终端 S | `ARM=eager_off RUN_ID=eager_off_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"` |
| 7 / 容器终端 C | `cd /results/manual; source env.sh`；设对应 RUN_ID/CASE_DIR，执行 6.2/6.3 和 9.1；先 N8192，再 512/128/2048；保存 8 rank 证据 |
| 8 / 容器 | 6.4 正常退出；用 `ARM=eager_on RUN_ID=eager_on_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"` 重复步骤 7 |
| 9 / 容器 | 退出后 `ARM=graph_off RUN_ID=graph_off_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"`；重复请求，增加第 7 节 replay 归属检查 |
| 10 / 容器 | 退出后 `ARM=graph_on RUN_ID=graph_on_profile PURPOSE=profile bash "$RUN_ROOT/start-server.sh"`；同样检查，不能用服务成功替代 fused graph 证据 |
| 11 / 容器 | 每个 profile 服务退出前用 8.2 对待测 workload 留 trace；四模式完成后运行 ` .venv/bin/python compare-logprobs.py`（先创建 9.1 脚本） |
| 12 / 容器 | 8.3 新开 perf 服务，先 N8192/C1/O1，Eager OFF/ON、Graph OFF/ON 各 3 次新进程；再按矩阵完成其余行；不带 profiler |
| 13 / 容器/宿主机 | 填下表，核对原始结果与数值状态，保存 GPU 进程状态，退出本次容器；不提交或推送代码 |

## 11. 结果填写模板

```text
日期/操作者：
结论：Ready for diagnostic testing / Conditional / Blocked
HEAD / baseline / source hash：
镜像 tag + digest / 容器 ID：
完整模型路径 / 层数 L / 权重格式 / config hash / 完整性依据：
GPU 0..7 UUID、型号、SM、显存、互联、其他任务：
Driver / Torch CUDA / Python / Torch / Triton / vLLM distribution：
vllm Python / Runtime / Kernel / 扩展 / FlashMLA 实际路径：
TP=8 PP=1 / runner / dtype / quantization / memory utilization：
max_model_len / token budget / max_seqs / capture_sizes / prefix cache：
ISL 请求值与实际值 / Tensor N 分布 / chunked prefill / 混合批次：
输入文件 SHA256 / 并发 / OSL / 采样 / 预热数 / 重复数：

模式           全量加载  8rank请求  capture/replay  fused每rank  Pre/Post计数  数值状态
Eager OFF       ___       ___       N/A            ___          ___          ___
Eager ON        ___       ___       N/A            ___          ___          ___
Graph OFF       ___       ___       ___            ___          ___          ___
Graph ON        ___       ___       ___            ___          ___          ___
Graph fused 节点关联证据（8 rank trace 文件/时间区间）：
N128 fallback / N512边界 / N2048 / N8192：
新增错误/更新输入/重复 replay/潜在陈旧状态：

每个 workload 和模式分别填：
重复 r1/r2/r3 目录：
加载/JIT/capture启动成本（非稳态计时）：
TTFT median/P90/P99，各重复 median 的中位数与范围：
ITL / TPOT（OSL1填N/A）：
E2EL median/P90/P99 / client queue：
input tok/s（服务吞吐代理）/ output tok/s / total tok/s / req/s：
OFF→ON 延迟改善% / 吞吐改善% / 离散范围：
对应 workload 融合命中证据；收益能否归因融合：
数值：OFF A/A、ON A/A、OFF/ON、Graph/Eager 的诊断指标：
严格 Golden 验收：未闭合 / 已有失败 / 新失败（不能留空）：
hidden states/raw logits：未采集 或 实际保存路径：
结论限定：诊断性性能，非正式正确性通过/非可合入结论。
未完成项与最小后续动作：
```

## 12. 常见失败与优先排查

| 现象 | 优先排查/记录；不在本文自动修复 |
|---|---|
| 配置 ON 但 fused=0 | 实际源码导入路径、两个 JSON bool、模型 enable_ihc/D/C、SM90、BF16/FP32、contiguous、物理 N、batch invariant/speculative/ubatch、Eager 混合批次、Graph 是否 breakable |
| Graph 返回文本但没有 replay | 检查是否误用 enforce-eager、PIECEWISE 是否退化、capture sizes 是否覆盖、实际 V2 runner、日志与 cudaGraphLaunch；不能写 Graph PASS |
| 有 fused 和 graph events 但无法关联 | 只证明两者均运行；保存 GPU node/correlation 信息，Graph fused 归属仍 PARTIAL |
| N128 反而融合 | 检查是否 pad 到 512 或更大；128 bucket 是否真正生效 |
| 只有部分 rank trace | 不接受只看 rank0；检查 start/stop_profile、worker 错误、目录共享和 TP rank 文件名 |
| `.so`/undefined symbol/缺 FlashMLA 接口 | 源码覆盖了 wheel 搜索路径或镜像 ABI 不匹配；记录路径和版本，换匹配环境，不重装整个环境碰碰运气 |
| 服务启动 OOM | 全量权重格式、量化识别、KV cache 与 capture 内存分别记录；OFF/ON 同配；缩小范围必须注明未测 shape |
| NCCL 卡住/worker 死亡 | 8 卡可见性、共享内存、互联、既有任务、各 rank 原始错误；不改通信/调度代码 |
| 性能抖动或收益消失 | GPU 干扰、频率/温度、首次 JIT、prefix cache、数据不一致、混合/分块 N、Graph padding；保存全部重复，不筛选最快结果 |
| 原路径 A/A 数值失败 | 对照已有报告，记录基线非确定性；不放宽阈值，不把文本相同当数值通过 |
| ON 特有 NaN/IMA/输出污染 | 停止对应测试，保存最小输入、shape、进程配置和 rank 日志；不切 Eager 冒充 Graph 通过，不在本轮修改 Kernel/Runtime |

现场尚需确认：H200 SSH/传输目录、源码新目录、完整权重路径/格式/层数、镜像 digest、Docker GPU 选择和挂载、可用显存/其他进程、uv 与镜像解释器、扩展 ABI/第三方接口、端口、上下文上限、capture 内存、真实业务长度与并发分布。这些都不能由 H100 三层测试推定。

本手册不涉及 MoE/HiSparse 调优、Decode Kernel 开发、其他 HC、非 SM90、公共 Graph 重构或历史数值修复。源码与模型只读，日志/trace/JSON 全部独立保存；无 Commit/Push/PR 步骤。
