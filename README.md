# VLAStudio

用于机器人 policy 训练、评估与部署的可扩展 Python 工具包。使用自己的 policy、dataset、robot、device、action manager 和 config，无需修改 VLAStudio 源码。

推荐使用 Python 脚本组织实验，同时保留 `vlastudio train` 和原来的 `python train.py`。不同 policy 的依赖由独立、可复用的 Python 环境管理。

## 源码布局

```text
vlastudio/
├── src/vlastudio/
│   ├── policy/          # 策略实现与 Trainer
│   ├── benchmark/       # 仿真与评估环境
│   ├── data_utils/      # 数据集、处理与缓存
│   ├── deploy/          # robot、device、action manager 与通信
│   ├── configs/         # 内置配置与别名
│   ├── utils/
│   ├── api.py           # 公开 Python API
│   ├── cli.py
│   └── entrypoints/     # 训练、评估、采集入口实现
├── examples/           # 可直接运行的扁平示例
├── docs/
├── pyproject.toml
└── train.py             # 兼容旧命令的薄入口
```

源码与安装包使用同一布局，例如 `vlastudio.policy`、`vlastudio.benchmark`。旧配置里的 `policy.act` 等引用仍可解析；新内置配置使用完整包路径。源码开发先 `pip install -e .`，普通安装用 `pip install .`。

> 当前版本为 `0.2.0.dev0`，尚未发布到 PyPI。请从本分支或 wheel 安装；当前不应直接用 `pip install vlastudio` 获取此开发版本。

## 安装

需要 Python 3.10 或更新版本：

```bash
git clone --branch codex/package-runtime-isolation https://github.com/WwZzz/VLAStudio.git
cd VLAStudio
python -m pip install -e .
# 或安装构建好的 wheel
python -m pip install /path/to/vlastudio-0.2.0.dev0-py3-none-any.whl
```

缓存默认位于 `~/.cache/vlastudio`。需要放到其他磁盘时只设置一个环境变量：

```bash
export VLASTUDIO_CACHE=/path/to/persistent-cache
```

基础安装只提供轻量入口。使用 `vlastudio` 命令时，启动器按配置创建环境、安装依赖并复用；使用 Python API 时则直接使用当前 Python 环境，不自动安装。GPU 驱动、系统库、仿真资源和硬件 SDK 的系统部分需要主机或容器支持。

`PyYAML`、`platformdirs`、`filelock` 和 `loguru` 属于 VLAStudio 公共运行时，基础包和每个托管环境都会安装。policy、dataset、仿真与 device 的 profile 只声明各自增加的依赖；使用完整 lock 文件时，环境准备阶段会先校验公共运行时，缺失时不会启动任务。

| Policy 实现 | 初始运行环境 | 平台 |
| --- | --- | --- |
| ACT / MLP | Python 3.10，Torch 2.4 | Windows / Linux x86-64 有锁文件 |
| OpenPI | Python 3.11，Torch 2.7.1 | Linux，glibc ≥ 2.31 |
| OpenVLA | Python 3.10，Torch 2.4 | Linux |
| 其他 / 自定义 policy | 配置声明 `runtime` | 由依赖决定 |

`--policy` 仍接收配置名称或 YAML 路径；例如 OpenPI 的内置配置使用 `pi0`，不是将 `--policy` 改成固定模型枚举。

## 用 Python 组织训练和评估

Python 脚本需要先安装其使用的组件。例如 ACT 训练和 ALOHA 仿真需要：

```bash
python -m pip install -e ".[act,aloha]"
```

如果希望完全自动管理环境，请使用 CLI：

```bash
vlastudio train -p act -t sim_transfer_cube_scripted -c default -o checkpoints/act_aloha
vlastudio eval-sim -m checkpoints/act_aloha -e aloha_transfer -o results/act_aloha
```

内置数据集和 policy 可以直接用别名，不需要自己写 YAML：

```python
import vlastudio as vla

dataset = vla.load_dataset("sim_transfer_cube_scripted")
dataset = vla.load_dataset("rlbench.reach_target", cache_dir="/datasets/vla-cache")
policy = vla.load_policy("act")
# 无参数时分别默认 sim_transfer_cube_scripted 和 act
dataset = vla.load_dataset()
policy = vla.load_policy()
# 同时支持用户自己的配置
dataset = vla.load_dataset("/my/configs/task.yaml", cache_dir="/datasets/custom-cache")
policy = vla.load_policy("/my/configs/policy.yaml")
```

点分别名 `rlbench.reach_target` 映射到包内 `configs/task/rlbench/reach_target.yaml`；其他内置名称遵循相同规则。别名只是配置入口，原始数据仍遵循配置中的路径或远程数据集 ID；`sim_transfer_cube_scripted` 在缓存为空时会下载内置配置指定的公开数据集。

`load_dataset(..., cache_dir=...)` 只控制数据缓存，优先于训练中的 `data_cache_dir` 和全局默认值，不移动原始数据，也不改变依赖环境和 checkpoint 目录。目录下 `huggingface/` 用于 HF Datasets 缓存，`lerobot/` 用于 LeRobot 默认下载位置，`normalize/` 用于统计量，`tasks/` 用于已启用的预处理缓存。显式数据源 `root` 仍由数据集配置决定，自定义数据集应遵守这些环境设置。指定缓存路径不会自动启用预处理缓存，启用仍需 task 的 `cache` 设置。

不指定时，沿用运行时默认数据缓存 `<cache_dir>/data`；现有 HF Datasets / LeRobot 环境变量和 task 显式 `cache.root` 保留。显式数据缓存参数或 `VLASTUDIO_DATA_CACHE_DIR` 会覆盖这些缓存位置。等价 CLI 参数是 `--data-cache-dir`。`load_policy(..., cache_dir=...)` 控制依赖环境的缓存根目录，模型下载缓存使用 `model_cache_dir`。

```python
import vlastudio as vla

dataset = vla.load_dataset("/my/configs/task.yaml")
policy = vla.load_policy("/my/configs/policy.yaml", cache_dir="/scratch/vlastudio")

result = vla.train(
    policy, dataset, "/my/configs/training.yaml",
    output_dir="/my/checkpoints/run1",
    overrides={"training.max_steps": 1000},
)
print(result.checkpoint)
print(policy.checkpoint)  # 成功训练后自动更新

bench = vla.load_env(
    "/my/configs/env.yaml",
    runtime_manifest="/my/configs/simulation-runtime.yaml",
)
evaluation = bench.evaluate(policy, output_dir="/my/results/run1", num_rollout=10)
print(evaluation.metrics)
```

以上路径需要替换为实际配置。无需数据下载的可运行案例：

```bash
python examples/train_mlp.py --output-dir ./checkpoints/toy
# 已有可用 NVIDIA GPU 时
python examples/train_mlp.py --output-dir ./checkpoints/toy-gpu --gpu
```

此例使用自定义合成数据集和小型 MLP，执行真实的两步优化并保存权重。首次仍需安装依赖。脚本按自身位置定位示例配置，也支持从其他目录使用脚本绝对路径运行。

通用训练脚本接收自己的配置：

```bash
python examples/train_policy.py \
  --policy /my/configs/policy.yaml \
  --task /my/configs/task.yaml \
  --training-config /my/configs/training.yaml \
  --output-dir /my/checkpoints/run1
```

加上 `--env /my/env.yaml --eval-runtime /my/simulation-runtime.yaml` 可接续评估。所有案例都位于 [examples](examples) 下。

ACT 在 ALOHA 上训练并接续仿真评估的最小案例只有五行公开 API：

```bash
python examples/_01_train_and_eval_act_on_aloha.py
```

脚本直接使用内置的 `sim_transfer_cube_scripted`、`act`、`default` 和
`aloha_transfer` 配置。修改这五行即可替换数据集、policy、训练配置、保存路径或评估环境。

Policy 与仿真器依赖冲突时，使用 [远程推理案例](examples/02_remote_inference/README.md)：
Policy 进程调用 `vla.serve(...)`，独立的仿真进程通过 TCP、HTTP(S) 或 SHM 地址评估。

### 对象与返回值

`load_policy`、`load_dataset`、`load_env` 返回轻量配置句柄：`vla.Policy`、`vla.Dataset`、`vla.Environment`。真实对象在当前 Python 环境的工作进程中创建；缺少依赖时会直接报错，不会修改当前环境。

`policy` 不是 `torch.nn.Module`，不能直接调用 `.parameters()`；`dataset` 也不是可迭代的 PyTorch Dataset。`load_dataset` 接收现有 **task 配置**，包含数据集列表、维度和归一化信息，而不只是数据文件路径。此接口用于 Python 实验编排。

`vla.train` 同步等待完成，复用原 processor、collator、数据缓存和 policy 专用 Trainer。成功返回 `TrainingResult(checkpoint, policy)` 并更新 `policy.checkpoint`；失败抛出 `vla.TaskError`，保留原 checkpoint，日志直接显示在终端。

需要直接操作张量、优化器或编写训练循环时，使用后文的自定义 `runtime.entrypoint`，在隔离进程中导入自己的 Python 模块，或使用原有源码环境。当前不提供逐个张量操作的跨进程代理。

## 配置、覆盖参数与 checkpoint

| 配置 | 内容 | 文档 |
| --- | --- | --- |
| policy | 实现模块、架构、初始化权重、运行环境 | [configs](src/vlastudio/configs/README.md) |
| task | 数据集、参数、维度、归一化与缓存 | [data_utils](src/vlastudio/data_utils/README.md) |
| training | batch size、步数、学习率、保存策略 | [training](src/vlastudio/configs/training/README.md) |
| env | 仿真任务、相机、控制参数 | [benchmark](src/vlastudio/benchmark) |
| action manager | 动作分块、同步与执行策略 | [action manager](src/vlastudio/configs/action_manager/README.md) |

`load_*` 支持现有配置名称、YAML 路径和 `@config/name`。配置路径在调用时解析；配置内部的数据或 Python 文件相对路径仍遵循原有调用目录语义。可复用脚本建议用绝对路径。设置 `VLASTUDIO_CONFIG_PATH` 可以添加配置搜索根目录。

Python 的 `overrides` 对应 CLI dotted overrides：

```python
vla.train(policy, dataset, "default", output_dir="/my/checkpoints/run1",
          overrides={"training.max_steps": 2000,
                     "training.per_device_train_batch_size": 8,
                     "policy.args.chunk_size": 16})
```

值使用字符串、数字或布尔值；列表和嵌套结构放入 YAML。`output_dir` 通过函数参数设置。续训沿用原 Trainer 的 `resume_from_checkpoint` 规则，在 training 配置中设置，并保留原输出目录及 `checkpoint-*` 子目录；原入口在输出目录没有 checkpoint 子目录时会清除续训设置。

已有 checkpoint 可以直接绑定后评估：

```python
policy = vla.load_policy("act", checkpoint="/my/checkpoints/run1")
```

`checkpoint=` 用于评估，不自动变成训练初始化权重。训练初始化仍使用 policy 的 `pretrained_weight_path` 等已有字段。

## 评估环境

`bench.evaluate(policy, ...)` 复用 `eval_sim.py`。支持 `num_rollout`、`batch_size`、`device`、`action_manager` 和 env overrides；默认 `batch_size=0` 顺序执行。真实机器人继续使用 `vlastudio eval-real` 的部署流程。

仿真依赖不一定包含在训练环境中，managed 评估需要**同时包含 policy 和仿真器依赖的完整 runtime manifest**。根据对应 benchmark 文档准备并锁定环境，不能只把训练锁文件当成完整仿真环境。当前 Python 已装好全部依赖时也可以：

```python
bench = vla.load_env("aloha_transfer", runtime="current")
evaluation = bench.evaluate(policy, output_dir="./results/new-run", device="cuda")
```

如果两套依赖无法共存，用 `vla.connect_policy("host:port")` 连接单独的
`vla.serve(...)` 进程。此时仿真进程只解析远程客户端和 env 的运行环境。

输出目录须为空或不存在，避免混入旧指标。返回值的 `output_dir` 是绝对路径，`metrics` 是按相对 JSON 文件名组织的字典；视频保留在输出目录。环境的相机、动作空间、归一化须与训练配置匹配。

## 缓存与保存路径

```python
policy = vla.load_policy("act", cache_dir="/scratch/vlastudio",
                         model_cache_dir="/datasets/huggingface-cache")
```

也可通过 API 的 `cache_dir=...` 显式覆盖，或设置 `VLASTUDIO_CACHE`。缓存根目录优先级：显式参数、`VLASTUDIO_CACHE`、兼容变量 `VLASTUDIO_CACHE_DIR`、旧 `ILSTD_CACHE`、用户 settings、`~/.cache/vlastudio`。

| 位置 | 内容 |
| --- | --- |
| `cache/envs` | 隔离 Python 环境与依赖锁 |
| `cache/apps` | 按代码内容区分的应用快照 |
| `cache/data` | 默认数据缓存，task 显式路径保留 |
| `cache/models` | 默认模型资源 |
| `cache/uv`、`cache/python` | 下载与解释器缓存 |
| `output_dir` | 自己指定的 checkpoint，不属于环境缓存 |

`model_cache_dir` 设置 Hugging Face 的 `HF_HOME`。已有 `TORCH_HOME`、`OPENPI_DATA_HOME`、`UV_CACHE_DIR` 等设置会被尊重。缓存建议放容量足够的持久化盘，`/tmp` 可能随实例重建消失。完整规则见 [运行环境文档](docs/package_runtime.md)。

```bash
vlastudio prepare --policy act --cache-dir /scratch/vlastudio
vlastudio doctor --cache-dir /scratch/vlastudio
```

Python 调用支持 `offline=True`，必须已有对应环境与模型资源。该选项设置 uv / Hugging Face 离线标志，不能约束自定义代码的任意网络请求。

## 自定义 policy、dataset、robot、device 与其他组件

所有组件继续遵循现有接口契约，可来自安装包或自己的文件，无需修改仓库。例如 task 中的条目：

```yaml
datasets:
  - name: custom
    type: /my/project/dataset.py:MyDataset
    args:
      root: /datasets/my-episodes
# task 的 meta 字段仍需按数据维度与归一化要求设置
```

自定义 policy 配置示意：

```yaml
name: my_policy
type: my_package.policy
args:
  pretrained_weight_path: /models/initial-weights
runtime:
  python: "3.11"
  requirements:
    - my-policy-plugin==1.0.0
```

示意包名需要替换为真实可安装依赖。policy 模块保留模型加载、processor、collator、Trainer hooks；dataset 保留样本契约；robot / device / action manager 遵循对应基类。参考 `examples/components.py`、`examples/policy.yaml` 与 [接口说明](docs/package_runtime.md#external-components)。

支持 `module.Class`、`module:Class`、`/absolute/file.py:Class`，以及包 entry points 注册的 `@policy/name`、`@dataset/name` 等。`vla.register()` 仅作用于当前进程；跨环境使用模块 / 文件引用或安装包的 entry points，并将包加入 runtime requirements。

当前进程已准备依赖时，可以直接构造真实组件：

```python
device = vla.create({"type": "my_package.camera:Camera", "args": {"name": "wrist"}},
                    kind="device")
```

`create` 不自动安装环境。独立 device 使用 `vlastudio device --config /my/device.yaml`，device 配置也可声明 runtime。

### 自定义 Python 训练循环

可以写完整 runtime manifest，通过 `runtime_manifest=` 或 `--runtime-manifest` 选择：

```yaml
python: "3.11"
requirements:
  - my-training-package==1.0.0
entrypoint: /my/project/training.py:run
```

`run(command, argv)` 在隔离环境中运行。成功返回 `0`，失败抛异常或返回非零值。函数可创建真实 dataset 和 policy，自行控制优化器。通过 `vla.train` 使用时，要在 `-o` 指定目录保存可评估的产物。标准评估需要标准 policy metadata 与权重；自定义评估入口可定义自己的 checkpoint 格式并输出指标 JSON。

同一训练进程内 dataset 与 policy 依赖仍须兼容；隔离环境无法让一个进程同时加载两套互不兼容的 Torch。跨实验可使用不同环境，在线不兼容组件可通过已有 policy server / client 拆分。

## 保留的 CLI 与源码方式

```bash
vlastudio train -p act -t /my/task.yaml -c /my/training.yaml -o /my/checkpoints/run1
vlastudio train --policy /my/policy.yaml --policy.args.chunk_size 16
vlastudio serve -m /my/checkpoints/run1
vlastudio eval-sim -m /my/checkpoints/run1 -e /my/env.yaml --runtime-manifest /my/simulation-runtime.yaml
# 已配置依赖的源码环境
python train.py -p act -t /my/task.yaml -c /my/training.yaml -o /my/checkpoints/run1
vlastudio train -p act --runtime current
```

原演示和源码使用说明保留在 [README.legacy.md](README.legacy.md)。其中的旧环境安装说明是历史参考；旧完整依赖在 `requirements-legacy.txt`，当前根目录的 `uv sync` 仅安装轻量包依赖。

## 排错

- 缺少运行环境声明：添加 `runtime` 或完整 manifest，或使用已配置好的 `runtime="current"`。
- 首次安装慢：查看下载日志、缓存盘容量和网络；重复相同环境会复用。
- Windows 路径过长：使用短缓存路径，例如 `C:/vla-cache`。
- 外部模块找不到：将包加入 runtime requirements，或使用绝对文件引用 / `plugin_path`。
- 仿真依赖缺失：按 benchmark 文档补全评估环境。
- 需要直接修改张量与优化器：使用自定义训练入口或源码环境，而非轻量配置句柄。
