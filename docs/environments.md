# 环境管理

base 默认存在：首次执行环境管理命令时，自动记录启动器所在的 Python 环境。
这一步不安装依赖；`env activate` 与 `env deactivate` 默认返回该环境。
已有 base 记录不会被自动覆盖，`VLASTUDIO_BASE_PYTHON` 可显式指定初始解释器。
`env install` 默认在当前环境安装并更新 base；`env create` 默认创建并准备
托管 base，已有用户登记的 base 则继续复用。

首次给当前终端加载 shell 支持：

```bash
eval "$(vlastudio env init --shell bash)"
# zsh: eval "$(vlastudio env init --shell zsh)"
```

PowerShell：

```powershell
vlastudio env init --shell powershell | Out-String | Invoke-Expression
```

初始化后可以创建并自动激活环境，也可以安装到当前环境：

```bash
vlastudio env create                   # 默认创建/复用 base；ACT 无需单独 create
vlastudio env create --policy smolvla
vlastudio env create --policy smolvla -n smol_work
vlastudio env create --env robotwin
vlastudio env install                  # 安装到当前环境，并登记为 base
vlastudio env install --policy smolvla -n work
vlastudio env list
vlastudio env activate work
vlastudio env activate                 # 默认切换到 base
vlastudio env deactivate               # 同样切换到 base
```

未初始化 shell 时，create 完成创建和安装，但不能切换父终端。
install 不创建 venv，使用增量安装；版本约束可能升级或降级现有包。

`-n` 与 `--name` 等价。显式命名的新环境使用独立目录
`$VLASTUDIO_CACHE/envs/named-<名称>`；两个名称不会因依赖相同而共用目录。
重复 create 会复用已登记环境，保留后续手动修改。旧记录仍沿用原路径。
不传名称时继续按依赖哈希复用缓存，默认仍为 base。

可以先激活环境再安装，也可以直接指定已登记的环境名称：

```bash
vlastudio env activate smol_work
vlastudio env install --policy smolvla
vlastudio env install -n act_work --policy act --env aloha_sim
```

install 指定已有 `-n` 时直接安装到该解释器，无需先激活。
指定尚未登记的名称时，将当前解释器登记为该名称，不创建新环境。
不指定 `-n` 时仍安装到当前环境并登记为 base。
`--policy` 和 `--env` 可同时用于 install，合并两份依赖声明；解析器会报告
版本冲突，不会静默覆盖约束。存在完整 lockfile 时需提供合并后的
`--runtime-manifest`。不能兼容的依赖仍应分开运行服务端和仿真端。
安装成功后登记到 `$VLASTUDIO_CACHE/environments.json`，保存环境名、解释器路径和来源。
list 也显示缓存中的旧环境，以及已删除解释器的 missing 状态。`-n` 指定环境名。
省略组件或环境名时，create、install、prepare、path、run、activate 均默认使用 base。
install 未指定 `-n` 时将当前解释器登记为 base，即使使用了 `--policy` 或 `--env`。
重新登记 base 会更新默认解释器，原记录保留为 `base-<路径哈希>`；其他同名环境附加路径哈希。
run、path 和 create/prepare 复用已登记的 base。deactivate 始终返回 base，不恢复之前的任意环境。
更新后请重新执行上述 shell 初始化命令，以启用 deactivate。
RoboTwin 和 BEHAVIOR 清单安装 Python 依赖；仿真资源、SDK 和系统驱动需另行准备。

安装 VLAStudio 入口后，按组件名称运行脚本，无需知道 extras 名称：

```bash
pip install -e .
vlastudio env run --policy act -- examples/01_train_and_eval_act_on_aloha/run.sh
vlastudio env run --policy smolvla -- python my_smolvla_script.py
vlastudio env run --policy pi0 -- python my_openpi_script.py
vlastudio env run --remote --env aloha_sim -- python examples/02_remote_inference/evaluate.py
```

ACT、MLP、DP 以及 ALOHA、MetaWorld、LIBERO 共用 base manifest。
SmolVLA 和 OpenPI 各自使用独立 manifest。`--policy` 接收已有 policy
配置名称或路径；环境命令额外接受 `dp`、`openpi` 别名。
`--env` 接收 env 配置名称或路径，`aloha_sim` 对应 `aloha_transfer`。

```bash
vlastudio env prepare --policy act
vlastudio env path --policy act
vlastudio env list
vlastudio env prepare --policy act --dry-run
```

`path` 只输出解释器路径，不触发安装。`prepare` 安装环境，`run` 安装后执行命令。
默认目录是 `~/.cache/vlastudio/envs/<依赖哈希>`，设置 `VLASTUDIO_CACHE`
修改缓存根目录。环境清单变化生成新目录，已有环境保持原样。
下载源继承进程环境和安装工具自身配置；VLAStudio 不指定镜像。

需要为本次创建或安装指定源时，使用 `-i`（等价于 `--index-url`）：

```bash
vlastudio env create --policy smolvla -i http://nexus.sii.shaipower.online/repository/pypi/simple/
vlastudio env install -i http://nexus.sii.shaipower.online/repository/pypi/simple/
```

`prepare` 和 `run` 首次准备环境时也支持这个参数。源地址只传给本次
依赖解析与安装，不写入全局配置，也不改变环境的依赖哈希。
命令中的地址应为纯 URL，不要包含 Markdown 的 `[地址](地址)` 格式。

已有准备好的 base 环境可以直接复用：

```bash
export VLASTUDIO_BASE_PYTHON=/absolute/path/to/.venv/bin/python
vlastudio env run --policy act -- python experiment.py
```

该解释器由用户维护，VLAStudio 不向其中安装或升级依赖。
复杂仿真所需 SDK、资源及依赖需由组件的 runtime 声明；没有完整声明时提供 manifest：

```bash
vlastudio env run --runtime-manifest /my/robotwin-runtime.yaml -- python evaluate.py
```

manifest 沿用现有 `python`、`requirements`、`lockfile` 和 `overlays` 字段。
不能共存的 policy 和仿真请分别启动服务端和客户端。
RoboTwin/BEHAVIOR 的仿真资源和平台 SDK 不会因创建 venv 自动出现。

base 清单的依赖解析与 GPU/仿真执行验证是两项独立检查；安装成功不代表模型下载、
场景资源和系统渲染驱动已经配置完成。
