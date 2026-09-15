# 环境管理

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
vlastudio env run --policy act -- python examples/_01_train_and_eval_act_on_aloha.py
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
