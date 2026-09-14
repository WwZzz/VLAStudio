# Remote inference with separate environments

这个例子把 policy 推理和仿真评测放进两个进程、两个 Python 环境。服务端只安装
policy 依赖，客户端只安装远程评测和仿真依赖，因此两侧可以使用互不兼容的
PyTorch、Transformers、MuJoCo 或其他运行库。

从仓库根目录执行以下命令。命令直接调用各环境的 Python，不需要反复切换
`activate`。

## 1. Policy 服务端

创建 Policy 环境：

```bash
python -m venv .venv-policy-act
.venv-policy-act/bin/python -m pip install -e ".[act]"
```

第一个终端启动服务：

```bash
.venv-policy-act/bin/python examples/02_remote_inference/serve.py
```

`serve.py` 从 `checkpoints/act_aloha` 加载 checkpoint，并在所有网卡的 TCP 5000
端口监听。服务会持续运行，按 `Ctrl+C` 停止。

## 2. 仿真评测端

创建独立的仿真环境：

```bash
python -m venv .venv-env-aloha
.venv-env-aloha/bin/python -m pip install -e ".[remote-eval,aloha]"
```

保持服务端运行，在第二个终端执行：

```bash
.venv-env-aloha/bin/python examples/02_remote_inference/evaluate.py
```

同一台机器使用 `127.0.0.1:5000`。跨机器运行时，将 `evaluate.py` 中的地址改为
Policy 服务器可访问的 IP，例如：

```python
policy = vla.connect_policy("192.168.1.20:5000")
```

也可以直接把地址传给 `evaluate`：

```python
bench = vla.load_env("aloha_transfer")
bench.evaluate("192.168.1.20:5000", output_dir="results/remote")
```

## 支持的通信形式

| 形式 | 服务端地址 | 评测端地址 | 适用范围 |
|---|---|---|---|
| TCP | `0.0.0.0:5000` | `host:5000` | 默认，支持跨机器 |
| HTTP | `http://0.0.0.0:8000` | `http://host:8000` | JSON/HTTP，支持跨机器 |
| HTTPS | `https://0.0.0.0:8443` | `https://host:8443` | 需要证书配置 |
| SHM | `shm://act_policy` | `shm://act_policy` | 仅限同一台机器 |

HTTP/HTTPS 服务端需要额外安装：

```bash
.venv-policy-act/bin/python -m pip install -e ".[act,serve-http]"
```

然后同时修改两个脚本中的地址。HTTPS 证书路径使用仓库已有的
`ILSTD_SSL_KEYFILE` 和 `ILSTD_SSL_CERTFILE` 环境变量。SHM 不经过网络，两个进程
必须能访问同一套共享内存。

TCP 传输使用 Python pickle，适合可信网络或本机隔离环境。跨不可信网络时应使用
HTTPS 或在可信隧道中运行。
