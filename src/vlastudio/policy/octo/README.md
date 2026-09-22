# Installation
```shell
cd policy/octo/octo_pytorch
cp ../pyproject.toml pyproject.toml
uv venv
source .venv/bin/activate
uv sync # if uv was not found, use 'pip install uv' to install it
uv pip install -e .
cd ../../..
```

# Example
### Train
```shell
uv run vlastudio train --runtime current -p octo_aloha -t sim_transfer_cube_scripted -c debug -o ckpt/octo_debug
```
### Eval
