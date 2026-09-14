# Installation
```shell
cd policy/qwen25dp
uv sync
source .venv/bin/activate
cd ../..
```

# Example
```shell
python train.py -c debug -p qwen25dp -t sim_transfer_cube_scripted -o ckpt/qwen25_test
```