# Installation

```shell
cd /path/to/ILStudio
cd policy/smolvla
uv sync
source .venv/bin/activate
cd ../../third_party/lerobot
uv pip install -e ".[smolvla]"
uv pip install numpy==1.26.4
cd ../..
```

# Examples
## Finetune on LIBERO-Object
```shell
python train.py -p smolvla_libero -t libero_object -c default -o ckpt/smolvla_libero

```

# TroubleShooting

- cannot import name 'AutoModelForImageTextToText' from 'transformers'. 