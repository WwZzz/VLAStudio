# Installation

```bash
cd /path/to/ILStudio/policy/xvla
uv venv
source .venv/bin/activate
uv sync

cd ../../lerobot
uv pip install -e ".[xvla]"
cd ..
```

# Usage

The ILStudio entrypoint is `configs/policy/xvla.yaml`.

Typical fine-tuning command:

```bash
python train.py -p xvla -t lerobot/xvla-soft-fold -o ckpt/xvla_soft_fold
```

If you want to start from a different X-VLA checkpoint, override `pretrained_model_name_or_path`
in the policy yaml or on the CLI.

# Notes

- This wrapper reuses the XVLA implementation under `lerobot/src/lerobot/policies/xvla`.
- Images are converted to float, normalized with ImageNet statistics, and packed into
  ILStudio batches in `policy/xvla/data_utils.py`.
- The custom trainer keeps XVLA's differential learning-rate optimizer and cosine-decay
  scheduler behavior.
