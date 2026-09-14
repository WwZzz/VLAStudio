# Qwen-OFT Policy

A lightweight VLA (Vision-Language-Action) implementation supporting both **Qwen2.5-VL** and **Qwen3-VL** backbones with continuous action prediction via L1 regression on action token hidden states.

Inspired by [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

## Installation

```bash
cd policy/qwen_oft
uv venv
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation
```

## Supported Models

| Qwen Version | Example Models |
|--------------|----------------|
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` |

The model version is **auto-detected** from the model path, or can be explicitly set via `qwen_version` in config.

## Usage

### Config Example

```yaml
# configs/policy/qwen_oft.yaml
type: policy.qwen_oft
name: qwen_oft
args:
  # Use Qwen2.5-VL
  vlm_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  # Or use Qwen3-VL
  # vlm_model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
  
  # Optional: explicitly set version ('qwen2.5' or 'qwen3')
  qwen_version: null  # auto-detect
  
  action_dim: 7
  state_dim: 7
  chunk_size: 16
```

### Python API

```python
from policy.qwen_oft import load_model, get_data_processor, get_data_collator

# The model will auto-detect Qwen version from path
model_components = load_model(args)
print(f"Using Qwen version: {model_components['qwen_version']}")
```
