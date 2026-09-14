# LLaVA-OFT Policy

A lightweight VLA (Vision-Language-Action) implementation using **LLaVA-OneVision** backbone with continuous action prediction via L1 regression on last token hidden states.

Reference: [UniAct](https://github.com/2toinf/UniAct) implementation.

## Installation

```bash
cd policy/llava_oft
uv venv
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation  # Optional, for acceleration
```

## Features

- **LLaVA-OneVision Backbone**: Uses LLaVA-OneVision as vision-language model
- **Last Token Extraction**: Uses last valid token hidden state for action prediction (UniAct style)
- **Flash Attention 2**: Optional flash attention support for acceleration
- **LoRA Support**: Optional LoRA fine-tuning for efficient training

## Supported Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | 0.5B | Lightweight model |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 7B | Full model |

## Architecture

```
Input Images + Instruction
        ↓
    LLaVA-OneVision VLM
        ↓
    Extract Last Token Hidden State
        ↓
    Expand to chunk_size
        ↓
    MLP ResNet Action Head (L1 Regression)
        ↓
    Predicted Actions [chunk_size, action_dim]
```

## Usage

### Config Example

```yaml
# configs/policy/llava_oft.yaml
type: policy.llava_oft
name: llava_oft
args:
  vlm_model_name_or_path: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  action_dim: 7
  state_dim: 7
  chunk_size: 16
  use_last_token: true  # Use last token for action prediction (UniAct style)
```

### Python API

```python
from policy.llava_oft import load_model, get_data_processor, get_data_collator

model_components = load_model(args)
model = model_components['model']

# Select action
action = model.select_action(batch_obs)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_model_name_or_path` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | Path to LLaVA model |
| `action_dim` | 7 | Action space dimension |
| `state_dim` | 7 | State space dimension |
| `chunk_size` | 16 | Action prediction horizon |
| `use_last_token` | true | Use last token for action queries |
| `action_head_num_blocks` | 2 | Number of MLP ResNet blocks |
| `action_head_hidden_mult` | 2 | Hidden dimension multiplier |

## License

MIT License

