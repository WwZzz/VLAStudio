# Florence2-OFT Policy

A lightweight VLA (Vision-Language-Action) implementation using **Florence2** encoder backbone with continuous action prediction via L1 regression.

## Installation

```bash
cd policy/florence2_oft
uv venv
source .venv/bin/activate
uv sync

```

## Features

- **Florence2 Encoder Backbone**: Uses Florence2's vision-language encoder (decoder removed for efficiency)
- **L1 Regression Action Head**: Predicts continuous actions via MLP ResNet
- **Memory Efficient**: Removes unused decoder and lm_head modules
- **LoRA Support**: Optional LoRA fine-tuning for efficient training

## Supported Models

| Model | Description |
|-------|-------------|
| `microsoft/Florence-2-base` | Base model (0.23B params) |
| `microsoft/Florence-2-large` | Large model (0.77B params) |

## Architecture

```
Input Images + Instruction
        ↓
    Florence2 Vision Encoder
        ↓
    Image-Text Merge
        ↓
    Language Encoder
        ↓
    Mean Pooling / Last Tokens
        ↓
    MLP ResNet Action Head (L1 Regression)
        ↓
    Predicted Actions [chunk_size, action_dim]
```

## Usage

### Config Example

```yaml
# configs/policy/florence2_oft.yaml
type: policy.florence2_oft
name: florence2_oft
args:
  vlm_model_name_or_path: microsoft/Florence-2-large
  action_dim: 7
  state_dim: 7
  chunk_size: 16
  use_pooled_output: true  # Use mean pooling for action prediction
```

### Python API

```python
from policy.florence2_oft import load_model, get_data_processor, get_data_collator

model_components = load_model(args)
model = model_components['model']

# Select action
action = model.select_action(batch_obs)
```

## Limitations

- **Single Image Only**: Florence2 currently only supports single image input per sample
- **Eager Attention**: Uses eager attention (no flash attention support)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_model_name_or_path` | `microsoft/Florence-2-large` | Path to Florence2 model |
| `action_dim` | 7 | Action space dimension |
| `state_dim` | 7 | State space dimension |
| `chunk_size` | 16 | Action prediction horizon |
| `use_pooled_output` | true | Use mean pooling for action queries |
| `action_head_num_blocks` | 2 | Number of MLP ResNet blocks |
| `action_head_hidden_mult` | 2 | Hidden dimension multiplier |

## License

MIT License
