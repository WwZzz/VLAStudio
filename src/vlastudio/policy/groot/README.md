# GR00T Policy

NVIDIA's **GR00T-N1.5** foundation model for robotics, integrated into ILStudio.

## Installation
```shell
cd policy/groot
uv venv
uv pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" # --index-url https://download.pytorch.org/whl/cu1XX
uv pip install ninja "packaging>=24.2,<26.0" # flash attention dependencies
uv pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
cd ../../third_party/lerobot
uv pip install -e ".[groot]"
```


## Features

- **Eagle2 VLM Backbone**: Vision-language model for multimodal understanding
- **Flow Matching Diffusion**: State-of-the-art action prediction
- **Multi-Embodiment**: Support for different robot embodiments
- **Pretrained Weights**: Uses `nvidia/GR00T-N1.5-3B` from HuggingFace

## Installation

GR00T requires the lerobot third_party module. Ensure it's available:

```bash
# The third_party/lerobot should already be in the repository
cd policy/groot
```

## Supported Models

| Model | Description |
|-------|-------------|
| `nvidia/GR00T-N1.5-3B` | Main GR00T model (3B params) |

## Architecture

```
Input Images + Instruction
        ↓
    Eagle2 VLM Backbone
        ↓
    Vision-Language Features
        ↓
    Flow Matching Diffusion Head
        ↓
    Predicted Actions [chunk_size, action_dim]
```

## Usage

### Config Example

```yaml
# configs/policy/groot.yaml
type: policy.groot
name: groot
args:
  base_model_path: nvidia/GR00T-N1.5-3B
  action_dim: 7
  state_dim: 7
  chunk_size: 16
  embodiment_tag: new_embodiment
  tune_projector: true
  tune_diffusion_model: true
```

### Python API

```python
from policy.groot import load_model, get_data_processor, get_data_collator

model_components = load_model(args)
model = model_components['model']

# Select action
action = model.select_action(batch_obs)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_path` | `nvidia/GR00T-N1.5-3B` | Path to GR00T model |
| `action_dim` | 7 | Action space dimension |
| `state_dim` | 7 | State space dimension |
| `chunk_size` | 16 | Action prediction horizon (max 16) |
| `max_state_dim` | 64 | Maximum state dimension for padding |
| `max_action_dim` | 32 | Maximum action dimension for padding |
| `embodiment_tag` | `new_embodiment` | Robot embodiment identifier |
| `tune_llm` | false | Fine-tune LLM backbone |
| `tune_visual` | false | Fine-tune vision tower |
| `tune_projector` | true | Fine-tune projector |
| `tune_diffusion_model` | true | Fine-tune diffusion head |

## Notes

- GR00T has a maximum action horizon of 16
- State and action dimensions are padded to `max_state_dim` and `max_action_dim`
- Requires Flash Attention 2 for optimal performance

## License

Apache 2.0 (NVIDIA)

