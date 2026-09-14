# RoboFlamingo Policy for ILStudio

RoboFlamingo is a pre-trained VLM-based robotics learning framework that learns a wide variety of language-conditioned robot skills by fine-tuning on offline free-form imitation datasets.

## Overview

This module integrates [RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo) into ILStudio, providing:

- OpenFlamingo backbone (CLIP ViT + LLM with cross-attention)
- Perceiver Resampler for vision token compression
- Multiple action decoder types (LSTM, FC)
- Multi-view image support (RGB + Gripper camera)
- Language-conditioned policy learning

## Installation

```bash
# Navigate to the roboflamingo directory
cd policy/roboflamingo

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# For development
uv sync --all-extras
```

### Additional Requirements

1. **OpenFlamingo**: For full functionality with cross-attention layers:
```bash
pip install open-flamingo
```

2. **CALVIN Benchmark** (optional): For CALVIN evaluation:
```bash
pip install pytorch3d pyrender
```

## Pre-trained Models

OpenFlamingo checkpoints are **automatically downloaded** from [Hugging Face](https://huggingface.co/openflamingo) when you specify `openflamingo_model` in the configuration.

### Available Models

| Model Name | Alias | Language Model | Vision Encoder | Parameters |
|------------|-------|---------------|----------------|------------|
| `openflamingo-3b-instruct` | `3b` | mpt-1b-dolly | CLIP ViT-L/14 | ~3B |
| `openflamingo-3b` | `3b-base` | mpt-1b-redpajama | CLIP ViT-L/14 | ~3B |
| `openflamingo-4b-instruct` | `4b` | RedPajama-3B-Instruct | CLIP ViT-L/14 | ~4B |
| `openflamingo-4b` | `4b-base` | RedPajama-3B-Base | CLIP ViT-L/14 | ~4B |
| `openflamingo-9b` | `9b` | mpt-7b | CLIP ViT-L/14 | ~9B |

### List Available Models (Python)

```python
from policy.roboflamingo import list_available_models
print(list_available_models())
```

## Configuration

### Basic Configuration with Auto-Download (Recommended)

```yaml
type: policy.roboflamingo
name: roboflamingo
args:
  # Auto-download OpenFlamingo model (recommended)
  openflamingo_model: 3b  # Options: 3b, 3b-base, 4b, 4b-base, 9b
  
  # Optional: Custom cache directory
  # cache_dir: /path/to/cache
  
  # Policy Configuration
  action_dim: 6           # Action dimension (excluding gripper)
  state_dim: 7            # State dimension
  window_size: 12         # Number of frames in context window
  
  # Decoder Configuration
  decoder_type: lstm      # lstm or fc
  decoder_hidden_size: 1024
  decoder_num_layers: 4
  
  # Multi-view Settings
  use_gripper: true       # Use gripper camera
  fusion_mode: post       # post, pre, or two_way
  sep_resampler: false    # Separate resampler for gripper
  
  # Training Settings
  freeze_vision: true     # Freeze CLIP vision encoder
  freeze_embed: false     # Freeze LLM embeddings
  use_state: false        # Use robot state input
  
  # Precision
  fp16: true
  bf16: false
```

### Advanced Configuration (Manual Paths)

If you prefer to specify paths manually:

```yaml
type: policy.roboflamingo
name: roboflamingo
args:
  # Manual LLM Configuration
  llm_name: mpt_dolly_3b  # Preset: mpt_3b, mpt_dolly_3b, mpt_4b, mpt_9b
  # Or specify paths directly:
  # lang_encoder_path: path/to/llm
  # tokenizer_path: path/to/tokenizer
  # cross_attn_every_n_layers: 1
  
  # Vision Encoder
  clip_vision_encoder_path: ViT-L-14
  clip_vision_encoder_pretrained: openai
  
  # Manual checkpoint path
  openflamingo_checkpoint: path/to/checkpoint.pt
  
  # ... other settings
```

## Usage

### Training

```bash
# Train on CALVIN dataset
python train.py \
    --policy_config configs/policy/roboflamingo.yaml \
    --task_config configs/task/calvin.yaml \
    --output_dir outputs/roboflamingo
```

### Inference

```python
from policy.roboflamingo import load_model

# Load model
args = SimpleNamespace(
    model_name_or_path="path/to/checkpoint",
    is_training=False,
    device="cuda"
)
components = load_model(args)
model = components['model']

# Run inference
obs = {
    'image': torch.randn(1, 2, 3, 224, 224),  # RGB + Gripper
    'state': torch.randn(1, 7),
    'raw_lang': "pick up the red block"
}
actions = model.select_action(obs)
```

## Architecture

```
RoboFlamingo
├── Vision Encoder (CLIP ViT-L/14, frozen)
│   ├── RGB Image → Patch Tokens
│   └── Gripper Image → Patch Tokens (optional)
├── Perceiver Resampler
│   ├── Compress vision tokens to fixed latents
│   └── Separate resampler for gripper (optional)
├── Language Model (MPT/LLaMA with cross-attention)
│   ├── Text tokenization
│   ├── Gated cross-attention layers
│   └── Language features
└── Action Decoder (LSTM/FC)
    ├── Arm actions (6D)
    └── Gripper action (1D)
```

## Key Features

1. **Multi-frame Context**: Uses sliding window of frames for temporal understanding
2. **Multi-view Fusion**: Supports RGB and gripper camera fusion
3. **Language Conditioning**: Natural language task instructions
4. **Pre-trained VLM**: Leverages OpenFlamingo's vision-language understanding

## Performance (CALVIN Benchmark)

| Method | Training Data | 1 | 2 | 3 | 4 | 5 | Avg Len |
|--------|--------------|---|---|---|---|---|---------|
| HULC | ABCD (Full) | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06 |
| RT-1 | ABCD (Lang) | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| **RoboFlamingo** | ABCD (Lang) | **0.964** | **0.896** | **0.824** | **0.740** | **0.66** | **4.09** |

## Citation

```bibtex
@article{li2023vision,
  title={Vision-Language Foundation Models as Effective Robot Imitators},
  author={Li, Xinghang and Liu, Minghuan and Zhang, Hanbo and Yu, Cunjun and Xu, Jie and Wu, Hongtao and Cheang, Chilam and Jing, Ya and Zhang, Weinan and Liu, Huaping and Li, Hang and Kong, Tao},
  journal={arXiv preprint arXiv:2311.01378},
  year={2023}
}
```

## License

MIT License - see the original [RoboFlamingo repository](https://github.com/RoboFlamingo/RoboFlamingo) for details.

