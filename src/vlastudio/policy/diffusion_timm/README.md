# DiffusionTimmPolicy

A flexible Diffusion Policy implementation using [timm](https://github.com/huggingface/pytorch-image-models) pretrained vision encoders with configurable diffusion backbone (1D U-Net or Transformer) for action generation.

## Features

- **Flexible Vision Backbone**: Support for any timm model (ResNet, ViT, ConvNeXt, etc.)
- **Dual Diffusion Backbone**: Choose between 1D U-Net or Transformer decoder
- **Spatial Softmax**: Keypoint-based feature aggregation for better spatial understanding
- **Configurable Diffusion**: DDPM/DDIM schedulers with various noise schedules
- **EMA Support**: Exponential moving average for stable training
- **Multi-Camera Support**: Handle multiple camera inputs with optional weight sharing

## Installation

```bash
pip install timm diffusers einops
```

## Configuration

### Basic Config (`configs/policy/diffusion_timm.yaml`)

```yaml
type: policy.diffusion_timm
name: diffusion_timm

args:
  action_dim: 7
  state_dim: 7
  chunk_size: 16  # action horizon
  
  camera_names:
    - primary
  
  # Vision encoder
  vision_model_name: resnet18
  vision_pretrained: true
  vision_frozen: false
  vision_feature_dim: 64
  feature_aggregation: spatial_softmax
  num_kp: 32
  
  # Backbone type: 'unet' or 'transformer'
  backbone_type: unet
  
  # Diffusion
  num_inference_steps: 10
  num_train_timesteps: 100
    
action_normalize: minmax
state_normalize: minmax
```

### Backbone Options

#### UNet (default)

```yaml
backbone_type: unet
diffusion_step_embed_dim: 256
down_dims: [256, 512, 1024]
kernel_size: 5
n_groups: 8
cond_predict_scale: true
```

#### Transformer

```yaml
backbone_type: transformer
n_layer: 8       # number of transformer layers
n_head: 8        # number of attention heads
n_emb: 256       # embedding dimension
p_drop_attn: 0.1 # attention dropout
```

### Backbone Comparison

| Aspect | UNet | Transformer |
|--------|------|-------------|
| Inductive Bias | Temporal locality | Global attention |
| Parameters | ~77M (default) | ~12M (default) |
| Training Speed | Faster | Slower per step |
| Long Horizons | May struggle | Better |
| Short Horizons | Excellent | Good |

### Vision Encoder Options

| Model | `vision_model_name` | Notes |
|-------|---------------------|-------|
| ResNet-18 | `resnet18` | Fast, lightweight |
| ResNet-34 | `resnet34` | Better capacity |
| ResNet-50 | `resnet50` | Strong baseline |
| ViT-Base | `vit_base_patch16_224` | Requires square input |
| DINOv2 | `vit_base_patch14_dinov2.lvd142m` | Strong pretrained features |
| ConvNeXt-Tiny | `convnext_tiny` | Modern CNN |

### Feature Aggregation Options

- `spatial_softmax`: Extract keypoints using spatial softmax (recommended for CNNs)
- `avg`: Global average pooling (recommended for ViT)
- `cls`: Use CLS token (for ViT only)
- `identity`: No pooling (flatten all features)

## Usage

### Training

```bash
# UNet backbone (default)
python train.py \
    --task_config configs/task/your_task.yaml \
    --policy_config configs/policy/diffusion_timm.yaml \
    --training_config configs/training/dp.yaml

# Transformer backbone
python train.py \
    --task_config configs/task/your_task.yaml \
    --policy_config configs/policy/diffusion_timm.yaml \
    --training_config configs/training/dp.yaml \
    --policy.args.backbone_type transformer
```

### Inference

```python
from policy.diffusion_timm import load_model

# Load trained model
args.is_training = False
args.model_name_or_path = "path/to/checkpoint"
args.model_args = {'using_ema': True, 'num_inference_steps': 10}

model_components = load_model(args)
model = model_components['model']

# Inference
batch_obs = {
    'image': images,  # (B, N, C, H, W)
    'qpos': states,   # (B, D)
}
actions = model.select_action(batch_obs)  # (B, T, A)
```

## Architecture

### UNet Backbone

```
Input:
  - image: (B, N, C, H, W) - N camera images
  - state: (B, D) - robot state/qpos

TimmObsEncoder:
  - Per-camera backbone (ResNet/ViT/etc.)
  - Spatial Softmax pooling
  - Linear projection to feature_dim
  - Concatenate all camera features + state

ConditionalUnet1D:
  - Sinusoidal timestep embedding
  - Conditional residual blocks with FiLM
  - U-Net structure with skip connections

Output:
  - action: (B, T, A) - action sequence
```

### Transformer Backbone

```
Input:
  - image: (B, N, C, H, W) - N camera images
  - state: (B, D) - robot state/qpos

TimmObsEncoder:
  - Per-camera backbone (ResNet/ViT/etc.)
  - Feature pooling (spatial softmax / avg)
  - Concatenate all camera features + state

TransformerForActionDiffusion:
  - Input embedding + positional encoding
  - Condition tokens (obs features + timestep)
  - Transformer decoder layers
  - Output projection

Output:
  - action: (B, T, A) - action sequence
```

## Differences from Original DiffusionPolicy

1. **No internal normalization**: ILStudio handles normalization externally
2. **Simplified interface**: Uses `chunk_size` instead of `action_horizon`
3. **timm integration**: More flexible vision backbone selection
4. **Dual backbone support**: Both UNet and Transformer architectures
5. **PreTrainedModel compatible**: Easy save/load with HuggingFace format

## References

- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [Diffusion Policy Transformer](https://arxiv.org/abs/2303.04137)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [diffusers library](https://github.com/huggingface/diffusers)
