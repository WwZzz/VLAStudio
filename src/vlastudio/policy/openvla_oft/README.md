# Installation

The openvla-oft environment is separate from the main ILStudio environment:

```bash
cd policy/openvla_oft/openvla-oft
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install -e .

# Install Flash Attention 2 for training (optional but recommended)
uv pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
uv pip install "flash-attn==2.5.5" --no-build-isolation
uv pip install loguru opencv-python tianshou==0.2.0 numpy==1.26.4
```

# Usage

### Training

```bash
# Activate the openvla-oft environment
cd policy/openvla_oft/openvla-oft
source .venv/bin/activate

# Train with ILStudio
cd ../../..  # Back to ILStudio root
python train.py -p openvla_oft -t your_task_config -o outputs/openvla_oft_experiment
```

### Using Pretrained Checkpoints

OpenVLA-OFT provides pretrained checkpoints for LIBERO tasks:

| Checkpoint | Task Suite |
|------------|------------|
| `moojink/openvla-7b-oft-finetuned-libero-spatial` | LIBERO-Spatial |
| `moojink/openvla-7b-oft-finetuned-libero-object` | LIBERO-Object |
| `moojink/openvla-7b-oft-finetuned-libero-goal` | LIBERO-Goal |
| `moojink/openvla-7b-oft-finetuned-libero-10` | LIBERO-10 |
| `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | All combined |

To use a pretrained checkpoint, modify the config:

```yaml
args:
  pretrained_checkpoint: moojink/openvla-7b-oft-finetuned-libero-spatial
```

**Note**: State/Action normalization is handled by ILStudio's unified normalization pipeline, not internally by the model.

### Evaluation

```bash
# Evaluate on simulation
python eval_sim.py -e libero_spatial -m path/to/checkpoint -o results/eval
```

## Configuration

### Policy Config (`configs/policy/openvla_oft.yaml`)

```yaml
type: policy.openvla_oft
name: openvla_oft

args:
  # Base model
  pretrained_checkpoint: openvla/openvla-7b
  
  # Action prediction (choose one)
  use_l1_regression: true     # Recommended
  use_diffusion: false
  
  # Architecture
  num_images_in_input: 2      # 1 or 2 cameras
  use_proprio: true           # Proprioceptive state
  use_film: false             # Language-conditioned vision
  
  # Action parameters
  action_dim: 7               # Action dimension
  state_dim: 8                # Proprio state dimension
  chunk_size: 8               # Actions per chunk
  
  # LoRA
  training_mode: lora
  lora_rank: 32
  lora_alpha: 16
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_l1_regression` | Use L1 regression action head | `true` |
| `use_diffusion` | Use diffusion action head | `false` |
| `num_images_in_input` | Number of camera inputs | `2` |
| `use_proprio` | Include proprioceptive state | `true` |
| `chunk_size` | Actions predicted per inference | `8` |
| `lora_rank` | LoRA rank for fine-tuning | `32` |
| `center_crop` | Apply center crop (for aug-trained models) | `true` |

## Architecture

```
OpenVLA-OFT
├── Vision Backbone (SigLIP)
│   └── Multi-image input support
├── Language Model (Llama-2 7B)
│   └── LoRA adapters
├── Proprio Projector (optional)
│   └── MLP: proprio_dim -> llm_dim
├── Action Head
│   ├── L1 Regression: MLP ResNet
│   └── Diffusion: DDIM noise prediction
└── Noisy Action Projector (diffusion only)
```

## Training Tips

1. **Learning Rate**: Use `5e-4` with 10x decay after 100k steps
2. **Batch Size**: 8 per GPU with ~62GB VRAM
3. **Image Augmentation**: Enable random crop during training, use center crop for eval
4. **LoRA Rank**: 32 works well for most tasks
5. **Action Chunks**: 8 actions per chunk balances performance and reactivity

## File Structure

```
policy/openvla_oft/
├── __init__.py          # Module interface (load_model, get_data_processor, etc.)
├── modeling.py          # OpenVLAOFTPolicy and OpenVLAOFTConfig
├── data_utils.py        # Data processors and collators
├── trainer.py           # Custom trainer
├── README.md            # This file
└── openvla-oft/         # Original openvla-oft repository
    ├── prismatic/       # Core model code
    ├── experiments/     # Evaluation scripts
    └── vla-scripts/     # Training scripts
```

## Citation

If you use OpenVLA-OFT, please cite:

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
