# Training Configuration System

This directory contains the training configuration system that organizes and manages training parameters for the IL-Studio project.

## Overview

The training configuration system intelligently separates parameters based on their usage patterns:

- **Command-line suitable parameters**: Core training parameters that change frequently between runs (e.g., `output_dir`, `learning_rate`, `max_steps`, `batch_size`) are kept in `HyperArguments` for easy command-line override
- **Stable process parameters**: Parameters that are typically set once and don't change between runs (e.g., `lora_enable`, `use_quantization`, `freeze_backbone`) are stored in YAML configuration files

This design provides the best of both worlds: easy command-line usage for frequently changed parameters and organized configuration files for stable settings.

## Files

- `default.yaml`: Default training configuration with stable process parameters
- `loader.py`: Training configuration loader and utilities
- `README.md`: This documentation file

## Usage

### Basic Usage

```python
from configs.training.loader import load_training_config, create_training_arguments

# Load training configuration
config = load_training_config('configs/training/default.yaml')

# Create TrainingArguments (requires HyperArguments)
from train import HyperArguments
hyper_args = HyperArguments()
training_args = create_training_arguments('configs/training/default.yaml', hyper_args)
```

### In Training Script

The training script (`train.py`) automatically loads the training configuration:

```bash
python train.py --task_name sim_transfer_cube_scripted --training_config configs/training/default.yaml
```

### Command Line Overrides

You can override core training parameters via command line:

```bash
python train.py \
    --task_name sim_transfer_cube_scripted \
    --training_config configs/training/default.yaml \
    --output_dir my_output \
    --learning_rate 0.001 \
    --max_steps 1000 \
    --per_device_train_batch_size 16
```

## Configuration Structure

The training configuration is organized into two main categories:

### Command-line Suitable Parameters (in HyperArguments)
These parameters change frequently between runs and are kept in `HyperArguments` for easy command-line override:

- `output_dir`: Output directory for checkpoints
- `num_train_epochs`: Number of training epochs
- `max_steps`: Maximum number of training steps
- `per_device_train_batch_size`: Training batch size per device
- `per_device_eval_batch_size`: Evaluation batch size per device
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `warmup_steps`: Number of warmup steps
- `warmup_ratio`: Warmup ratio
- `lr_scheduler_type`: Learning rate scheduler type
- `optim`: Optimizer type
- `adam_beta1`, `adam_beta2`, `adam_epsilon`: Adam optimizer parameters
- `logging_dir`: Logging directory
- `logging_strategy`: Logging strategy
- `logging_steps`: Logging frequency
- `report_to`: Reporting backend
- `save_strategy`: Saving strategy
- `save_steps`: Saving frequency
- `save_total_limit`: Maximum number of checkpoints to keep
- `save_latest_checkpoint_only`: Manual opt-in that saves the newest checkpoint before deleting every older `checkpoint-*` directory. Defaults to `false`, forces `save_total_limit=1`, and cannot be combined with `load_best_model_at_end`.
- `resume_from_checkpoint`: Whether to resume from checkpoint
- `dataloader_num_workers`: Number of data loader workers
- `dataloader_pin_memory`: Whether to pin memory
- `remove_unused_columns`: Whether to remove unused columns
- `do_eval`: Whether to do evaluation
- `eval_steps`: Evaluation frequency
- `seed`: Random seed

### Stable Process Parameters (in YAML config)
These parameters are typically set once and don't change between runs:

#### Data Processing
- `preload_data`: Whether to preload data
- `lazy_preprocess`: Whether to use lazy preprocessing
- `episode_first`: Whether to sample episodes first
- `use_reasoning`: Whether to use reasoning data
- `use_prev_subtask`: Whether to use previous subtask
- `select_seg_token_mask`: Whether to select segment token mask
- `is_multimodal`: Whether to use multimodal data
- `image_aspect_ratio`: Image aspect ratio
- `skip_mirrored_data`: Whether to skip mirrored data
- `history_images_length`: Length of history images

#### Model Architecture
- `model_name_or_path`: Model name or path
- `is_pretrained`: Whether model is pretrained
- `using_ema`: Whether to use exponential moving average
- `flash_attn`: Whether to use flash attention
- `freeze_vision_tower`: Whether to freeze vision tower
- `freeze_backbone`: Whether to freeze backbone
- `tune_mm_mlp_adapter`: Whether to tune multimodal MLP adapter
- `llm_loss_weight`: Language model loss weight
- `load_pretrain`: Whether to load pretrained model

#### LoRA Settings
- `lora_enable`: Whether to enable LoRA
- `lora_module`: LoRA modules
- `lora_task_type`: LoRA task type
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha
- `lora_dropout`: LoRA dropout
- `lora_weight_path`: LoRA weight path
- `lora_bias`: LoRA bias
- `lora_lr`: LoRA learning rate

#### Quantization Settings
- `use_quantization`: Whether to use quantization
- `bits`: Number of bits for quantization
- `double_quant`: Whether to use double quantization
- `quant_type`: Quantization type

#### Other Settings
- `cache_dir`: Cache directory

## Creating Custom Configurations

You can create custom training configurations by copying `default.yaml` and modifying the parameters:

```yaml
# configs/training/my_config.yaml
output_dir: "ckpt/my_training"
learning_rate: 0.0005
max_steps: 2000
per_device_train_batch_size: 16
# ... other parameters
```

Then use it in training:

```bash
python train.py --training_config configs/training/my_config.yaml
```

## Benefits

1. **Intelligent Organization**: Parameters are organized based on usage patterns - command-line suitable vs. stable settings
2. **Easy Command-line Usage**: Core training parameters can be easily overridden via command line
3. **Stable Configuration**: Process parameters are organized in YAML files for easy management
4. **Reusability**: Easy to create and share different training configurations
5. **Type Safety**: Parameters are validated and converted to appropriate types
6. **Documentation**: Clear documentation of all available parameters
7. **Maintainability**: Easy to add, remove, or modify training parameters
8. **Backward Compatibility**: All existing functionality is preserved

## Migration from Old System

The old system had all training parameters defined in the `HyperArguments` dataclass. The new system:

1. **Keeps command-line suitable parameters** in `HyperArguments` for easy override
2. **Moves stable process parameters** to YAML configuration files
3. **Loads training parameters** from config files and applies them to `HyperArguments`
4. **Creates `transformers.TrainingArguments`** from the combined parameters
5. **Supports command-line overrides** for all core training parameters

This design provides the best of both worlds: easy command-line usage for frequently changed parameters and organized configuration files for stable settings.
