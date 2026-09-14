# Command-Line Configuration Override System

IL-Studio supports flexible configuration management through command-line argument overrides that can modify parameters at any nesting depth in configuration files.

## Supported Configuration Types

The system supports override for the following 6 configuration types:
- `task` - Task configuration
- `policy` - Policy configuration
- `training` - Training configuration
- `robot` - Robot configuration
- `teleop` - Teleoperation configuration
- `env` - Environment configuration

## Override Syntax

### Basic Syntax

Two syntax formats are supported:

**Space-separated format:**
```bash
--<config_type>.<parameter_path> <value>
```

**Equals format:**
```bash
--<config_type>.<parameter_path>=<value>
```

### Supported Nesting Depth

The system supports arbitrary nesting depth for parameters. Examples:

- **2-level nesting**: `--policy.camera_names`
- **3-level nesting**: `--policy.model_args.backbone`
- **4-level nesting**: `--training.optimizer.lr_scheduler.type`
- **5-level nesting**: `--task.env.simulation.physics.timestep`
- **Deeper nesting**: `--config.a.b.c.d.e.f.value`

## Override Examples

### Policy Configuration Overrides

```bash
# Basic parameter override
python train.py --policy act --policy.camera_names '["primary", "wrist"]'

# Nested parameter override
python train.py --policy act --policy.model_args.backbone resnet50
python train.py --policy act --policy.model_args.hidden_dim 1024
python train.py --policy act --policy.model_args.enc_layers 6

# Deep nesting override
python train.py --policy act --policy.model_args.optimizer.lr 0.001
python train.py --policy act --policy.model_args.optimizer.lr_scheduler.type cosine
```

### Training Configuration Overrides

```bash
# Training parameter override
python train.py --training.learning_rate 0.0001
python train.py --training.num_train_epochs 10

# Optimizer configuration override
python train.py --training.optimizer.weight_decay 0.01
python train.py --training.optimizer.lr_scheduler.warmup_steps 1000
python train.py --training.optimizer.lr_scheduler.type linear
```

### Task Configuration Overrides

```bash
# Basic task parameters
python train.py --task.action_dim 14
python train.py --task.state_dim 14

# Environment parameter override
python train.py --task.env.simulation.physics.timestep 0.01
python train.py --task.env.simulation.physics.gravity -9.81
python train.py --task.env.simulation.rendering.width 1920
python train.py --task.env.simulation.rendering.height 1080
```

### Combined Usage

```bash
python train.py \
  --policy act \
  --task sim_transfer_cube_scripted \
  --training default \
  --policy.model_args.backbone resnet50 \
  --policy.model_args.hidden_dim 1024 \
  --policy.camera_names '["primary", "wrist", "overhead"]' \
  --training.learning_rate 0.0001 \
  --training.optimizer.lr_scheduler.type cosine \
  --task.env.simulation.physics.timestep 0.01
```

## Data Type Handling

The system automatically attempts to convert string values to appropriate data types:

- **Integer**: `"100"` → `100`
- **Float**: `"0.001"` → `0.001`
- **Boolean**: `"true"` → `True`, `"false"` → `False`
- **List**: `'["a", "b", "c"]'` → `["a", "b", "c"]`
- **Dictionary**: `'{"key": "value"}'` → `{"key": "value"}`
- **String**: Values that cannot be converted remain as strings

The type conversion is handled by the `_convert_to_type` utility function, which intelligently infers types based on value format.

## Override Priority

Parameter override priority (from highest to lowest):

1. **Command-line overrides** (highest priority)
2. **YAML configuration file parameters**
3. **Default values** (lowest priority)

This means command-line arguments always override the same parameters in configuration files.

## Practical Use Cases

### 1. Quick Experimentation with Different Model Architectures

```bash
# Try different backbones
python train.py --policy act --policy.model_args.backbone resnet34
python train.py --policy act --policy.model_args.backbone resnet50

# Adjust network depth
python train.py --policy act --policy.model_args.enc_layers 6
python train.py --policy act --policy.model_args.dec_layers 8
```

### 2. Debugging and Testing

```bash
# Quickly reduce training steps for testing
python train.py --training.max_steps 100

# Adjust batch size
python train.py --training.per_device_train_batch_size 4

# Modify camera configuration
python train.py --policy.camera_names '["primary"]'
```

### 3. Hyperparameter Search

```bash
# Learning rate search
python train.py --training.learning_rate 0.001
python train.py --training.learning_rate 0.0001
python train.py --training.learning_rate 0.00001

# Network size search
python train.py --policy.model_args.hidden_dim 256
python train.py --policy.model_args.hidden_dim 512
python train.py --policy.model_args.hidden_dim 1024
```

## Important Notes

1. **Quoting Complex Values**: For values containing spaces, special characters, or complex structures, use quotes:
   ```bash
   --policy.camera_names '["primary", "wrist"]'
   --policy.description "Multi-camera policy"
   ```

2. **Boolean Values**: Use `true`/`false` strings:
   ```bash
   --training.do_eval true
   --policy.model_args.use_pretrained false
   ```

3. **Path Parameters**: Ensure path correctness:
   ```bash
   --training.output_dir "/path/to/output"
   --policy.model_args.pretrained_path "/path/to/checkpoint"
   ```

4. **Parameter Validation**: The system validates parameter validity. Invalid parameters will be reported or ignored.

## Error Handling

- If an invalid configuration type is specified, the system will ignore that override
- If a parameter path doesn't exist, the system will create the necessary nested structure
- If data type conversion fails, the system will keep the original string value
- Errors and warnings are displayed at the start of training

## Implementation Details

The override system works by:

1. **Parsing**: The `parse_overrides()` function extracts override arguments from unknown command-line arguments
2. **Type Conversion**: The `_convert_to_type()` function converts string values to appropriate Python types
3. **Application**: The `apply_overrides_to_mapping()` or `apply_overrides_to_object()` functions apply overrides to configuration objects using nested path resolution

This powerful override mechanism allows you to flexibly adjust any parameter without modifying configuration files, greatly improving experimentation efficiency!
