# MLP Policy for IL-Studio

Simple Multi-Layer Perceptron (MLP) policy for imitation learning tasks.

## Features

- **Pure State Input**: Default uses only state vectors as input
- **Multi-modal Support**: Optional camera image input support
- **Chunked Output**: Supports outputting multi-step action sequences (chunk_size > 1)
- **Flexible Configuration**: Configurable layers, dimensions, activation functions, etc.
- **Framework Compatible**: Fully compatible with IL-Studio framework interface

## Architecture

### Input
- **State Mode**: `state_dim` dimensional state vector
- **Multi-modal Mode**: Concatenated vector of `state_dim + image_dim` dimensions

### Network Structure
```
Input Layer: Linear(input_dim, hidden_dim) + Activation
Hidden Layers: [Linear(hidden_dim, hidden_dim) + Activation] × (num_layers - 2)
Output Layer: Linear(hidden_dim, chunk_size * action_dim)
Output Reshape: reshape to (batch_size, chunk_size, action_dim)
```

### Output
- Shape: `(batch_size, chunk_size, action_dim)`
- Supports multi-step action prediction

## Configuration Files

### Basic Configuration (`configs/policy/mlp.yaml`)
```yaml
name: mlp
module_path: policy.mlp
config_class: MLPPolicyConfig
model_class: MLPPolicy
data_processor: get_data_processor
data_collator: get_data_collator
trainer_class: null
pretrained_config:
  model_name_or_path: null
  is_pretrained: false
model_args:
  state_dim: 14          # State dimension
  action_dim: 14         # Action dimension
  num_layers: 3          # Number of network layers
  hidden_dim: 256        # Hidden layer dimension
  activation: "relu"     # Activation function
  dropout: 0.1           # Dropout rate
  use_camera: false      # Whether to use camera
  chunk_size: 1          # Action sequence length
  learning_rate: 1e-3    # Learning rate
```

### Multi-modal Configuration (`configs/policy/mlp_camera.yaml`)
```yaml
name: mlp_camera
module_path: policy.mlp
model_args:
  use_camera: true       # Enable camera input
  num_layers: 4          # More layers for image processing
  hidden_dim: 512        # Larger hidden dimension
```

## Usage

### Training
```bash
python train.py --policy mlp --config configs/policy/mlp.yaml
```

### Evaluation
```bash
python eval.py --policy mlp --model_name_or_path /path/to/checkpoint
```

### Multi-modal Training
```bash
python train.py --policy mlp --config configs/policy/mlp_camera.yaml
```

## Framework Interface

Following IL-Studio's policy rules, MLP policy provides three necessary interfaces:

### 1. `load_model(args)`
- Load original model or trained checkpoint
- Returns dictionary containing `model` key

### 2. `get_data_processor(args, model_components)`
- Returns sample-level data processor
- Processes state vectors and optional image data
- Ensures data format meets model requirements

### 3. `get_data_collator(args, model_components)`
- Returns batch processing function
- Organizes multiple samples into batches
- Handles tensorization of different data types
- **Automatically ignores text modalities** (raw_lang, instruction, task, etc.)

### Model Inference Interface

### `select_action(obs)`
- Used for action selection during evaluation and inference
- Input: Observation dictionary containing `state` (and optional `image`)
- Output: Action sequence in numpy array format

## Data Processing Strategy

### Supported Modalities
- ✅ **State Modality**: `state` - Robot state vector
- ✅ **Image Modality**: `image` - Camera image (optional)
- ✅ **Action Modality**: `action` - Target action sequence

### Ignored Modalities
- ❌ **Text Modality**: Automatically ignores all text-related fields
  - `raw_lang`, `lang`, `language`, `text`
  - `instruction`, `task_description`
  - `task`, `episode_id`, `trajectory_id`
  - `dataset_name`, etc.

### Processing Flow
1. **Data Processor**: Extracts relevant modalities from input samples, ignores text
2. **Batch Processor**: Only tensorizes `state`, `action`, `image`
3. **Model Inference**: Uses only state and image information for action prediction

## Design Principles

1. **Simplicity**: MLP is the simplest neural network architecture, suitable for rapid prototyping and benchmarking
2. **Modularity**: Supports optional image modality for easy extension
3. **Framework Compatibility**: Strictly follows IL-Studio's interface specifications
4. **Flexible Configuration**: Easy parameter adjustment through YAML configuration files

## Applicable Scenarios

- **Simple State Space**: Tasks with low-dimensional state observations
- **Rapid Prototyping**: Baseline model for new tasks
- **Benchmarking**: Simple baseline for comparison with complex models
- **Debug Verification**: Verify correctness of data processing pipeline

## Notes

1. MLP does not process sequence information, each timestep predicts independently
2. Image input will be flattened, losing spatial structure information
3. Suitable for relatively simple control tasks
4. For complex vision-motor tasks, recommend using CNN or Transformer policies