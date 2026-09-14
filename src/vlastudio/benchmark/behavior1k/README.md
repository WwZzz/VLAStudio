# BEHAVIOR-1K Benchmark for ILStudio

## Quick Installation (Recommended)

Use `setup.sh` for one-click installation:

```shell
cd benchmark/behavior1k

git clone -b v3.7.2 https://github.com/StanfordVL/BEHAVIOR-1K.git
# Full installation (create environment + install all components + download dataset)
./setup.sh --all --accept-nvidia-eula --accept-dataset-tos

# Or install step by step
./setup.sh --new-env --bddl --omnigibson --joylo --dataset
```

### Installation Options

```shell
./setup.sh --help
```

| Option | Description |
|--------|-------------|
| `--new-env` | Create a new uv virtual environment |
| `--bddl` | Install BDDL (Behavior Domain Definition Language) |
| `--omnigibson` | Install OmniGibson + Isaac Sim (core physics simulator) |
| `--joylo` | Install JoyLo (teleoperation interface) |
| `--dataset` | Download BEHAVIOR dataset |
| `--primitives` | Install motion planning support |
| `--eval` | Install evaluation dependencies |
| `--dev` | Install development dependencies |
| `--cuda-version` | Specify CUDA version (default: 12.4) |
| `--accept-nvidia-eula` | Automatically accept NVIDIA EULA |
| `--accept-dataset-tos` | Automatically accept dataset terms of service |
| `--all` | Equivalent to `--new-env --bddl --omnigibson --joylo --dataset` |


```shell
cd /path/to/ILStudio
source benchmark/behavior1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=True

# Test environment setup (DummyTask, no task loading)
# Note: R1Pro robot requires 23-dimensional actions
python eval_sim.py -e behavior1k.dummy -m __dummy-23 --batch_size 0 -n 1
```

### Evaluating Specific Tasks

```shell
# Evaluate assembling_gift_baskets task
python eval_sim.py -e behavior1k.example -m /path/to/checkpoint --batch_size 0 -n 10
```

**Important Notes**:
- **R1Pro robot action space is 23-dimensional** (use `__dummy-23` for testing)
- Must set `--batch_size 0` because OmniGibson does not support multiprocessing
- Environment variables `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=True` must be set before running
- Environment configuration automatically uses official `eval.py` settings (R1Pro, assisted grasping, etc.)

### Configuration Files

Environment configuration files are located in `configs/env/behavior1k/`:

- `dummy.yaml` - DummyTask configuration for testing
- `example.yaml` - Example configuration with a specific task

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | str | Task name (e.g., `assembling_gift_baskets`) |
| `scene_model` | str | Scene model (e.g., `Rs_int`) |
| `robot_type` | str | Robot type (e.g., `R1`, `R1Pro`, `Fetch`) |
| `max_timesteps` | int | Maximum number of steps per episode |
| `ctrl_space` | str | Control space: `ee` (end-effector) or `joint` (joint) |
| `ctrl_type` | str | Control type: `delta` (incremental) or `abs` (absolute) |
| `image_size` | list | Image resolution `[height, width]` |
| `obs_modalities` | list | Observation modalities: `rgb`, `depth`, `proprio` |
| `headless` | bool | Whether to run in headless mode |
| `instance_id` | int | Task instance ID (for different initialization variants) |

### Custom Configuration

Create a new configuration file `configs/env/behavior1k/my_task.yaml`:

```yaml
- type: benchmark.behavior1k.Behavior1kEnv
  name: my_custom_task
  args:
    task: cleaning_up_the_kitchen_only
    scene_model: Rs_int
    robot_type: R1
    max_timesteps: 500
    ctrl_space: ee
    ctrl_type: delta
    image_size:
      - 256
      - 256
    obs_modalities:
      - rgb
      - proprio
    headless: true
```

### Evaluating with Trained Models

```shell
# Use local model
python eval_sim.py -e behavior1k/example -m /path/to/checkpoint -n 10

# Use remote policy server
python eval_sim.py -e behavior1k/example -m localhost:5000 -n 10
```

## Important Notes

1. **Python Version**: Must use Python 3.10
2. **CUDA Version**: Default is CUDA 12.4, can be adjusted via `--cuda-version`
3. **NVIDIA EULA**: Installing OmniGibson and Isaac Sim requires accepting NVIDIA license agreement
4. **Dataset License**: BEHAVIOR-1K dataset is for non-commercial academic research only
5. **Dependencies**:
   - `--omnigibson` requires `--bddl`
   - `--primitives` and `--eval` require `--omnigibson`
   - `--dataset` requires `--omnigibson`
6. **Isaac Sim Installation**: setup.sh automatically downloads and installs Isaac Sim wheel package

## Troubleshooting

### Segmentation Fault Error

If you encounter a Segmentation fault during startup, follow these steps:

#### 1. Install Required System Libraries (Most Common Cause)

Isaac Sim requires OpenGL-related libraries. Missing libraries will cause Segmentation fault:

```shell
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libglu1-mesa libgl1-mesa-glx libegl1-mesa libxrender1 libsm6

# Verify library exists
ldconfig -p | grep libGLU
```

#### 2. Set Required Environment Variables

```shell
export OMNI_KIT_ACCEPT_EULA=YES    # Accept NVIDIA EULA
export OMNIGIBSON_HEADLESS=True    # Enable headless mode
```

#### 3. Check GPU Driver Version

Isaac Sim 4.5.0 requires NVIDIA driver >= 535.129.03:
```shell
nvidia-smi
```

#### 4. Use Virtual Display on Systems Without Display Server

```shell
# Install Xvfb
apt-get install xvfb

# Run with virtual display
xvfb-run -a python eval_sim.py -e behavior1k/dummy ...
```

#### 5. Check CUDA Version Compatibility

```shell
nvcc --version
nvidia-smi
```

### Isaac Sim Download Failure

If Isaac Sim package download fails, try:

1. Check network connection (requires access to pypi.nvidia.com)
2. Manually download wheel package from https://pypi.nvidia.com
3. Ensure sufficient disk space (Isaac Sim requires approximately 10GB)

### Insufficient Memory

Isaac Sim requires substantial memory. Recommended:
- System RAM >= 32GB
- GPU VRAM >= 8GB
