# 10. Custom Benchmarks

A benchmark defines a standard task or environment for training and evaluation. This guide explains how to create a new one for simulated environments.

## What is a Benchmark?

In IL-Studio, a benchmark is a Python package inside the `benchmark/` directory that contains:
1.  **An Environment Wrapper**: A class that wraps a simulation environment (e.g., from `gym`, `dm_control`, `robosuite`) and standardizes its interface.
2.  **A `create_env` function**: A factory function in your package's `__init__.py` that instantiates the environment.
3.  **A Configuration File**: A YAML file in `configs/env/` that specifies the environment's parameters.

## Step 1: Create the Environment Wrapper

Create a new directory `benchmark/my_benchmark` and add a file `benchmark/my_benchmark/env.py`.

```python
# In benchmark/my_benchmark/env.py
import gym

class MyBenchmarkEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ... initialize your simulation ...
        # Define action and observation space
        self.action_space = ...
        self.observation_space = ...

    def reset(self):
        # ... reset the environment and return the first observation ...
        return obs

    def step(self, action):
        # ... apply action, step simulation, return obs, reward, done, info ...
        return obs, reward, done, {}, info # The third return val is terminated, the fourth is truncated
```

## Step 2: Create the Factory Function

In `benchmark/my_benchmark/__init__.py`, create the `create_env` function. The `vlastudio evaluate` script will call this function to create instances of your environment.

```python
# In benchmark/my_benchmark/__init__.py
from .env import MyBenchmarkEnv

def create_env(config):
    """Factory function for MyBenchmarkEnv."""
    return MyBenchmarkEnv(config)
```

## Step 3: Create the Configuration File

Create `configs/env/my_benchmark.yaml`. This file links the environment name to your code and defines its parameters.

```yaml
type: "my_benchmark"  # MUST match the directory name in `benchmark/`
task: "reach_the_target"
max_timesteps: 500
# ... other environment-specific parameters to be passed in the config object
```

## Step 4: Use the Benchmark

You can now use your new benchmark in evaluation scripts by referencing its configuration name. The script will look for a directory in `benchmark/` that matches the `type` field in the YAML file.

```bash
vlastudio evaluate --runtime current -m <model_path> -e my_benchmark
```
