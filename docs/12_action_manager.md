# 12. Action Manager

The **Action Manager** is a critical component for successful real-world robot deployment in `vlastudio infer`. It solves the fundamental mismatch between the *inference rate* of the policy and the *control rate* of the robot.

## The Problem: Rate Mismatch

*   **Policy Inference**: A typical vision-based policy is computationally expensive. It might run at a low frequency, like **5-10 Hz**, and it often produces a whole "chunk" of future actions (e.g., 16-64 steps) at once.
*   **Robot Control**: Most robot hardware requires a smooth, continuous stream of commands at a high frequency, like **50-100 Hz**, to avoid jerky movements.

Directly sending the policy's chunky, low-frequency actions to the robot would result in poor and unsafe performance.

## The Solution: A Smart Buffer

The Action Manager, located in `deploy/action_manager/`, acts as a thread-safe, smart buffer and interpolator between the inference thread and the main robot control loop.

```
+--------------------------+
|    Inference Thread      |
| (Low Frequency, ~10 Hz)  |
+------------+-------------+
             |
             | Produces a chunk of 16 future actions
             | e.g., [a_t, a_t+1, ..., a_t+15]
             v
+------------+-------------+
|      Action Manager      |
| (Thread-safe Queue/Buffer)|
+------------+-------------+
             ^
             |
             | Queries for an action at the current timestamp
             | e.g., "Give me the action for t+0.02s"
             |
+------------+-------------+
| Main Robot Control Loop  |
| (High Frequency, ~50 Hz) |
+--------------------------+
```

1.  The **Inference Thread** calls `action_manager.put(action_chunk, timestamp)`.
2.  The **Main Control Loop** calls `action_manager.get(current_time)` at a much higher rate.
3.  The Action Manager's job is to look at its buffer of future action chunks and intelligently return the best possible action for the requested `current_time`. This might involve interpolation between two waypoints in the action chunk.

## Available Managers

IL-Studio provides several Action Manager strategies. Each manager is implemented in a separate module under `deploy/action_manager/` and has a corresponding configuration file in `configs/action_manager/`.

*   **`BasicActionManager`** (`basic`): Directly replaces the previous chunk when a new one arrives. Simple and responsive.
*   **`OlderFirstManager`** (`older_first`): Refuses new chunks until the current chunk is sufficiently executed (controlled by `coef` parameter).
*   **`TemporalAggManager`** (`temporal_agg`): Exponentially averages old and new chunks for smoother transitions (controlled by `coef` parameter).
*   **`TemporalOlderManager`** (`temporal_older`): Combines temporal aggregation with older-first strategy (uses both `coef` and `older_coef`).
*   **`DelayFreeManager`** (`delay_free`): Compensates for inference/network delays by skipping outdated actions (uses `duration` parameter).
*   **`TruncatedManager`** (`truncated`): Truncates action chunks by dropping the first and last portions, then executes like OlderFirstManager (uses `start_ratio`, `end_ratio`, and `older_coef` parameters).

For detailed information about each manager and their parameters, see `configs/action_manager/README.md`.

## Configuration

You select and configure the Action Manager via command-line arguments or configuration files in `vlastudio infer`.

### Using Config Names

```bash
# Use older_first manager with default config values
vlastudio infer --runtime current --action_manager older_first

# Use temporal aggregation with default config values
vlastudio infer --runtime current --action_manager temporal_agg

# Use delay-free manager with default config values
vlastudio infer --runtime current --action_manager delay_free
```

### Using Custom Config Files

**To customize parameters, create a custom config file:**

```bash
# Create your own config file with custom parameters
cat > configs/action_manager/my_tuned.yaml << EOF
name: my_tuned
manager_name: TemporalAggManager
coef: 0.15
EOF

# Use the custom config
vlastudio infer --runtime current --action_manager my_tuned

# Or use the full path
vlastudio infer --runtime current --action_manager configs/action_manager/my_tuned.yaml
```

**Note:** All parameters must be defined in YAML config files. Command-line parameter overrides (like `--manager_coef`) are not supported to ensure reproducible, version-controlled configurations.

### Legacy Support

For backward compatibility, you can still use class names:

```bash
vlastudio infer --runtime current --action_manager OlderFirstManager
```

This will use the default parameters for that manager class.

## Customization

You can create your own Action Manager by:

### 1. Create a New Manager Class

Create a new file in `deploy/action_manager/` (e.g., `my_custom.py`):

```python
from .basic import BasicActionManager

class MyCustomManager(BasicActionManager):
    """My custom action management strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.my_param = getattr(config, 'my_param', 1.0)
    
    def put(self, chunk, timestamp: float = None):
        # Custom logic for storing action chunks
        pass
    
    def get(self, timestamp: float = None):
        # Custom logic for retrieving actions
        pass
```

### 2. Register in `__init__.py`

Add your manager to `deploy/action_manager/__init__.py`:

```python
from .my_custom import MyCustomManager

__all__ = [
    # ... existing managers ...
    'MyCustomManager',
]
```

### 3. Update the Loader

Add your manager to the `MANAGER_MAP` in `deploy/action_manager/loader.py`:

```python
MANAGER_MAP = {
    # ... existing mappings ...
    'MyCustomManager': MyCustomManager,
    'my_custom': MyCustomManager,
}
```

### 4. Create a Config File (Optional)

Create `configs/action_manager/my_custom.yaml`:

```yaml
name: my_custom
manager_name: MyCustomManager

my_param: 1.5
```

### 5. Create a Config File and Use Your Custom Manager

```bash
# Create config file
cat > configs/action_manager/my_custom.yaml << EOF
name: my_custom
manager_name: MyCustomManager
my_param: 2.0
EOF

# Use it
vlastudio infer --runtime current --action_manager my_custom
```

This allows you to experiment with different interpolation strategies (e.g., linear, spline) or buffering techniques to achieve the smoothest possible robot motion.
