# Action Manager Configuration

Action managers control how action chunks from the policy are buffered and executed in the real-world control loop.

## Configuration Format

All action manager configs use the following YAML format:

```yaml
type: deploy.action_manager.module_name.ClassName  # Full class path (required)
name: config_name                                   # Config name (optional)
args:                                               # Constructor arguments (optional)
  param1: value1
  param2: value2
```

The `type` field specifies the full Python module path to the manager class. The `args` field contains keyword arguments passed directly to the class constructor.

## Available Managers

### BasicActionManager (`basic.yaml`)
The simplest manager that directly replaces the previous action chunk when a new one arrives.

**Use case:** When you want immediate responsiveness and don't need smoothing.

**Parameters:** None

---

### OlderFirstManager (`older_first.yaml`)
Refuses new action chunks until the current chunk is sufficiently executed.

**Use case:** When you want to ensure action chunks are mostly completed before accepting new ones, avoiding interruptions.

**Parameters:**
- `coef` (default: 1.0): Threshold ratio. New chunks are accepted only when `current_step >= len(chunk) * coef`
  - `1.0`: Wait until current chunk is fully executed
  - `0.75`: Accept new chunk after 75% of current chunk is executed
  - `0.5`: Accept new chunk after 50% of current chunk is executed

---

### TemporalAggManager (`temporal_agg.yaml`)
Smoothly blends old and new action chunks using exponential averaging.

**Use case:** When you want smooth transitions between action chunks to reduce jerkiness.

**Parameters:**
- `coef` (default: 0.1): Exponential averaging weight for old actions
  - Formula: `action = (1 - coef) * new_action + coef * old_action`
  - `0.0`: No smoothing (same as BasicActionManager)
  - `0.1`: Light smoothing (recommended)
  - `0.5`: Heavy smoothing (may feel sluggish)

---

### TemporalOlderManager (`temporal_older.yaml`)
Combines temporal aggregation with the older-first strategy.

**Use case:** When you want both smooth transitions AND controlled chunk replacement.

**Parameters:**
- `coef` (default: 0.1): Exponential averaging weight (same as TemporalAggManager)
- `older_coef` (default: 0.75): Threshold ratio for accepting new chunks (same as OlderFirstManager)

---

### DelayFreeManager (`delay_free.yaml`)
Compensates for inference/network delay by skipping outdated actions in the chunk.

**Use case:** When you have variable inference delays and want to maintain real-time performance.

**Parameters:**
- `duration` (default: 0.05): Expected duration for each action step in seconds
  - Used to calculate: `skip_count = delay_time / duration`
  - For 20Hz control: `duration = 0.05` (50ms)
  - For 50Hz control: `duration = 0.02` (20ms)

---

### TruncatedManager (`truncated.yaml`)
Truncates action chunks by dropping the first and last portions, then executes the remaining middle portion like OlderFirstManager.

**Use case:** When you want to:
- Skip warm-up actions that may be unstable (drop beginning)
- Avoid uncertain far-future predictions (drop end)
- Still maintain controlled chunk replacement (like OlderFirstManager)

**Parameters:**
- `start_ratio` (default: 0.0): Ratio of actions to drop from the beginning (0.0 to 1.0)
  - **Important**: NOT applied to the first chunk (ensures smooth startup)
  - Only subsequent chunks have their beginning truncated
  - `0.0`: No truncation at start
  - `0.1`: Drop first 10% of actions (except first chunk)
  - `0.2`: Drop first 20% of actions (except first chunk)
- `end_ratio` (default: 0.0): Ratio of actions to drop from the end (0.0 to 1.0)
  - `0.0`: No truncation at end
  - `0.2`: Drop last 20% of actions
  - `0.3`: Drop last 30% of actions
- `older_coef` (default: 1.0): Threshold ratio for accepting new chunks (same as OlderFirstManager)
  - `1.0`: Wait until current truncated chunk is fully executed
  - `0.75`: Accept new chunk after 75% of truncated chunk is executed

**Example use cases:**
1. **Conservative** (`truncated_conservative.yaml`): `start_ratio=0.1, end_ratio=0.2, older_coef=1.0`
   - Drop first 10% (warm-up) and last 20% (uncertain predictions)
   - Wait for full execution before accepting new chunk
2. **Aggressive** (`truncated_aggressive.yaml`): `start_ratio=0.05, end_ratio=0.25, older_coef=1.0`
   - Drop first 5% and last 25% (very uncertain predictions)

---

### InferWithFPSManager (`infer_w_fps.yaml`)
Triggers inference at a minimum FPS rate, enabling asynchronous/pipelined inference.

**Use case:** When you want to:
- Hide inference latency by overlapping inference with action execution
- Maintain a consistent inference rate regardless of chunk consumption speed

**Parameters:**
- `fps` (default: 10.0): Minimum inference frequency (inferences per second)
  - `fps=10` → inference triggered at least every 100ms
  - `fps=50` → inference triggered at least every 20ms

---

## Usage Examples

### 1. Using in `eval_real.py` with command line

```bash
# Use basic manager (default)
python eval_real.py -am basic

# Use older_first manager (uses default config values)
python eval_real.py -am older_first

# Use truncated_conservative preset
python eval_real.py -am truncated_conservative

# Override parameters via command line
python eval_real.py -am truncated_conservative \
    --manager.start_ratio 0.15 \
    --manager.end_ratio 0.25 \
    --manager.older_coef 0.9
```

### 2. Using with custom config file

To use custom parameters, create a custom config file:

```yaml
# configs/action_manager/my_custom.yaml
type: deploy.action_manager.truncated.TruncatedManager
name: my_custom
args:
  start_ratio: 0.12
  end_ratio: 0.22
  older_coef: 0.88
```

Then use it:
```bash
# Use the custom config (by name)
python eval_real.py -am my_custom

# Or use the full path
python eval_real.py -am configs/action_manager/my_custom.yaml

# Override specific parameters via command line
python eval_real.py -am my_custom --manager.older_coef 0.95
```

**Command-line overrides** (`--manager.xxx`) have the highest priority and will override values in the YAML file.

### 3. Programmatic usage

```python
from deploy.action_manager import load_action_manager

# Load from config name (recommended)
manager = load_action_manager('basic')
manager = load_action_manager('temporal_agg')

# Load from YAML file path
manager = load_action_manager('configs/action_manager/my_custom.yaml')

# Load from config dict
config = {
    'type': 'deploy.action_manager.truncated.TruncatedManager',
    'args': {
        'start_ratio': 0.1,
        'end_ratio': 0.2
    }
}
manager = load_action_manager(config=config)
```

## Creating Custom Configurations

Create a new YAML file in `configs/action_manager/` with your custom parameters:

```yaml
# my_custom.yaml
type: deploy.action_manager.temporal_agg.TemporalAggManager
name: my_custom
args:
  coef: 0.15  # Custom smoothing coefficient
```

Then use it:
```bash
python eval_real.py --action_manager my_custom
```

You can also use the config without placing it in `configs/action_manager/`:
```bash
python eval_real.py --action_manager /path/to/my_custom.yaml
```

## Choosing the Right Manager

| Priority | Manager | Reason |
|----------|---------|--------|
| Responsiveness | BasicActionManager | No delay, immediate updates |
| Smoothness | TemporalAggManager | Blends actions for smooth motion |
| Stability | OlderFirstManager | Avoids frequent interruptions |
| Balanced | TemporalOlderManager | Smooth + stable |
| Low Latency | DelayFreeManager | Compensates for delays |
| Robustness | TruncatedManager | Filters unstable/uncertain actions |
| Async Inference | InferWithFPSManager | Overlaps inference with execution |

## Parameter Configuration

**Parameters can be specified in YAML config files and overridden via command line.** This ensures:
- Clear, version-controlled configurations (YAML files)
- Flexible experimentation (command-line overrides)
- Reproducible experiments (commit configs to git)
- Easy sharing and iteration

**Priority order (highest to lowest):**
1. **Command-line overrides** (`--manager.xxx`)
2. **Custom YAML config file** (if specified)
3. **Default YAML config file** in `configs/action_manager/`
4. **Default values in manager class constructor**

### Example: Parameter Override

```bash
# YAML file has: start_ratio: 0.1, end_ratio: 0.2
python eval_real.py -am truncated_conservative

# Override end_ratio via command line
python eval_real.py -am truncated_conservative --manager.end_ratio 0.3
# Result: start_ratio=0.1 (from YAML), end_ratio=0.3 (from CLI)
```

For detailed information, see `CONFIGLOADER_INTEGRATION.md`.
