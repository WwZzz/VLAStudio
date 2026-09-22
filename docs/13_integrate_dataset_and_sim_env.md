# Integrating a Dataset and a Simulation Environment

This guide is for contributors who want to plug **a new dataset** or **a new
simulation benchmark** into VLAStudio *without modifying the package source*.
Everything is config-driven: you write one Python class, reference it from a YAML
config by its import path, and the built-in `train` / `evaluate` flows pick it up.

Two extension points are covered:

1. [Datasets](#1-integrating-a-dataset) ??training data for policies.
2. [Simulation environments](#2-integrating-a-simulation-environment) ??closed-loop
   evaluation of a trained policy.

---

## 0. How VLAStudio resolves your class

Every `type:` field in a config is resolved by `vlastudio.extensions.resolve`.
You may use any of:

| Form | Example |
| --- | --- |
| packaged module | `vlastudio.data_utils.datasets.lerobotv30_wrapper.WrappedLerobotV30Dataset` |
| your own package | `my_pkg.datasets:MyDataset` |
| a single file | `/abs/path/my_dataset.py:MyDataset` or `examples/my_dataset.py:MyDataset` |
| a package directory | `/abs/path/my_env_pkg/` (loads `__init__.py`) |
| an installed entry point | `@dataset/my_dataset`, `@env/my_env` (see ?1.5, ?2.5) |

Nothing else is required ??no registry edits, no package reinstall (when using a
file path). Configs are plain YAML, loaded by `vlastudio.configs.loader.ConfigLoader`.

---

## 1. Integrating a dataset

### 1.1 What training consumes

A dataset is a `torch.utils.data.Dataset`. Each `__getitem__(i)` must return a
dict with (at least) these keys:

```python
{
  "image":     np.ndarray | torch.Tensor,  # (K, 3, H, W) uint8, one entry per camera
  "state":     torch.Tensor,               # (state_dim,)  float32
  "action":    torch.Tensor,               # (chunk_size, action_dim) float32
  "is_pad":    torch.Tensor,               # (chunk_size,) bool ??True = padded step
  "raw_lang":  str,                        # language instruction (may be "")
  # optional:
  "reasoning": dict,
  "timestamp": int,
  "episode_id":int,
}
```

Notes

- `chunk_size` is the action horizon; the last action dimension is conventionally
  the gripper.
- `state` and `action` are **raw** here (no normalization). Normalization is
  applied by the policy/training layer using the `meta` block of the config
  (`action_normalize` / `state_normalize`, see ?1.4).
- `image` is fed to the policy after the shared resize/pad utilities, so any
  resolution is accepted as long as `meta.image_size` matches the env you evaluate in.

The base class `vlastudio.data_utils.datasets.base.EpisodicDataset` implements the
common machinery for HDF5 episode datasets (episode indexing, chunked sampling,
optional preloading). Subclass it if your data is HDF5; otherwise implement a
plain `torch.utils.data.Dataset`.

### 1.2 Prefer reusing `WrappedLerobotV30Dataset`

If your data is in **LeRobot v3.0** format you do **not** need a new class:

```yaml
datasets:
- type: vlastudio.data_utils.datasets.lerobotv30_wrapper.WrappedLerobotV30Dataset
  name: my_task
  args:
    dataset_path_list: [<hf-repo-id or local path>]
    chunk_size: 50
    ctrl_space: joint
    ctrl_type: abs
    state_key: observation.state       # feature name in meta/info.json
    action_key: action
    camera_names:
      - observation.images.cam_high
      - observation.images.cam_left_wrist
      - observation.images.cam_right_wrist
    image_size: [640, 480]             # [W, H]
    local_only: false
    download_videos: true
    # scope a subset of episodes (e.g. a single task):
    # episode_filter: {episode_index: [0, 1, 2, ...]}
```

It reads the dataset directory directly from parquet + mp4 (no `lerobot`
dependency) and downloads from the Hub on first use.

### 1.3 Writing your own dataset class

```python
# my_pkg/datasets.py
import numpy as np
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, chunk_size: int = 50, image_size=(640, 480), **kwargs):
        # build an index of (episode, start_frame) here
        self.samples = ...
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ep, t = self.samples[i]
        images = ...   # (K, 3, H, W) uint8
        state = ...    # (state_dim,) float32
        action = ...   # (chunk_size, action_dim) float32, zero-padded at the tail
        is_pad = ...   # (chunk_size,) bool
        return {
            "image": images,
            "state": torch.as_tensor(state, dtype=torch.float32),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "is_pad": torch.as_tensor(is_pad, dtype=torch.bool),
            "raw_lang": "stack the blocks",
        }
```

Any `**kwargs` the trainer passes (e.g. `subsample`, `preload_data`) can be
accepted and ignored if unused.

### 1.4 The task config

A dataset is described by a task config under `src/vlastudio/configs/task/<group>/<name>.yaml`
(a path also works). Example (`configs/task/tabletop_sim/dish_drainer.yaml`):

```yaml
name: my_task
datasets:
- type: my_pkg.datasets:MyDataset      # or a file path, see ?0
  name: my_task
  args:
    data_root: /path/to/data
    chunk_size: 50
    ctrl_space: joint
    ctrl_type: abs
    image_size: [640, 480]
    camera_names: [cam_high, cam_left_wrist, cam_right_wrist]
meta:
  action_dim: 14
  state_dim: 14
  image_size: [640, 480]
  action_normalize: zscore
  state_normalize: zscore
```

The `meta` block is consumed by the training/action layers:

| key | meaning |
| --- | --- |
| `action_dim` / `state_dim` | dimensionality of the raw action/state vectors |
| `image_size` | `[W, H]` used for the policy + evaluation camera |
| `action_normalize` / `state_normalize` | `zscore`, `minmax`, or `none` |
| `ctrl_space` / `ctrl_type` | passed through to the policy's MetaAction convention |

`datasets` is a **list**: you can concatenate several sources, or add weighted /
transformed variants (see the other `example_*.yaml` files in the same folder).

### 1.5 Registering (optional)

Installed packages can expose datasets as entry points so configs can use a short
name. In your `pyproject.toml`:

```toml
[project.entry-points."vlastudio.dataset"]
my_task = "my_pkg.datasets:MyDataset"
```

Then `type: @dataset/my_task` works everywhere.

---

## 2. Integrating a simulation environment

### 2.1 The `MetaEnv` contract

A benchmark is a subclass of `vlastudio.benchmark.base.MetaEnv`, which wraps a raw
environment (e.g. a `dm_env`/IsaacLab env) and translates between its native
obs/action and the VLAStudio `MetaObs`/`MetaAction` convention:

```python
from vlastudio.benchmark.base import MetaEnv, MetaObs, MetaAction

class MyEnv(MetaEnv):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.camera_names = list(getattr(config, "camera_names", ["cam_high"]))
        self.image_size = tuple(getattr(config, "image_size", [640, 480]))  # (W, H)
        self.max_timesteps = int(getattr(config, "max_timesteps", 250))
        self.ctrl_space = getattr(config, "ctrl_space", "joint")
        self.ctrl_type = getattr(config, "ctrl_type", "abs")
        super().__init__(self._build_env())          # env is created once

    def _build_env(self):
        return make_my_sim(self.config.task)         # your simulator handle

    # ---- REQUIRED: raw <-> meta conversion ----
    def obs2meta(self, raw_obs) -> MetaObs:
        # return the CANONICAL observation; do NOT normalize here
        return MetaObs(
            state=np.asarray(raw_obs["joint_state"], dtype=np.float32),
            state_joint=np.asarray(raw_obs["joint_state"], dtype=np.float32),
            image=self._stack_cameras(raw_obs["images"]),   # (K, C, H, W) uint8
            raw_lang=str(raw_obs.get("instruction", "")),
            timestep=int(raw_obs.get("step", -1)),
        )

    def meta2act(self, maction: MetaAction):
        # map the policy action onto the raw env action; NO normalization here
        return np.asarray(maction.action, dtype=np.float32)

    # ---- OPTIONAL overrides ----
    def reset(self):
        raw = self.env.reset()
        self.prev_obs = self.obs2meta(raw)
        return self.prev_obs

    def close(self):
        self.env.close()
```

**Which methods must you implement?**

| method | required | purpose |
| --- | --- | --- |
| `__init__(config, ...)` | yes | read the config, build the raw env, set `max_timesteps` / `ctrl_space` / `ctrl_type` / cameras |
| `obs2meta(raw_obs) -> MetaObs` | **yes** | canonicalize one observation |
| `meta2act(MetaAction) -> raw_action` | **yes** | map a policy action to the raw env action |
| `reset()` | no | defaults to `env.reset()` + `obs2meta`; override for seeding / benchmark init poses |
| `step()` | no | inherited: `meta2act ??env.step ??obs2meta` |
| `close()` | no | inherited: `env.close()` |

A module-level factory `create_env(config)` is the conventional entry used by the
loader; returning `MyEnv(config)` is enough:

```python
def create_env(config):
    return MyEnv(config)
```

### 2.2 `MetaObs` and `MetaAction`

`MetaObs` (`vlastudio.benchmark.base`) ??fill what your env has, leave the rest `None`:

| field | shape | meaning |
| --- | --- | --- |
| `state` | `(S,)` | generic state (required by most policies) |
| `state_joint` / `state_ee` / `state_obj` | `(S,)` | typed variants |
| `image` | `(K, C, H, W)` | uint8 RGB, one per `camera_names` entry |
| `depth` | `(K, H, W)` | optional |
| `pc` | `(N, 3)` | optional point cloud |
| `raw_lang` | `str` | language instruction |
| `timestep` | `int` | step counter |

`MetaAction` ??what the policy returns and you must consume:

| field | meaning |
| --- | --- |
| `action` | `(A,)` array; the last dim is the gripper by convention |
| `ctrl_space` | `joint` / `ee` / `other` |
| `ctrl_type` | `abs` / `rel` / `delta` |
| `gripper_continuous` | whether the gripper is a continuous position ratio |

The evaluator normalizes/denormalizes around `meta2act`/`obs2meta` using the
`ctrl_space` / `ctrl_type` you declare, so **keep those consistent with your
raw action semantics**.

### 2.3 The env config

```yaml
# configs/env/<group>/<name>.yaml
type: my_pkg.env:MyEnv            # or a file path, see ?0
name: my_env
args:
  task: my_task                   # simulator task id
  action_space: joint_pos
  ctrl_space: joint
  ctrl_type: abs
  max_timesteps: 250              # per-episode horizon
  camera_names: [cam_high, cam_left_wrist, cam_right_wrist]
  image_size: [640, 480]          # [W, H]
```

### 2.4 How evaluation drives your env

`vlastudio evaluate` (and `vla.load_env(...).evaluate(...)`) does, per rollout:

```
obs = env.reset()                       # -> MetaObs
for t in range(max_timesteps):
    maction = action_manager.act(obs)   # policy chunk -> MetaAction
    obs, reward, done, info = env.step(maction)   # meta2act -> env.step -> obs2meta
```

Keep `camera_names` / `image_size` **identical to the training config** and
`ctrl_space` / `ctrl_type` consistent with how the policy was trained. Reward and
success are read from the env's returned `info` (see `benchmark/utils.evaluate`).

### 2.5 Registering

Environments are resolved through the same extension mechanism as every other
kind (`vlastudio.extensions.resolve`, `kind="env"`). Your `type:` may be a
packaged dotted path, your own package, a local file path, or an installed
entry point ??exactly the forms in ?0. The reference may resolve to a module that
exposes `create_env(config)`, or directly to a class / callable used as the
factory.

```toml
[project.entry-points."vlastudio.env"]
my_task = "my_pkg.env:MyEnv"           # then: type: @env/my_task
```

The loader (`vlastudio/entrypoints/eval_sim.py::load_env_module`) also picks up an
optional module-level `evaluate(...)` to override the default rollout loop, so you
can ship environment-specific success metrics next to the env class.

---

## 3. Train + evaluate

```bash
# Train a policy on your task config
vlastudio train --policy smolvla --dataset configs/task/my_group/my_task.yaml \
    --training smolvla --output-dir checkpoints/my_task

# Serve the trained policy and evaluate it in your env
vlastudio serve --policy <checkpoint> --address 0.0.0.0:5000 &
vlastudio evaluate --env configs/env/my_group/my_env.yaml \
    --policy configs/policy/smolvla.yaml --checkpoint checkpoints/my_task \
    --output-dir results/my_task --num-rollout 4
```

From Python the same flow is:

```python
import vlastudio as vla
dataset = vla.load_dataset("my_group/my_task")     # resolves configs/task/my_group/my_task.yaml
policy  = vla.load_policy("configs/policy/smolvla.yaml")
vla.train(policy, dataset, "smolvla", output_dir="checkpoints/my_task")

env = vla.load_env("my_group/my_env")              # resolves configs/env/my_group/my_env.yaml
remote = vla.connect_policy("127.0.0.1:5000")
env.evaluate(remote, output_dir="results/my_task", num_rollout=4)
```

---

## 4. Worked example: RoboDojo `stack_blocks`

RoboDojo is an eval-only benchmark built on Isaac Sim 5.1 / IsaacLab 2.3 with
official LeRobot v3.0 data. The integration is exactly the two extension points
above.

**Dataset** ??reuse `WrappedLerobotV30Dataset` (no new class):

```yaml
# configs/task/robodojo/stack_blocks.yaml
name: robodojo_stack_blocks
datasets:
- type: vlastudio.data_utils.datasets.lerobotv30_wrapper.WrappedLerobotV30Dataset
  name: robodojo_stack_blocks
  args:
    dataset_path_list: [<robodojo lerobot v3.0 root for stack_blocks>]
    chunk_size: 50
    ctrl_space: joint
    ctrl_type: abs
    state_key: observation.state
    action_key: action
    camera_names:
      - observation.images.cam_high
      - observation.images.cam_left_wrist
      - observation.images.cam_right_wrist
    image_size: [640, 480]
    episode_filter: {episode_index: [...]}   # the stack_blocks episodes
meta:
  action_dim: 14
  state_dim: 14
  image_size: [640, 480]
  action_normalize: zscore
  state_normalize: zscore
```

**Environment** ??a thin `MetaEnv` around RoboDojo's simulator client
(`src/eval_client`). Implement `obs2meta` (map RoboDojo's camera streams +
14-DoF joint state into `MetaObs`) and `meta2act` (map the policy's 14-DoF joint
action back to the client action). Keep `camera_names` / `image_size` equal to the
dataset config, and declare `ctrl_space: joint` / `ctrl_type: abs`.

With those two classes + two YAML files, `vlastudio train` and
`vlastudio evaluate` work end-to-end ??no source changes.
