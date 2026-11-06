# RoboCasa

The adapter lives in `benchmark/robocasa`, following the current benchmark plugin
layout. Environment selectors are `robocasa.restock_pantry` and
`robocasa.microwave_thawing`; matching dataset selectors use the same names.

```python
import vlastudio as vla
dataset = vla.load_dataset('robocasa.restock_pantry', check_dependencies=False)
env = vla.load_env('robocasa.restock_pantry', check_dependencies=False)
```

Copy the packaged YAML configs into your project and set `dataset_path` (environment)
and `dataset_path_list` (training) to the downloaded image HDF5 file or directory.
The environment reads `data.attrs['env_args']` from that file to preserve the
recorded controller and task definition. Alternatively supply explicit `env_kwargs`.
Dataset task names must agree with the environment task. These configs are templates;
they do not download demonstrations or include machine-specific absolute paths.

Install the simulation runtime with
`vlastudio env create --env robocasa.restock_pantry -n robocasa` on Linux. The
runtime pins RoboCasa 1.0 and robosuite 1.5.2. Install scene/object assets using the
[upstream instructions](https://github.com/robocasa/robocasa/tree/8f3c96ec8d1bfcd8126cad2bca887da98d30e997).
Set `MUJOCO_GL=egl` (GPU) or `osmesa` (CPU) before launching simulation; the
corresponding system rendering libraries must be installed separately. Policies
with incompatible NumPy pins should use VLAStudio's remote policy interface.
The HDF5 dataset alone requires no RoboCasa/robosuite installation.

## Contract

- Native actions are preserved, including base, torso and mode channels. The
  default PandaOmron configuration has 12 action dimensions; gripper commands
  retain the native sign convention. No seven-dimensional arm-only conversion is
  performed. Other robot/controller dimensions must be reflected in task metadata.
- State is EEF position, XYZW quaternion converted to rotation vector, and two
  gripper joint positions (8 dimensions). Raw MuJoCo `states` are never model inputs.
- Camera order is left agent view, right agent view, wrist. Samples and simulator
  observations expose uint8 `[K,C,H,W]` images. Stored RGB is assumed upright;
  online OpenGL images are flipped vertically. Both settings are explicit YAML
  options: check them against your dataset's own playback before training.
- Instructions come from each demonstration's `ep_meta.lang`, or the current
  simulator episode metadata. They are never inferred from filenames.
- `filter_by_attribute: train` selects `mask/train` and fails if it is absent.
  Create a demonstration-level train/validation mask before using these templates.
  Set this option to null explicitly when intentionally selecting every demo.
- Tail chunks are zero-padded with `is_pad`; they never read the next episode.
- Seeds use `seed + rollout_index`, advancing on repeated resets. Seeds alone do
  not guarantee identical generated scenes across machines: paired comparisons
  should also save/restore and verify model XML, episode metadata and simulator state.
- Only the simulator's task predicate counts as success. Timeout and other
  termination are failures; completed vector slots stay frozen.

## Validation

`python -m pytest tests/test_robocasa.py` exercises configuration/runtime resolution,
HDF5 masks, multi-file indexing, per-demo instructions, image/state agreement,
padding, serialization, CPU backward/checkpoint reload, and vector rollout scoring
with a simulator double. Actual kitchen rendering needs the external assets and
is not covered by the asset-free unit suite. Before a benchmark run, replay a
held-out demonstration with its original controller and verify RGB orientation,
state ordering and native action dimensions against the loaded environment.
