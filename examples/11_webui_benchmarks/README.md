# Interactive benchmarks

The deployment profiles below use `BenchmarkRobot`, simulator-specific wrappers,
and action adapters. They do not replace the original benchmark evaluation
classes or their return contracts. One dashboard can select different installed
simulator environments using a site runtime registry:

```bash
python -m vlastudio.webui --host 127.0.0.1 --port 8765 --runtimes /site/webui-runtimes.json
```

Robot profiles declare a runtime name (`metaworld`, `rlbench`, `aloha`, `libero`).
The site JSON maps names to an argument-list launcher, environment variables,
and optional extra package paths. Paths belong in site configuration, not the
versioned robot YAML. For example:

```json
{
  "runtimes": {
    "metaworld": {
      "command": ["/site/envs/metaworld/bin/python"],
      "env": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    },
    "rlbench": {
      "command": ["/site/envs/rlbench/bin/python"],
      "env": {"COPPELIASIM_ROOT": "/site/CoppeliaSim", "DISPLAY": ":99"}
    }
  }
}
```

Alternatively, `command` can be `["bash", "/site/run.sh", "rlbench"]` when a
launcher prepares native libraries before executing Python. It must forward its
arguments, inherited file descriptors and termination signals, preferably using
`exec`. Configure graphics library paths in `env` or the launcher. `extra_paths`
lists additional Python package directories; the current VLAStudio source is
also added to `PYTHONPATH`.

Select the registry with `--runtimes`, `VLASTUDIO_WEBUI_RUNTIMES`, or
`$XDG_CONFIG_HOME/vlastudio/webui-runtimes.json` (default
`~/.config/vlastudio/webui-runtimes.json`). Without a matching entry, the current
interpreter is used and must have the simulator installed. Configured runtimes
clear inherited simulator-specific Python and graphics variables before applying
their own settings. Collection, inference and preview use the selected launcher.
Imports and required settings are checked there before device processes start;
missing dependencies fail with a readable error instead of leaving a
keyboard-only session running.

| Robot profile | Task catalog | Native policy action | Teleoperation |
| --- | --- | --- | --- |
| `metaworld` | 50 v3 tasks | 4 values: normalized XYZ displacement and signed gripper effort; `ee/delta` | XYZ and gripper; fixed orientation |
| `rlbench` | 106 configured tasks | 8 values: world XYZ, quaternion XYZW, gripper opening; `ee/abs` | XYZ, rotation, gripper, using incremental IK targets |
| `aloha_sim` | Transfer cube, insertion | 16 values: two XYZ + quaternion WXYZ + gripper targets; `ee/abs` | Both arms |
| `aloha_sim_joint` | Transfer cube, insertion | 14 joint/gripper values; `joint/abs` | Inference only; use this for the original ACT model |
| `libero` | Existing task catalog | 7 native OSC/gripper values; `ee/delta` | Existing single-arm control |

Choose the same state representation, camera order, image size, action units and
controller as the policy's training data. MetaWorld exposes its original
four-value state as `state`, and additional measured joint/EE fields separately.
RLBench's `state` and `state_ee` contain the pose and gripper opening.
`aloha_sim_joint` uses one 640 x 480 camera, matching the default ACT example.

MetaWorld's fourth action is signed effort, positive for closing; it is not a
gripper opening position. Keyboard increments are converted to its native action
scale, and rotation keys are disabled. See the [official action contract](https://metaworld.farama.org/benchmark/action_space/).
The interactive profile runs continuously instead of stopping at MetaWorld's
500-step evaluation horizon. Reset is explicit. Custom deployments can set a
device `max_steps` when they intentionally need a limit, or use
`environment.args.continuous: false` to retain the native task horizon.

RLBench's interactive profile defaults to `EndEffectorPoseViaIK`, which is suited
to small keyboard increments. This is a distinct controller from the original
benchmark's planning mode. A custom profile can set `environment.args.arm_mode`
to `planning` when a policy requires it. Keyboard targets are anchored to the
measured pose; an unreachable increment is rejected without disabling reverse
motion. Native policy targets that fail IK pause the device with an explicit UI
status. Reset restores the task and reanchors the target. Both interactive profiles have no
default device step limit. Closing the deployment
wrapper shuts down CoppeliaSim; the benchmark's environment-reuse behavior is
unchanged.

MetaWorld and RLBench were exercised on QZ with CPU software rendering. RLBench
still needs a working X/Qt/OpenGL environment even with `headless: true`; CUDA is
not required for these simulator checks. Software rendering can be much slower
than the requested control rate. Each WebUI session owns its simulator in one
process. Use separate simulator environments when their dependency versions
conflict, and stop a collection session before starting another session sharing
the keyboard channel.

Recordings contain `observation/state`, `observation/state_ee`,
`observation/state_joint`, `observation/images/*`, and native executed actions in
`observation/action` and `observation/executed_action`. Raw keyboard deltas remain
in `observation/teleop/action`. Training adapters must select the native action
and the state used by the target profile; do not use raw keyboard input as the
policy target. These HDF5 files use the deployment recording schema, not the
original ALOHA or RLBench demonstration file layout. Existing demonstration
dataset loaders must not be pointed at these files without a schema adapter.

Policy output already uses native units and bypasses teleoperation scaling.
The UI accepts a VLAStudio server address or compatible local checkpoint.
Local checkpoint inference automatically uses CUDA when available and CPU
otherwise. API callers can select `device: cpu` or `device: cuda` explicitly.
RoboTwin is not included in this CPU validation: its SAPIEN renderer and EE path
planner still need a separate GPU-backed interactive adapter and live checks.

## Reproduce the checks

```bash
python -m pytest tests/test_benchmark_deployment.py -q
python examples/11_webui_benchmarks/validate_session.py \
  --profile metaworld --result /tmp/metaworld-session.json
python examples/11_webui_benchmarks/validate_session.py \
  --profile rlbench --result /tmp/rlbench-session.json
python examples/11_webui_benchmarks/validate_session.py \
  --profile aloha_sim_joint --result /tmp/aloha-joint-session.json
```

Run each command in its simulator environment, with appropriate graphics
variables already set. `--task` selects another catalog task. Each check records
a new uniquely named test dataset under `data/webui/`, then starts a local TCP
hold-pose policy and exercises inference, pause, reset and shutdown. The policy
tests transport/control plumbing; they do not establish learned task success.
The task catalog is larger than the set of tasks actually exercised.
