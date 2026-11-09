# Interactive benchmarks

The deployment profiles below use `BenchmarkRobot`, simulator-specific wrappers,
and action adapters. They do not replace the original benchmark evaluation
classes or their return contracts. One dashboard can select different installed
simulator environments using a site runtime registry:

```bash
python -m vlastudio.webui --host 127.0.0.1 --port 8765 --runtimes /site/webui-runtimes.json
```

Robot profiles declare a runtime name (`metaworld`, `rlbench`, `aloha`, `libero`, `robosuite`, `robotwin`, `behavior`).
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
    },
    "robosuite": {
      "command": ["/site/envs/robosuite/bin/python"],
      "env": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
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
| `robosuite` | Lift, PickPlaceCan, NutAssemblySquare, ToolHang (Panda) | 7 native OSC/gripper values; `ee/delta` | Single arm, XYZ, rotation and gripper |
| `robosuite_transport` | TwoArmTransport (two Pandas) | 14 values, robot0 then robot1; `ee/delta` | Both arms, with a separate 20 Hz keyboard profile |
| `robotwin` | click_bell, adjust_bottle | 16 values: left then right XYZ + quaternion WXYZ + opening; `ee/abs` | Both arms using the installed native planner |
| `behavior` | R1Pro empty scene; configured household tasks require full licensed assets | Native controller vector, currently 21 values; `ee/delta` | Holonomic base, both arms, rotation and grippers |

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

The robosuite profiles use robosuite 1.5 BASIC composite controllers with
world-frame delta OSC_POSE and robomimic 0.4, in a runtime separate from
LIBERO's robosuite 1.4. Keyboard translation/rotation increments are converted
using each active arm controller's input/output bounds. Native policy actions
already use those controller units; positive gripper effort closes, negative
opens. Unsupported controller layouts fail explicitly instead of silently
reordering actions. See the [robosuite 1.5 controller implementation](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/controllers/parts/arm/osc.py).

Their `state` retains the original robomimic benchmark's 8 values per robot:
XYZ, axis-angle rotation, and both finger joint positions. `state_ee` exposes
the same measured arm fields; `state_joint` has 9 values per Panda. Images use
the original 256 x 256 size. Single-arm camera order is scene then wrist;
ToolHang uses sideview as its scene camera. Transport uses shoulder0,
shoulder1, wrist0, wrist1. Match the camera selection to the policy's data.
These interactive profiles continue past the native horizon and task success;
Reset explicitly restores the seeded scene. `environment.args.continuous: false`
restores the native horizon, reported as a truncation in the WebUI.
CPU software rendering works, but Transport's four cameras can make its actual
step rate much lower than the configured 20 Hz. Keyboard motion then advances
more slowly in wall time; use a GPU renderer for smoother interaction.

QZ CPU checks covered XYZ, rotation, grippers and seeded Reset for all five
robosuite tasks. The Lift and Transport WebUI sessions also covered camera
preview, HDF5 recording, keyboard response after 550 steps, pause, Reset and
TCP policy round trips through the existing collection/inference entrypoints.
The TCP checks used a diagnostic policy; learned task success was not evaluated.

## RoboTwin GPU runtime

Reuse the installed RoboTwin environment and assets. Current RoboTwin uses
`env_cfg/task_config` and the public `scripts.eval_policy_xpolicylab` loader;
the deployment wrapper handles that layout without modifying native task code.
The runtime registry entry needs a launcher and `VLASTUDIO_RUNTIME_ROBOTWIN_ROOT`.
For the prepared QZ launcher, set `ROBOTWIN_EXTRA_PYTHONPATH` to this checkout's
`src` directory so collection, inference and preview run the same source.

The profile selects `args.process_start_method: spawn` and
`args.spawns_children: true`. CuRobo initializes CUDA during import and cannot
run in a forked device after the inference supervisor has initialized CUDA.
Spawn explicitly transfers the private WebUI control mailbox. Existing profiles
retain their normal multiprocessing defaults. The non-daemon device also allows
native planners that create their own child processes.

`state` and `state_ee` retain the native 16-value end-pose representation;
`state_joint` contains 14 measured joint/gripper values. The three RGB cameras
are head, left wrist and right wrist, at 320 x 240. The planner runs each
increment to completion, so configured 20 Hz is not a wall-clock rate guarantee.
Interactive collection continues after native evaluation success; explicit Reset
restores the seeded scene. Policies must match this EE controller and image/state
contract. Joint-space policies need their own matching deployment profile.

Keyboard translation and rotation act at the measured gripper TCP, with
world-aligned axes. The adapter obtains each TCP from the loaded robot and
calibrates its rigid transform from the native EE frame after every Reset.
Rotation targets keep the gripper center fixed, then convert back to RoboTwin's
native EE action frame. The requested TCP goal is retained between frames;
servo tracking error is not accumulated as another movement command.
For the prepared embodiment those two
origins differ by 12 cm; that distance is not hard-coded in the adapter.
Preview axes are labeled `TCP-L` / `TCP-R` and use the same measured TCP origins.
The recorded `state`, `state_ee` and 16-value policy actions retain the original
native EE representation; they must not be interpreted as TCP poses.

On QZ, after lifting both arms into a reachable pose, rotations of about 0.20 rad
around each world X/Y/Z axis kept both measured TCPs within 0.08 mm of their
starting positions. The native home pose did not track a larger positive X
rotation: a planned joint target was not physically reached. TCP target
conversion does not remove native motion constraints; lift the arms before
attempting that rotation from home.

On QZ RTX 4090, WebUI checks covered both arms and grippers, three camera marker
packets, unmarked HDF5 recording, pause, seeded Reset, and TCP policy requests
through the existing `collect_data` / `eval_real` entrypoints. The policy check
used a diagnostic hold policy and does not establish learned task success.

## BEHAVIOR / OmniGibson mobile robot

The `behavior` profile uses OmniGibson 3.7.2 / Isaac Sim 4.5 with R1Pro.
Its dedicated runtime needs `VLASTUDIO_RUNTIME_BEHAVIOR_ROOT`,
`OMNIGIBSON_DATA_PATH`, `OMNIGIBSON_APPDATA_PATH`, `OMNIGIBSON_HEADLESS`,
and the accepted Isaac Sim EULA setting. Install simulator dependencies in this
runtime, separately from RoboTwin. Full household tasks additionally require
the licensed BEHAVIOR assets and challenge task instances. The `empty` scene
only exercises the robot and floor; it is not a household-task success test.

Use `keyboard_mobile_2arm`: Up/Down move forward/back, Left/Right strafe,
and Home/End turn. Base velocity is in the robot frame; arm XYZ/rotation keys
retain world-aligned axes and are converted to the robot frame for native IK.
Release, focus loss, input timeout and pause produce zero base velocity.

The adapter obtains action slots from `controller_action_idx`, rather than
assuming two seven-value arms. With the configured R1Pro controllers the native
policy/recording action is: base body velocity (3), absolute trunk positions (4),
left IK pose delta (6), left gripper (1), right IK pose delta (6), right gripper
(1). Translation uses metres, rotation radians and base velocity metres/second
or radians/second; smooth gripper input is in [-1, 1]. Policies must match this
controller configuration, not the original benchmark's joint-controller setup.
The policy path forwards this full native vector without teleoperation scaling.

`state` and `state_joint` contain all 28 measured joint coordinates, including
virtual base and trunk joints. `state_ee` contains two world XYZ + quaternion
XYZW + mean measured finger position blocks; the finger position is in metres.
Camera order is head, left wrist, right wrist, at 320 x 240. Optional UI markers
read native camera calibration and measured end-effector poses. They are drawn
on preview copies and removed from the recorded frame metadata.
The initial end effectors can be outside the head camera's view; their axes are
visible in the wrist views. Out-of-view marker origins are omitted.

The deployment wrapper leaves benchmark classes and upstream simulation code
unchanged. Its dedicated device stops after the current native frame and closes
Kit before Python finalization; it allows 30 seconds for process shutdown.
The profile uses spawn and limits native task/physics threads to eight for the
10-CPU QZ instance. Cold initialization can take several minutes; it has a
900-second connection timeout. Warm starts also depend on shader and filesystem
caches, so a requested control rate is not an initialization-time guarantee.

QZ checks with driver 550.163.01 and the prepared Python 3.10.14 runtime covered
all six base-key directions, both arms and grippers, release/timeout stops,
three camera marker packets, unmarked HDF5 recording of 21-value native actions,
pause, Reset, TCP diagnostic policy requests and clean process shutdown.
The `assembling_gift_baskets` household scene also passed base motion, both-arm
rotation, three-camera recording, pause, Reset and clean shutdown through the
existing collection entrypoint. Its first initialization took about 552 seconds;
recording ran at roughly 5 FPS on this 10-CPU / RTX 4090 instance. Configuring
20 Hz does not guarantee that wall-clock rate. TCP policy checks used the empty
scene and a diagnostic policy; learned task success was not evaluated.

The fixed household archives passed their complete SHA256 checks. The native
loader nevertheless warned that some object USD hashes differ from their saved
states. Keep that warning when assessing asset/task-instance compatibility for
formal benchmark evaluation; interactive control checks do not establish it.

## Optional UI marker providers

Simulator classes and action adapters have no display-method requirement.
A robot profile opts in by binding a provider in its `webui` section:

```yaml
webui:
  marker_provider: vlastudio.webui.marker_providers.RLBenchMarkerProvider
```

A specification may also use `{type: my_project.ui.Markers, args: {...}}`.
The UI creates `Markers(environment, image_names=[...], **args)` in the owning
simulator process after Reset. The provider may cache static calibration,
frame names or a kinematic model there. It implements:

```python
def get_marker_from_observation(self, observation):
    return {
        "version": 1,
        "images": {
            "head_camera": [
                {"type": "axes", "label": "EE",
                 "points": [[0.5, 0.5], [0.6, 0.5], [0.5, 0.4], [0.5, 0.6]]},
                {"type": "joint", "id": "J1", "center": [0.3, 0.4],
                 "positive_direction": [1.0, -1.0], "radius": 0.025,
                 "value": 0.4},
            ]
        },
    }
```

`observation` is the read-only MetaObs field dictionary for that control frame.
A provider can use it, read existing native FK/frame transforms from its bound
instance, or compute FK from its own model. It must not step/reset the simulator,
change its model, draw into RGB, or mutate observation arrays. Built-in RLBench
and robosuite providers read native tip/site transforms, with no dependency on
`state_ee` dimensions. Moving camera poses are read for the same frame. Providers
are rebound after Reset so changed scene handles/calibration cannot become stale.

The shared boundary is a versioned packet of **2D markers**, not a prescribed
camera or robot-state representation. `images` maps the profile's image names
to marker lists. Point coordinates are normalized to the image: `(0,0)` is the
top-left pixel center and `(1,1)` the bottom-right. Endpoints outside the image
are allowed. Nothing simulator-specific reaches the drawing process.

| Type | Required fields | Optional fields / behavior |
| --- | --- | --- |
| `axes` | `points`: origin followed by X/Y/Z endpoints, four 2D points | `label`; `toward_viewer`: three booleans for axes nearly aligned with the camera; an unavailable endpoint can be `None`. Uses red X, green Y, blue Z. |
| `joint` | `center`: 2D point; `positive_direction`: nonzero image-plane vector | `id`, `label`, numeric `value`, `radius` as a fraction of the shorter image side (default 0.025). Draws a circle, positive/negative direction, and ID/value. The provider determines what positive joint motion means. |

`draw_markers(image, markers)` returns an annotated BGR copy. It knows marker
categories, not simulation APIs or FK. Unknown types and invalid markers are
omitted independently. For a custom category, configure a trusted callback:

```yaml
webui:
  marker_provider: my_project.ui.Markers
  marker_renderers:
    contact: my_project.ui.draw_contact
```

A renderer has signature `draw_contact(image, marker)` and draws on the supplied
private preview buffer. The module must be available in the selected runtime.
No provider configuration, `None`, an empty packet, unavailable geometry, or a
provider failure all leave the RGB preview usable without those markers.

Provider binding uses the existing WebUI worker setup hook and a per-instance
observation observer. It does not add methods to or modify the simulator class.
Ordinary CLI training/evaluation/collection does not bind UI providers. Built-in
bindings cover RLBench, robosuite, ALOHA EE, LIBERO, RoboTwin and BEHAVIOR;
profiles can choose other providers or omit them entirely.
The built-in providers currently emit EE axes. Custom providers can emit joint
markers with their own joint IDs and projected positive-motion directions.

`__preview__` carries only the typed marker packet. The monitor draws on a private
copy before JPEG encoding. Policy conversion ignores this side channel, and the
recording conversion/save hook excludes it; original RGB is recorded. No marker
objects are inserted into a simulator scene. The earlier environment-level
`get_preview_metadata` and action-adapter `preview_data` methods are removed.

Policy output already uses native units and bypasses teleoperation scaling.
The UI accepts a VLAStudio server address or compatible local checkpoint.
Local checkpoint inference automatically uses CUDA when available and CPU
otherwise. API callers can select `device: cpu` or `device: cuda` explicitly.
RoboTwin and BEHAVIOR use the GPU deployment profiles described above; they are
not part of the CPU simulator checks.

## Reproduce the checks

```bash
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
