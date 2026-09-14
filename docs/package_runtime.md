# Package and managed runtimes (development branch)

The package is now named `vlastudio`. This branch has not been published to PyPI.
Install a locally built wheel with `pip install dist/vlastudio-*.whl`, or use
`pip install -e .` for development. `import vlastudio` does not import Torch,
TensorFlow, device SDKs, or install anything.

## Training without activating environments

```sh
vlastudio train -p act -t /my/configs/task.yaml -c /my/configs/train.yaml -o /checkpoints/run1
vlastudio train --policy /my/configs/custom_policy.yaml --policy.args.chunk_size 16
vlastudio prepare --policy act
vlastudio train --policy act --runtime current
vlastudio serve -m /checkpoints/run1
```

The training arguments are forwarded unchanged to the existing `train.py`.
`--policy` remains a config name or YAML path, not a restricted model identifier.
The working directory never changes. Relative dataset/checkpoint/plugin file paths
remain relative to the caller. Existing `python train.py` workflows remain available.
Install the light package into a source environment to enable new plugin references.

The default `managed` mode resolves a profile before importing the policy, prepares
an isolated Python interpreter with uv, locks dependencies, and reuses the environment.
The cache key includes requirements, Python version selector, OS, architecture and
patches. The application snapshot has a separate content hash, so code changes do
not reinstall model dependencies. User-site packages and the parent's site-packages
are not added to a managed worker. Explicit local plugin paths are supported.

Built-in initial recipes: ACT/MLP (Python 3.10), OpenPI (Python 3.11, Linux/glibc >= 2.31), OpenVLA
(Python 3.10, Linux). OpenPI uses pinned upstream commits and installs its required
Transformers overlay only into its own environment. Other policies must declare
`runtime` or use `--runtime current`; dependency compatibility is never guessed.
Recipes describe dependencies, not a guarantee of GPU training on every driver.
See the validation report for actual checks performed on this branch.

## Paths

```sh
vlastudio train -p act --cache-dir /scratch/vlastudio --model-cache-dir /models/hf -o /results/run1
export VLASTUDIO_CACHE=/scratch/vlastudio
export VLASTUDIO_CONFIG_PATH=/my/configs
export VLASTUDIO_PLUGIN_PATH=/my/project
```

Cache root precedence: `--cache-dir`, `VLASTUDIO_CACHE`, compatibility variable
`VLASTUDIO_CACHE_DIR`, legacy `ILSTD_CACHE`, user settings, then
`~/.cache/vlastudio`. `settings.json` in the platform VLAStudio config directory
may set `cache_dir`; `VLASTUDIO_SETTINGS` can point to a different settings file.
Within it are `envs`, `apps`, `uv`, `python`, `data`, `models`.
`--model-cache-dir` overrides HF_HOME; otherwise existing HF_HOME/TORCH_HOME are respected. OpenPI assets default to
`models/openpi` under the cache root, with `OPENPI_DATA_HOME` respected when set.
Explicit dataset cache paths in task configs retain their existing behavior.
`--data-cache-dir` (or Python `load_dataset(..., cache_dir=...)`) selects a separate
data cache. `VLASTUDIO_DATA_CACHE_DIR` is its environment-variable equivalent.
Explicit data-cache selection overrides task `cache.root` and the default HF
Datasets/LeRobot cache locations; without it existing explicit locations are kept.
Task preprocessing defaults to `data/tasks` within the runtime cache and is still
enabled only by the task's `cache` setting. Local dataset source paths are unchanged.
Checkpoint outputs are not cache files and are never cleaned by the runtime manager.

`--plugin-path` and `--config-path` may be repeated. Installed config files are a
fallback; user config files and search paths take precedence. `--dry-run` resolves
configuration without creating environments. `--offline` only uses an already
prepared environment and sets the Hugging Face/uv offline flags. It cannot prevent
network requests made by arbitrary user code. `vlastudio doctor` lists cached environments.

## External components

No repository edits are required. All component kinds support `type` references:

```yaml
type: my_package.devices:Camera
args:
  name: wrist_camera
```

Supported references:

- Existing dotted path: `my_package.devices.Camera`.
- Module and object: `my_package.devices:Camera`.
- A local file: `/my/project/device.py:Camera` (a policy module can use `policy.py:`).
- Registered plugin: `@device/camera`, `@robot/arm`, `@policy/model`,
  `@dataset/episodes`, `@action_manager/chunks`.

The component must implement the existing interface for the context where it runs.
For example, a robot used by the robot classification helpers subclasses BaseRobot;
a device implements BaseDevice's abstract methods. The generic resolver does not
pretend all objects implement those interfaces. Existing policy modules keep their
load_model, processor, collator and trainer hooks. Dataset constructors and sample
contracts are unchanged. `vlastudio.interfaces` exposes legacy base types lazily;
accessing them requires the corresponding runtime dependencies.

```python
from vlastudio import create, register
register("device", "camera", "my_package.devices:Camera")
device = create({"type": "@device/camera", "args": {"name": "wrist"}})
```

In-process registration is process-local. For CLI/managed workers, use module/file
references or an installed plugin distribution:

```toml
[project.entry-points."vlastudio.device"]
camera = "my_package.devices:Camera"
[project.entry-points."vlastudio.policy"]
model = "my_package.policy"
[project.entry-points."vlastudio.config"]
task = "my_package.configs:task_path"
```

There are also groups `vlastudio.robot`, `vlastudio.action_manager`, and
`vlastudio.dataset`. A config entry point returns a YAML path; `@config/task` can be
passed where a config path is expected. Config providers must be available to the
lightweight launcher. Worker-only plugins belong in the runtime requirements.
Duplicate plugin registrations fail rather than selecting an arbitrary plugin.

## Runtime declarations

A policy config may declare its own dependencies, without needing publication:

```yaml
type: my_package.policy
name: my_model
runtime:
  python: "3.11"
  requirements:
    - "my-policy-plugin==1.0.0"
    - "torch==2.7.1"
```

Or `runtime: ./runtime.yaml` references a manifest relative to that config file.
`--runtime-manifest /path/runtime.yaml` selects a complete task environment explicitly.
Components nested in a task can also declare runtime requirements. They are merged
with the policy requirements and resolved together; incompatible Python versions
are rejected. Adding component requirements to a locked base resolves a new environment while preserving the base pins. Multiple component lockfiles are rejected; supply one complete task lock instead. Requirements for uninstalled robot SDKs,
custom datasets and devices should be declared here as well.

To reproduce exact transitive dependencies on another host of the same platform,
copy `envs/<key>/requirements.lock` to a project and use `runtime.lockfile` (relative
to the manifest/config). It must include PyYAML, platformdirs and filelock.
Built-in platform locks are shipped for Windows/Linux x86-64 ACT/MLP and Linux x86-64 OpenPI/OpenVLA. For other recipes without a supplied lock, first preparation resolves and saves a lock locally;
fresh hosts may resolve newer transitive dependencies. Check the supplied lock into
your project for reproducibility. Use immutable versions/revisions for plugins.

The launcher does not install host drivers or system SDKs. GPU/runtime compatibility
and simulation dependencies still need a matching recipe or a provisioned container.
A single training process cannot load mutually incompatible library versions: use
separate policy/device services when those interfaces can be separated.

An optional `runtime.entrypoint: module:function` replaces the legacy task entry
with a function `(command, argv) -> exit_code`. This is useful for a custom training
stack, and intentionally runs only inside the selected environment. See
`examples/components.py` and `examples/policy.yaml` provide a dependency-free end-to-end example.

## Packaging architecture

All implementations live under `src/vlastudio`: `policy`, `benchmark`, `data_utils`,
`deploy`, `utils`, and built-in `configs`. The wheel uses that same layout without
copying files into an `_legacy` tree. Internal imports use the `vlastudio` namespace.
Root training/evaluation scripts are thin compatibility entrypoints only.
The extension resolver accepts historical config module references; task workers
also enable aliases for old serialized module names. Aliases point to the same
canonical modules, so class identity is preserved. Merely importing `vlastudio`
does not install these global import aliases or import ML libraries.
Git submodules remain optional; profiles/plugins own their installation.

The previous development dependency list is preserved in `requirements-legacy.txt`.
The root uv.lock now covers the lightweight package and its development tools.


## Standalone custom devices and Python calls

```sh
vlastudio device --config /my/devices/camera.yaml
```

The device config contains `type`, `args`, and `runtime`. The worker constructs the
device, calls `start()`, and calls `close()` on exit. Existing BaseDevice subclasses
can be loaded through `vlastudio.interfaces.BaseDevice`; no fixed device list is
imposed. Use installed modules or absolute file paths for reusable device configs.

```python
import vlastudio
code = vlastudio.run("train", ["-p", "my_policy.yaml", "-o", "/results/run"],
                    cache_dir="/scratch/vlastudio")
```

The Python API also exposes `load_dataset`, `load_policy`, `train` and `load_env`.
These return configuration handles and dispatch training/evaluation to workers;
they do not expose tensors in the caller. `train(policy, dataset, config,
output_dir=...)` updates `policy.checkpoint` after success; `bench.evaluate(policy,
output_dir=...)` evaluates saved artifacts. Managed evaluation needs a complete
simulation runtime manifest. See the [Python quick start](../README.md#用-python-组织训练和评估)
and [runnable scripts](../examples). For direct inference use
`vlastudio serve` and the existing communication client interfaces.

From a checkout, the real CPU smoke example is:

```sh
vlastudio train -p examples/policy_mlp.yaml -t examples/task_mlp.yaml -c examples/training_mlp.yaml -o /tmp/my-checkpoints
```


## GPU integration checks

Install the wheel, then run from the source distribution or a checkout:

```sh
python examples/_01_train_and_eval_act_on_aloha.py
```

Each case uses a managed worker with the corresponding policy environment. Tests
assert CUDA execution, finite loss/gradients, an actual optimizer update, and GPU
inference, and record peak allocated memory. They use synthetic inputs: OpenPI
uses randomly initialized weights; OpenVLA uses a reduced architecture and a locally
saved checkpoint. These checks do not establish pretrained 7B checkpoint quality
or task success rates. Run cases sequentially to limit peak GPU memory.

For an installation mirror, set `UV_DEFAULT_INDEX` before invoking the launcher.
`UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` can select existing shared download and
interpreter caches; otherwise both live under the selected VLAStudio cache root.
TorchInductor and Triton compilation caches also default under that root, with
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` overrides respected.
`UV_LOCK_TIMEOUT` defaults to 3600 seconds so another environment can finish a
large CUDA wheel download before the cache lock expires.


OpenVLA's default environment covers its PyTorch model and VLAStudio data path.
Its optional RLDS pipeline is lazy-loaded. For that dataset path, declare the
following requirements in the dataset's runtime (Python 3.10):

```yaml
runtime:
  python: "3.10"
  requirements:
    - tensorflow==2.15.0
    - tf-keras==2.15.0
    - tensorflow-datasets==4.9.9
    - tensorflow-graphics==2021.12.3
    - dlimp @ git+https://github.com/kvablack/dlimp.git@5edaa4691567873d495633f2708982b42edf1972
```
