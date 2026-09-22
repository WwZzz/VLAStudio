# VLAStudio

An extensible Python toolkit for robot policy training, evaluation, and deployment.
Use your own policies, datasets, robots, devices, action managers, and configurations
without changing VLAStudio's source code.

Organize experiments as Python scripts or use the `vlastudio` CLI. Reusable Python
environments keep policy and simulator dependencies separate. ACT, MLP, DP, and
basic simulators share the base profile; SmolVLA and OpenPI use separate profiles.

## Repository layout

```text
VLAStudio/
├── src/vlastudio/
│   ├── policy/          # Policies and trainers
│   ├── benchmark/       # Simulation and evaluation
│   ├── data_utils/      # Datasets, processing, and caching
│   ├── deploy/          # Robots, devices, action managers, and communication
│   ├── configs/         # Built-in configurations and aliases
│   ├── utils/
│   ├── api.py           # Public Python API
│   ├── cli.py
│   └── entrypoints/     # Training, evaluation, serving, and collection implementations
├── examples/
├── docs/
├── pyproject.toml
└── README.md
```

Source checkouts and installed packages use the same layout: `vlastudio.policy`,
`vlastudio.benchmark`, and other modules. Legacy configuration references such as
`policy.act` still resolve; new built-in configurations use full package paths.
The former root scripts have been removed. Use the CLI or Python API instead.

## Installation

The lightweight package requires Python 3.10 or newer:

```bash
git clone --branch codex/package-runtime-isolation https://github.com/WwZzz/VLAStudio.git
cd VLAStudio
python -m pip install -e .
# Regular installation from a checkout:
python -m pip install .
# Or install a built wheel:
python -m pip install /path/to/vlastudio-0.2.0.dev0-py3-none-any.whl
```

This development version is `0.2.0.dev0`. It has not been published to PyPI;
install from this branch or a built wheel to obtain it.

The package installs the lightweight launcher. Managed CLI execution prepares and
reuses dependencies declared by the selected runtime. Python API scripts use their
current Python environment and do not automatically install dependencies.
GPU drivers, system libraries, simulator assets, and hardware SDK system components
must be available on the host or in the container.

`PyYAML`, `platformdirs`, `filelock`, and `loguru` are shared runtime dependencies.
They are installed in the package and every managed environment. Component profiles
declare additional requirements. Complete lockfiles must also contain these shared
dependencies; preparation verifies them before starting a task.

## Environment management

Choose components without having to look up package extras:

```bash
vlastudio env create
vlastudio env create --policy smolvla -n smol_work
vlastudio env create --env robotwin -n robotwin_work
vlastudio env list
vlastudio env run -n smol_work -- python my_training_script.py
```

Base is automatically recorded from the launcher's Python environment when the
environment manager is first used. Registration does not install training dependencies.
`env create` prepares a managed base when only the automatic default exists;
an explicitly registered base is reused. `env install` adds dependencies to the
current environment and records it as base unless a name is supplied:

```bash
vlastudio env install
vlastudio env install -n act_work --policy act --env aloha_sim
```

An existing `-n/--name` selects that registered interpreter for installation.
A new name records the current interpreter without creating a venv. Policy and
simulator requirements can be combined for installation; incompatible constraints
produce a resolver error. Use separate processes for incompatible components.

New explicitly named environments have independent directories at
`~/.cache/vlastudio/envs/named-<name>`. Repeated creation reuses a registered
environment and preserves manual modifications. Unnamed environments reuse a
dependency hash. Existing registrations retain their original paths.

Package sources inherit the process environment and installation tool configuration.
VLAStudio does not select a mirror by default. Override the source for one invocation:

```bash
vlastudio env create --policy smolvla -n smol_work -i http://nexus.sii.shaipower.online/repository/pypi/simple/
vlastudio env install -n base --index-url https://pypi.org/simple/
```

`prepare` and `run` also accept `-i/--index-url`. The URL is not saved globally
and does not change the dependency hash. Pass a plain URL.

`env run` requires no shell activation. Switching the current terminal through
`env activate` or `env deactivate` requires shell integration; see
[environment management](docs/environments.md). Activation defaults to base;
deactivation returns to base. `env path` prints an interpreter without installing it.

| Policy profile | Python and main libraries | Platform |
| --- | --- | --- |
| Base: ACT / MLP / DP | Python 3.10, Torch 2.4 | Subject to component dependencies |
| SmolVLA | Python 3.10, Torch 2.7.1 | Subject to LeRobot dependencies |
| OpenPI | Python 3.11, Torch 2.7.1 | Linux, glibc >= 2.31 |
| OpenVLA | Python 3.10, Torch 2.4 | Linux |
| Custom policy | Declares `runtime` in its configuration | Subject to declared dependencies |

Policy selectors accept configuration names or YAML paths. OpenPI's built-in
configuration is `pi0`; environment commands also accept the `openpi` alias.
The `dp` environment alias selects `diffusion_policy`.

## Python training and evaluation

Each example `run.sh` creates an environment, activates it, and runs the script.
For ACT and ALOHA:

```bash
examples/01_train_and_eval_act_on_aloha/run.sh
```

For multitask SmolVLA on LIBERO-Object:

```bash
examples/04_multitask_policy/run.sh
```

For LoRA-finetuning π0.5 on Tabletop-Sim:

```bash
examples/05_finetune_vla_pi05/run.sh
```

If a command fails, see that example's README. You can also run a script in a
selected environment without `run.sh`:

```bash
vlastudio env run --policy act -- python examples/01_train_and_eval_act_on_aloha/train_and_evaluate.py
```

The minimal ACT example uses five lines:

```python
import vlastudio as vla
dataset = vla.load_dataset("sim_transfer_cube_scripted")
policy = vla.load_policy("act")
vla.train(policy, dataset, "default", output_dir="checkpoints/act_aloha")
vla.load_env("aloha_transfer").evaluate(policy, output_dir="results/act_aloha")
```

Built-in datasets and policies accept aliases; custom configurations accept YAML paths:

```python
dataset = vla.load_dataset("rlbench.reach_target", cache_dir="/datasets/vla-cache")
dataset = vla.load_dataset("/my/configs/task.yaml", cache_dir="/datasets/custom-cache")
policy = vla.load_policy("/my/configs/policy.yaml")
# Defaults are sim_transfer_cube_scripted and act:
dataset = vla.load_dataset()
policy = vla.load_policy()
```

The dotted alias `rlbench.reach_target` maps to the packaged
`configs/task/rlbench/reach_target.yaml`. Other names follow the same rule.
Aliases select configurations; data paths and remote IDs remain defined by those
configurations. With an empty cache, `sim_transfer_cube_scripted` downloads the
public dataset specified by its built-in configuration.

Use explicit paths for custom experiments:

```python
import vlastudio as vla

dataset = vla.load_dataset("/my/configs/task.yaml")
policy = vla.load_policy("/my/configs/policy.yaml", cache_dir="/scratch/vlastudio")
result = vla.train(
    policy, dataset, "/my/configs/training.yaml",
    output_dir="/my/checkpoints/run1",
    overrides={"training.max_steps": 1000},
)
print(result.checkpoint)
print(policy.checkpoint)  # Updated after successful training.

bench = vla.load_env("/my/configs/env.yaml")
evaluation = bench.evaluate(policy, output_dir="/my/results/run1", num_rollout=10)
print(evaluation.metrics)
```

Replace placeholder paths with actual configurations. The
[remote inference example](examples/02_remote_inference/README.md) separates
`vla.serve(...)` and simulation evaluation into independent processes and environments.
It supports TCP, HTTP(S), and shared memory.

### Handles and return values

`load_policy`, `load_dataset`, and `load_env` return lightweight configuration handles:
`vla.Policy`, `vla.Dataset`, and `vla.Environment`. Actual objects are created in a
worker process using the current Python environment. Missing dependencies raise
an error instead of modifying that environment.

The policy handle is not a `torch.nn.Module` and does not expose `.parameters()`.
The dataset handle is not an iterable PyTorch Dataset. `load_dataset` accepts a
task configuration containing dataset entries, dimensions, and normalization
settings, rather than only a data file path. These APIs organize experiments.

`vla.train` waits for completion and reuses the processor, collator, data cache,
and policy-specific Trainer. Success returns `TrainingResult(checkpoint, policy)`
and updates `policy.checkpoint`. Failure raises `vla.TaskError`, preserves the
previous checkpoint, and prints worker logs to the terminal. For direct tensor
access or a custom optimization loop, use a runtime entrypoint described below.

## Configuration, overrides, and checkpoints

| Configuration | Contents | Documentation |
| --- | --- | --- |
| policy | Implementation, architecture, initial weights, runtime | [configs](src/vlastudio/configs/README.md) |
| task | Datasets, dimensions, normalization, cache | [data utilities](src/vlastudio/data_utils/README.md) |
| training | Batch size, steps, learning rate, saving | [training](src/vlastudio/configs/training/README.md) |
| env | Simulator tasks, cameras, control settings | [benchmarks](src/vlastudio/benchmark) |
| action manager | Action chunks, synchronization, execution | [action managers](src/vlastudio/configs/action_manager/README.md) |

The `load_*` functions accept configuration names, YAML paths, and `@config/name`.
Configuration paths are resolved when called. Relative data and Python file paths
inside configurations retain their working-directory semantics. Use absolute paths
for reusable scripts. `VLASTUDIO_CONFIG_PATH` adds configuration search directories.

Python overrides correspond to CLI dotted overrides:

```python
vla.train(policy, dataset, "default", output_dir="/my/checkpoints/run1",
          overrides={"training.max_steps": 2000,
                     "training.per_device_train_batch_size": 8,
                     "policy.args.chunk_size": 16})
```

Override values are strings, numbers, or booleans. Put lists and nested structures
in YAML. Set `output_dir` through the function argument. Resuming follows the
Trainer's `resume_from_checkpoint` rules: configure it in the training configuration
and retain the output directory and its `checkpoint-*` subdirectories. The training
entrypoint clears resume settings when no checkpoint subdirectory is available.

Bind a checkpoint for evaluation:

```python
policy = vla.load_policy("act", checkpoint="/my/checkpoints/run1")
```

`checkpoint=` selects evaluation weights; it does not automatically initialize
training. Training initialization still uses fields such as `pretrained_weight_path`.

## Evaluation

`bench.evaluate(policy, ...)` uses the packaged simulation evaluation entrypoint.
It supports `num_rollout`, `batch_size`, `device`, `action_manager`, and environment
overrides. The default `batch_size=0` runs sequentially. Real robots use the
`vlastudio infer` deployment workflow.

A managed evaluation runtime must include both policy and simulator requirements.
Prepare a complete runtime manifest using the benchmark documentation; a training
lockfile alone may not contain simulator dependencies. With dependencies already
installed in the current environment:

```python
bench = vla.load_env("aloha_transfer", runtime="current")
evaluation = bench.evaluate(policy, output_dir="./results/new-run", device="cuda")
```

For incompatible dependencies, connect to a separate policy process:

```python
remote_policy = vla.connect_policy("host:5000")
vla.load_env("aloha_transfer").evaluate(remote_policy, output_dir="results/remote")
```

Evaluation output directories must be absent or empty to avoid mixing metrics.
The returned `output_dir` is absolute; `metrics` is keyed by relative JSON filenames.
Videos remain in the output directory. Cameras, action spaces, and normalization
must match the training configuration.

## Cache and output paths

The default cache is `~/.cache/vlastudio`. Override it with an environment variable
or an explicit API/CLI argument:

```bash
export VLASTUDIO_CACHE=/path/to/persistent-cache
vlastudio doctor
```

```python
policy = vla.load_policy("act", cache_dir="/scratch/vlastudio",
                         model_cache_dir="/datasets/huggingface-cache")
```

Cache-root precedence is: explicit argument, `VLASTUDIO_CACHE`, compatibility
variable `VLASTUDIO_CACHE_DIR`, legacy `ILSTD_CACHE`, user settings,
then `~/.cache/vlastudio`.

| Location | Contents |
| --- | --- |
| `cache/envs` | Isolated Python environments and dependency locks |
| `cache/apps` | Application snapshots identified by source content |
| `cache/data` | Default dataset cache; explicit task paths are preserved |
| `cache/models` | Default model resources |
| `cache/uv`, `cache/python` | Downloads and interpreter caches |
| `output_dir` | User-selected checkpoints or evaluation outputs |

`load_dataset(..., cache_dir=...)` controls dataset caching only. It takes precedence
over training `data_cache_dir` and the global default, without moving source data or
changing checkpoints or dependency environments. Subdirectories include
`huggingface/`, `lerobot/`, `normalize/`, and enabled preprocessing caches under
`tasks/`. The configuration still controls the source `root`. A cache path does
not enable preprocessing; the task's `cache` setting must enable it.

Without an explicit dataset cache, the default is `<cache_dir>/data`.
Existing HF Datasets and LeRobot environment variables and task `cache.root`
are preserved. An explicit dataset cache or `VLASTUDIO_DATA_CACHE_DIR` overrides
those cache locations. The equivalent CLI option is `--data-cache-dir`.
Policy `cache_dir` controls the runtime cache root; `model_cache_dir` sets
Hugging Face's `HF_HOME`. Existing `TORCH_HOME`, `OPENPI_DATA_HOME`, and
`UV_CACHE_DIR` values are respected.

Use persistent storage with enough capacity. `/tmp` may disappear when an instance
is recreated. See [runtime documentation](docs/package_runtime.md) for details.
`offline=True` requires prepared environments and model resources. It sets uv and
Hugging Face offline flags but cannot constrain arbitrary network access in custom code.

## Custom policies, datasets, robots, devices, and action managers

Components retain their interface contracts and can come from installed plugins
or your own files. A task dataset entry can refer to a custom class:

```yaml
datasets:
  - name: custom
    type: /my/project/dataset.py:MyDataset
    args:
      root: /datasets/my-episodes
# Also configure task metadata for dimensions and normalization.
```

A custom policy configuration can declare dependencies:

```yaml
name: my_policy
type: my_package.policy
args:
  pretrained_weight_path: /models/initial-weights
runtime:
  python: "3.11"
  requirements:
    - my-policy-plugin==1.0.0
```

Replace placeholder package names with installable dependencies. Policy modules
provide loading, processor, collator, and Trainer hooks. Datasets follow the sample
contract; robots, devices, and action managers follow their respective interfaces.
See [external component interfaces](docs/package_runtime.md#external-components).

Supported references include `module.Class`, `module:Class`, `/absolute/file.py:Class`,
and package entry points such as `@policy/name` and `@dataset/name`.
`vla.register()` affects the current process only. Across environments, use module
or file references or installed package entry points, and include plugins in
the runtime requirements.

With dependencies installed, construct an actual component directly:

```python
device = vla.create({"type": "my_package.camera:Camera", "args": {"name": "wrist"}},
                    kind="device")
```

`create` does not prepare an environment. To run a standalone device, use
`vlastudio device --config /my/device.yaml`; device configurations may declare a runtime.

### Custom Python training loops

Select a full runtime manifest through `runtime_manifest=` or `--runtime-manifest`:

```yaml
python: "3.11"
requirements:
  - my-training-package==1.0.0
entrypoint: /my/project/training.py:run
```

The function `run(command, argv)` executes in the selected worker environment.
Return `0` on success; raise an exception or return a nonzero value on failure.
The function can instantiate actual datasets and policies and control its optimizer.
When called through `vla.train`, save evaluable outputs to the directory passed with
`-o`. Standard evaluation requires standard policy metadata and weights. A custom
evaluator can define its own checkpoint format and emit metric JSON files.

Dataset and policy dependencies in one process must be compatible. Isolation cannot
load incompatible Torch versions into the same process. Use separate environments
across experiments and policy servers for incompatible online components.

## CLI workflows

```bash
vlastudio train -p act -t /my/task.yaml -c /my/training.yaml -o /my/checkpoints/run1
vlastudio train --policy /my/policy.yaml --policy.args.chunk_size 16
vlastudio serve -m /my/checkpoints/run1
vlastudio evaluate -m /my/checkpoints/run1 -e /my/env.yaml --runtime-manifest /my/simulation-runtime.yaml
# Use an already configured Python environment:
vlastudio train -p act --runtime current
```

Commands retain the policy/task/training/output flag semantics of the original
entrypoints. Policy selectors remain configuration names or paths.
The [legacy overview](README.legacy.md) retains demonstrations and historical setup
context, with commands updated to the CLI. Legacy complete dependencies are in
`requirements-legacy.txt`; root `uv sync` installs only the lightweight package.

## Troubleshooting

- Missing runtime declaration: add `runtime` or a complete manifest, or use a configured current environment.
- Slow first installation: check download logs, cache capacity, and networking; matching environments are reused.
- Long Windows paths: choose a short cache path such as `C:/vla-cache`.
- Missing external module: add its package to runtime requirements, or use an absolute file reference or plugin path.
- Missing simulation dependency: prepare the benchmark's evaluation requirements.
- Native build failures: the host may need compilers, Python development headers, or simulator system libraries.
- Direct tensor or optimizer access: use a custom runtime entrypoint rather than a configuration handle.
