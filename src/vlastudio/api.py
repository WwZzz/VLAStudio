"""Configuration handles for training and evaluation workers.

These are deliberately not torch modules or iterable datasets. They can be
created in a lightweight Python process without importing policy dependencies.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

from .configuration import resolve_config


def _path(value):
    return Path(value).expanduser().resolve()


def _config(value, category):
    return resolve_config(str(value), category)


def _overrides(values):
    result = []
    for key, value in (values or {}).items():
        if not isinstance(key, str) or not key or key.startswith('-'):
            raise ValueError("Override keys must be argument names without leading dashes")
        if isinstance(value, (dict, list, tuple)) or value is None:
            raise TypeError("Overrides must be scalar; put structured values in YAML")
        result.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return result


class TaskError(RuntimeError):
    """A worker failed; logs are streamed to the calling process's terminal."""


def _run(command, args, options):
    from . import run
    code = run(command, args, **options)
    if code:
        raise TaskError(f"{command} failed with exit code {code}; see worker output above")


@dataclass(frozen=True)
class Dataset:
    """A task configuration, including dataset mixtures and normalization metadata."""
    config_path: Path
    cache_dir: Path | None = None


@dataclass
class Policy:
    """A policy configuration and, after training, its saved checkpoint location."""
    config_path: Path
    checkpoint: Path | None = None
    runtime_options: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class RemotePolicy:
    """A policy server address consumed by an evaluation worker."""
    address: str


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    policy: Policy


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics: dict


@dataclass
class Environment:
    """Simulation evaluation configuration; construction does not start a simulator."""
    config_path: Path
    runtime_options: dict = field(default_factory=dict, repr=False)

    def evaluate(self, policy, *, output_dir, num_rollout=4, batch_size=0,
                 device="cuda", action_manager=None, overrides=None, **runtime_options):
        """Evaluate a saved policy using the existing eval_sim.py pipeline.

        Built-in environments declare their simulator dependencies. A custom
        environment may supply runtime_manifest or use runtime='current'.
        """
        if isinstance(policy, str):
            policy = connect_policy(policy)
        if isinstance(policy, RemotePolicy):
            model = policy.address
            policy_options = {}
        elif isinstance(policy, Policy) and policy.checkpoint is not None:
            model = str(policy.checkpoint)
            policy_options = {k: v for k, v in policy.runtime_options.items()
                              if k != 'runtime_manifest'}
        else:
            raise ValueError("Evaluation requires a Policy checkpoint, RemotePolicy, or server address")
        if num_rollout <= 0 or batch_size < 0:
            raise ValueError("num_rollout must be positive and batch_size nonnegative")
        reserved = {'o', 'output_dir', 'm', 'model_name_or_path', 'e', 'env',
                    'n', 'num_rollout', 'bs', 'batch_size', 'device', 'am', 'action_manager'}
        if reserved.intersection(overrides or {}):
            raise ValueError("Use evaluate() parameters for output, checkpoint, environment and rollout options")
        options = {**policy_options, **self.runtime_options,
                   **_normalize_options(runtime_options)}
        output = _path(output_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty evaluation output directory to avoid mixing old metrics")
        args = ["-m", model, "-e", str(self.config_path),
                "-o", str(output), "-n", str(num_rollout), "-bs", str(batch_size), "--device", device]
        if action_manager is not None:
            args.extend(["-am", str(_config(action_manager, "action_manager"))])
        args.extend(_overrides(overrides))
        _run("eval-sim", args, options)
        metrics = {str(p.relative_to(output)): json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(output.rglob("*.json"))}
        return EvaluationResult(output, metrics)


def load_dataset(config="sim_transfer_cube_scripted", *, cache_dir=None):
    """Select a task alias or custom YAML, with an optional dataset-only cache.

    Examples: 'sim_transfer_cube_scripted', 'rlbench.reach_target', '/my/task.yaml'.
    None defers to the runtime's default data cache; no directories are created here.
    """
    return Dataset(_config(config, "task"), _path(cache_dir) if cache_dir is not None else None)


def load_policy(config="act", *, checkpoint=None, **runtime_options):
    """Describe a policy using the same selector as train.py --policy.

    checkpoint is for evaluation. To initialize training weights use the policy
    config's pretrained_weight_path; to resume use training overrides.
    """
    options = _normalize_options(runtime_options)
    options.setdefault("runtime", "managed" if options.get("runtime_manifest") else "current")
    return Policy(_config(config, "policy"), _path(checkpoint) if checkpoint else None, options)


def connect_policy(address):
    """Create a remote policy handle for TCP, HTTP(S), or shared memory."""
    from .deploy.comm import is_server_address
    value = str(address)
    if not is_server_address(value):
        raise ValueError(
            "Policy server address must be host:port, http(s)://host:port, or shm://name"
        )
    return RemotePolicy(value)


def load_env(config, **runtime_options):
    """Describe a simulation benchmark that runs in the current Python environment."""
    options = _normalize_options(runtime_options)
    options.setdefault("runtime", "managed" if options.get("runtime_manifest") else "current")
    return Environment(_config(config, "env"), options)


def serve(policy, *, address="0.0.0.0", port=None, device="cuda",
          dataset_id="", chunk_size=-1, **runtime_options):
    """Serve a local policy checkpoint until interrupted.

    address accepts a TCP bind host, host:port, http(s)://host:port, or shm://name.
    """
    if not isinstance(policy, Policy) or policy.checkpoint is None:
        raise ValueError("serve() requires a Policy with a supplied or trained checkpoint")
    from .deploy.comm import is_http_address, is_server_address, is_shm_address, parse_server_address
    bind = str(address)
    if is_server_address(bind) and not is_http_address(bind) and not is_shm_address(bind):
        bind, address_port = parse_server_address(bind)
        if port is not None and port != address_port:
            raise ValueError("port conflicts with the port included in address")
        port = address_port
    args = ["-m", str(policy.checkpoint), "--host", bind, "--device", str(device),
            "--dataset_id", str(dataset_id), "--chunk_size", str(chunk_size)]
    if port is not None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        args.extend(["--port", str(port)])
    options = {**policy.runtime_options, **_normalize_options(runtime_options)}
    _run("serve", args, options)


def _normalize_options(options):
    allowed = {"runtime", "runtime_manifest", "cache_dir", "model_cache_dir",
               "plugin_path", "config_path", "offline", "data_cache_dir"}
    unknown = options.keys() - allowed
    if unknown:
        raise TypeError(f"Unknown runtime options: {sorted(unknown)}")
    result = dict(options)
    for key in ("runtime_manifest", "cache_dir", "model_cache_dir", "data_cache_dir"):
        if result.get(key) is not None:
            result[key] = str(_path(result[key]))
    for key in ("plugin_path", "config_path"):
        if key in result:
            paths = result[key] if isinstance(result[key], (list, tuple)) else [result[key]]
            result[key] = [str(_path(p)) for p in paths]
    return result


def train(policy, dataset, training_config_path="default", *, output_dir,
          overrides: Mapping | None = None, **runtime_options):
    """Train synchronously, update policy.checkpoint on success and return artifacts.

    Uses the original policy-specific processors, collators, cache and trainer.
    Python API calls use the current environment by default. Pass runtime="managed"
    explicitly to opt into the CLI-style environment manager.
    """
    if not isinstance(policy, Policy) or not isinstance(dataset, Dataset):
        raise TypeError("Use load_policy() and load_dataset() handles with managed train()")
    output = _path(output_dir)
    reserved = {"policy", "task", "training_config", "output_dir", "p", "t", "c", "o"}
    if reserved.intersection(overrides or {}):
        raise ValueError("Use function parameters for policy, task, training_config and output_dir")
    args = ["-p", str(policy.config_path), "-t", str(dataset.config_path),
            "-c", str(_config(training_config_path, "training")), "-o", str(output),
            *_overrides(overrides)]
    options = {**policy.runtime_options, **_normalize_options(runtime_options)}
    if dataset.cache_dir is not None:
        options["data_cache_dir"] = str(dataset.cache_dir)
    _run("train", args, options)
    if not output.is_dir():
        raise TaskError(f"Training exited successfully but did not create {output}")
    policy.checkpoint = output
    # Carry cache settings into evaluation, but not a training-only entrypoint.
    policy.runtime_options.update({k: v for k, v in options.items() if k != "runtime_manifest"})
    return TrainingResult(output, policy)
