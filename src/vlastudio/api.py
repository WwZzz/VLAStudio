"""Configuration handles for training and evaluation in isolated workers.

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


@dataclass
class Policy:
    """A policy configuration and, after training, its saved checkpoint location."""
    config_path: Path
    checkpoint: Path | None = None
    runtime_options: dict = field(default_factory=dict, repr=False)


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

        A complete simulation runtime_manifest or runtime='current' is required.
        Custom runtime entrypoints may implement their own evaluation contract.
        """
        if not isinstance(policy, Policy) or policy.checkpoint is None:
            raise ValueError("Evaluation requires a Policy with a trained or supplied checkpoint")
        if num_rollout <= 0 or batch_size < 0:
            raise ValueError("num_rollout must be positive and batch_size nonnegative")
        reserved = {'o', 'output_dir', 'm', 'model_name_or_path', 'e', 'env',
                    'n', 'num_rollout', 'bs', 'batch_size', 'device', 'am', 'action_manager'}
        if reserved.intersection(overrides or {}):
            raise ValueError("Use evaluate() parameters for output, checkpoint, environment and rollout options")
        options = {**{k: v for k, v in policy.runtime_options.items() if k != 'runtime_manifest'},
                   **self.runtime_options, **_normalize_options(runtime_options)}
        # A training-only environment is not a complete simulator environment.
        if options.get("runtime", "managed") == "managed" and not options.get("runtime_manifest"):
            raise ValueError("Provide load_env(..., runtime_manifest='simulation.yaml') or runtime='current'")
        output = _path(output_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty evaluation output directory to avoid mixing old metrics")
        args = ["-m", str(policy.checkpoint), "-e", str(self.config_path),
                "-o", str(output), "-n", str(num_rollout), "-bs", str(batch_size), "--device", device]
        if action_manager is not None:
            args.extend(["-am", str(_config(action_manager, "action_manager"))])
        args.extend(_overrides(overrides))
        _run("eval-sim", args, options)
        metrics = {str(p.relative_to(output)): json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(output.rglob("*.json"))}
        return EvaluationResult(output, metrics)


def load_dataset(config):
    """Describe a dataset/task YAML or built-in task name; defer data I/O to train."""
    return Dataset(_config(config, "task"))


def load_policy(config, *, checkpoint=None, **runtime_options):
    """Describe a policy using the same selector as train.py --policy.

    checkpoint is for evaluation. To initialize training weights use the policy
    config's pretrained_weight_path; to resume use training overrides.
    """
    return Policy(_config(config, "policy"), _path(checkpoint) if checkpoint else None,
                  _normalize_options(runtime_options))


def load_env(config, **runtime_options):
    """Describe a simulation benchmark and its complete dependency environment."""
    return Environment(_config(config, "env"), _normalize_options(runtime_options))


def _normalize_options(options):
    allowed = {"runtime", "runtime_manifest", "cache_dir", "model_cache_dir",
               "plugin_path", "config_path", "offline"}
    unknown = options.keys() - allowed
    if unknown:
        raise TypeError(f"Unknown runtime options: {sorted(unknown)}")
    result = dict(options)
    for key in ("runtime_manifest", "cache_dir", "model_cache_dir"):
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
    Runtime dependencies are installed in a cached worker, never in this process.
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
    _run("train", args, options)
    if not output.is_dir():
        raise TaskError(f"Training exited successfully but did not create {output}")
    policy.checkpoint = output
    # Carry cache settings into evaluation, but not a training-only entrypoint.
    policy.runtime_options.update({k: v for k, v in options.items() if k != "runtime_manifest"})
    return TrainingResult(output, policy)
