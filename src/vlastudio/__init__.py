"""VLAStudio's lightweight public API. Importing it never installs dependencies."""
from .extensions import register, resolve, create
from .api import (load_dataset, load_policy, load_env, connect_policy, train, serve,
                  show_camera,
                  Dataset, Policy, RemotePolicy, Environment, TrainingResult,
                  EvaluationResult, TaskError)

__version__ = "0.2.0.dev0"
__all__ = ["register", "resolve", "create", "load_dataset", "load_policy", "load_env",
           "connect_policy", "train", "serve", "show_camera",
           "Dataset", "Policy", "RemotePolicy", "Environment", "TrainingResult",
           "EvaluationResult", "TaskError"]


def run(command, args=(), **options):
    """Run a task using the CLI's managed environments; return its exit code.

    Example: run("train", ["-p", "my_policy.yaml", "-o", "/checkpoints/run"])
    """
    from .cli import main
    argv = [command, *map(str, args)]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            values = value if isinstance(value, (tuple, list)) else [value]
            for item in values:
                argv.extend([flag, str(item)])
    return main(argv)

__all__.append("run")
