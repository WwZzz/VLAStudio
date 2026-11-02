"""Lightweight entry points for persistent simulation and robot execution."""
from copy import deepcopy
from pathlib import Path


def _configuration(config):
    if isinstance(config, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise TypeError('Execution configuration must be a mapping or YAML path')
    return deepcopy(config)


def open_execution(config, *, output_dir, cancel_event=None):
    """Open a session; update_context supplies low-rate policy conditions.

    Observations, inference and action publication run in owned subprocesses.
    Close the returned session or use it as a context manager.
    """
    from .deploy.execution_session import ExecutionSession
    return ExecutionSession(_configuration(config), output_dir=output_dir,
                            cancel_event=cancel_event)


def connect_target(config, *, startup_timeout_s=60):
    """Keep a configured simulation or robot connected across execution sessions."""
    from .deploy.execution_session.target import ConnectedTarget
    return ConnectedTarget(_configuration(config), startup_timeout_s=startup_timeout_s)
