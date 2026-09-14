"""Lazy access to existing interfaces for extension authors in a runtime environment."""
import sys
from .paths import legacy_root
from .extensions import resolve

_TARGETS = {
    "MetaObs": "benchmark.base.MetaObs",
    "MetaAction": "benchmark.base.MetaAction",
    "MetaPolicy": "benchmark.base.MetaPolicy",
    "BaseRobot": "deploy.robot.base.BaseRobot",
    "BaseDevice": "deploy.base.BaseDevice",
    "BaseActionManager": "deploy.action_manager.base.AbstractActionManager",
}


def __getattr__(name):
    if name not in _TARGETS:
        raise AttributeError(name)
    root = str(legacy_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    value = resolve(_TARGETS[name])
    globals()[name] = value
    return value
