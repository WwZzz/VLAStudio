"""Lazy access to existing interfaces for extension authors in a runtime environment."""
from .extensions import resolve

_TARGETS = {
    "MetaObs": "vlastudio.benchmark.base.MetaObs",
    "MetaAction": "vlastudio.benchmark.base.MetaAction",
    "MetaPolicy": "vlastudio.benchmark.base.MetaPolicy",
    "BaseRobot": "vlastudio.deploy.robot.base.BaseRobot",
    "BaseDevice": "vlastudio.deploy.base.BaseDevice",
    "BaseActionManager": "vlastudio.deploy.action_manager.base.AbstractActionManager",
}


def __getattr__(name):
    if name not in _TARGETS:
        raise AttributeError(name)
    value = resolve(_TARGETS[name])
    globals()[name] = value
    return value
