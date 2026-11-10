"""Compatibility import for existing robot environment configurations."""
from vlastudio.deploy.robot.benchmark_adapters.robosuite import RobosuiteEnvironment as RobosuiteDeploymentEnv

__all__ = ["RobosuiteDeploymentEnv"]
