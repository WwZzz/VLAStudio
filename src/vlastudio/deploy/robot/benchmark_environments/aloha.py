"""Compatibility import for existing robot environment configurations."""
from vlastudio.deploy.robot.benchmark_adapters.aloha import AlohaEnvironment as AlohaDeploymentEnv

__all__ = ["AlohaDeploymentEnv"]
