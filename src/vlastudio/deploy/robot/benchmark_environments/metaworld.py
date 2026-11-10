"""Compatibility import for existing robot environment configurations."""
from vlastudio.deploy.robot.benchmark_adapters.metaworld import MetaWorldEnvironment as MetaWorldDeploymentEnv

__all__ = ["MetaWorldDeploymentEnv"]
