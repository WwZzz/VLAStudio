"""Compatibility import for existing robot environment configurations."""
from vlastudio.deploy.robot.benchmark_adapters.rlbench import RLBenchEnvironment as RLBenchDeploymentEnv

__all__ = ["RLBenchDeploymentEnv"]
