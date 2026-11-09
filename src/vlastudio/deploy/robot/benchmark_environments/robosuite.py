"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.robomimic.interactive import RobosuiteInteractiveEnv as RobosuiteDeploymentEnv

__all__ = ["RobosuiteDeploymentEnv"]
