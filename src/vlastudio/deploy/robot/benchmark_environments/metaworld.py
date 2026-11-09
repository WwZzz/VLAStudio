"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.metaworld.interactive import MetaWorldInteractiveEnv as MetaWorldDeploymentEnv

__all__ = ["MetaWorldDeploymentEnv"]
