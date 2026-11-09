"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.libero.interactive import LiberoInteractiveEnv as LiberoDeploymentEnv

__all__ = ["LiberoDeploymentEnv"]
