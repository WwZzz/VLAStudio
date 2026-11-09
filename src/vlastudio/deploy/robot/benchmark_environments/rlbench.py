"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.rlbench.interactive import RLBenchInteractiveEnv as RLBenchDeploymentEnv

__all__ = ["RLBenchDeploymentEnv"]
