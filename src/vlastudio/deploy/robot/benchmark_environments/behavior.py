"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.behavior1k.interactive import BehaviorInteractiveEnv as BehaviorDeploymentEnv

__all__ = ["BehaviorDeploymentEnv"]
