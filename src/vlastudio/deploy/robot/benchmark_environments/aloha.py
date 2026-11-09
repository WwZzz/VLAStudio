"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.aloha.interactive import AlohaInteractiveEnv as AlohaDeploymentEnv

__all__ = ["AlohaDeploymentEnv"]
