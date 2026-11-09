"""Compatibility import for existing deployment configurations.

Interactive environment behavior now lives alongside the benchmark environment.
"""
from vlastudio.benchmark.robotwin.interactive import RoboTwinInteractiveEnv as RoboTwinDeploymentEnv

__all__ = ["RoboTwinDeploymentEnv"]
