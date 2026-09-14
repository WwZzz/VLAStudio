"""
Utils package initialization.
Automatically configures logging when utils is imported.
"""

# Configure logging on package import
from .logger import logger

# Exploration strategies
from .exploration import (
    BaseExploration,
    NoExploration,
    GaussianNoise,
    OUNoise,
    CustomNoise,
    RandomExploration,
    ExplorationScheduler,
)

__all__ = [
    'logger',
    # Exploration
    'BaseExploration',
    'NoExploration', 
    'GaussianNoise',
    'OUNoise',
    'CustomNoise',
    'RandomExploration',
    'ExplorationScheduler',
]

