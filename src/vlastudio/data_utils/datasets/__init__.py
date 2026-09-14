"""
Dataset modules for IL-Studio.

This package contains individual dataset implementations that inherit from the base EpisodicDataset class.
Each dataset is implemented in its own file for better modularity and extensibility.
"""

from .base import EpisodicDataset
from .aloha_sim import AlohaSimDataset
from .aloha_sii import AlohaSIIDataset
from .aloha_sii_v2 import AlohaSIIv2Dataset
from .robomimic_dataset import RobomimicDataset
from .koch_dataset import KochDataset
from .d4rl import D4RLDataset
try:
    from .lerobot_wrapper import WrappedLerobotDataset
except ImportError:
    WrappedLerobotDataset = None
from .rlbench_dataset import RLBenchDataset
# LeRobot standalone wrappers (no lerobot dependency)
from .lerobotv20_wrapper import WrappedLerobotV20Dataset
from .lerobotv21_wrapper import WrappedLerobotV21Dataset
from .lerobotv30_wrapper import WrappedLerobotV30Dataset

__all__ = [
    'EpisodicDataset',
    'AlohaSimDataset', 
    'AlohaSIIDataset',
    'AlohaSIIv2Dataset',
    'RobomimicDataset',
    'KochDataset',
    'D4RLDataset',
    "WrappedLerobotDataset",
    "RLBenchDataset",
    # LeRobot standalone wrappers
    "WrappedLerobotV20Dataset",
    "WrappedLerobotV21Dataset",
    "WrappedLerobotV30Dataset",
]
