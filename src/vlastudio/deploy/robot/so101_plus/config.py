#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Adapted for SO101 Plus (wrist_yaw) in ILStudio.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_plus")
@dataclass
class SO101PlusConfig(RobotConfig):
    """Config for single-arm SO101 Plus follower with an extra wrist_yaw joint (ID 7)."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety. Positive scalar applies to all motors, or a dict maps
    # motor names to per-motor limits.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to True for backward compatibility with previous policies/datasets
    use_degrees: bool = False
