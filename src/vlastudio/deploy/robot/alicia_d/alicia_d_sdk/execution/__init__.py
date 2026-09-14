# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Execution Layer

Provides trajectory execution functionality, including:
- Joint space trajectory execution
- Cartesian space trajectory execution
- Drag teaching and pose recording
"""

from alicia_d_sdk.execution.trajectory_executor import JointTrajectoryExecutor, CartesianTrajectoryExecutor

__all__ = [
    "JointTrajectoryExecutor",
    "CartesianTrajectoryExecutor"
]