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
Hardware Layer

Provides low-level hardware driver functionality, including:
- Serial communication with robot hardware
- Data parsing and protocol handling
- Servo motor control and state management
"""

from alicia_d_sdk.hardware.servo_driver import ServoDriver
from alicia_d_sdk.hardware.serial_comm import SerialComm
from alicia_d_sdk.hardware.data_parser import DataParser, JointState


__all__ = [
    "ServoDriver",
    "SerialComm", 
    "DataParser",
    "JointState",
]