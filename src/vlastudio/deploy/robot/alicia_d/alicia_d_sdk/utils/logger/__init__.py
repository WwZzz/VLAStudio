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

from .beauty_logger import *
from datetime import datetime
from .beauty_logger import hex_print


_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = BeautyLogger(log_dir="./logs", log_name=f"alicia_d_sdk_{_log_timestamp}.log", verbose=True, min_level=LogLevel.INFO)
