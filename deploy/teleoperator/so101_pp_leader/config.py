#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_pp_leader")
@dataclass
class SO101PPLeaderConfig(TeleoperatorConfig):
    port: str
    use_degrees: bool = False
