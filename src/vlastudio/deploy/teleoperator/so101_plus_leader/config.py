#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_plus_leader")
@dataclass
class SO101PlusLeaderConfig(TeleoperatorConfig):
    # Port to connect to the leader arm
    port: str

    use_degrees: bool = False
