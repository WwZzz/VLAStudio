"""Align an actuated input source before accepting a fresh absolute joint target."""
import json
import socket
import time
import uuid

import numpy as np
from loguru import logger

from .bridge import PAUSE, RUN, RESET_A, RESET_B, RESET_AND_RESUME, RESETTING


class LeaderGate:
    def __init__(self, box, endpoint):
        self.box = box
        self.endpoint = str(endpoint)
        self.previous = self.token = None
        self.waiting = False
        self.last_frame = 0.

    def send(self, payload):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.settimeout(.1)
            sock.sendto(json.dumps(payload).encode(), self.endpoint)

    def cancel(self):
        self.token = None
        self.waiting = False
        try:
            self.send({})
        except OSError:
            pass

    def pause(self, robot):
        self.cancel()
        robot.bus.cancel_pending_target()
        self.previous = PAUSE
        self.box[0] = self.box[1] = self.box[4] = PAUSE

    def filter(self, robot, message):
        command = self.box[0]
        now = time.monotonic()
        try:
            if command != self.previous:
                self.previous = command
                self.box[4] = command
                if command == PAUSE:
                    self.pause(robot)
                    return
                self.box[1] = RESETTING
                robot.bus.cancel_pending_target()
                resetting = command in (RESET_A, RESET_B, RESET_AND_RESUME)
                if resetting:
                    self.cancel()
                    if not robot.reset_for_teleoperation():
                        raise ValueError("Follower did not confirm home")
                if self.box[0] != command:
                    return
                self.token = uuid.uuid4().hex
                target = np.zeros(7) if resetting else robot.get_observation()["qpos"]
                self.send(dict(target=target.tolist(), token=self.token))
                self.waiting = True
                self.last_frame = time.monotonic()
                return
            if command == PAUSE:
                return
            if now-self.last_frame > (25 if self.waiting else .35):
                raise ValueError("Leader alignment or live feedback timed out")
            if not message or message.get("align_token") != self.token:
                return
            if not 0 <= now-message["__timestamp__"] < .25:
                return
            if self.waiting:
                delta = abs(message["action"]-robot.get_observation()["qpos"])
                if np.any(delta[:6] > np.deg2rad(4)) or delta[6] > .06:
                    raise ValueError("Leader and follower differ after releasing torque")
                if self.box[0] != command:
                    return
                if command in (RESET_A, RESET_B):
                    self.pause(robot)
                    return
                self.previous = RUN
                self.box[0] = self.box[1] = self.box[4] = RUN
                self.waiting = False
                self.last_frame = now
                return
            self.last_frame = now
            return message if self.box[0] == RUN else None
        except (ValueError, OSError) as exc:
            logger.error("Leader handoff paused: {}", exc)
            self.pause(robot)
