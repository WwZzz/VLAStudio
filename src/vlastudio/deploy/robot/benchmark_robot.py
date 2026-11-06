"""Expose a configured MetaEnv through the normal deployment device interface."""
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
import time

import numpy as np

from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.utils.extensions import resolve


class BenchmarkRobot(BaseRobot):
    supports_reset = True

    def __init__(self, name, environment, action, observation, task=None,
                 on_success="continue", on_termination="pause", max_steps=None,
                 control_shm_name=None, fps=30, max_size_mb=64, input_mode="ee_delta", **kwargs):
        super().__init__(name, max_size_mb, fps, control_shm_name)
        self.environment_config = environment
        self.action_config = action
        self.observation_config = observation
        self.task = task
        self.input_mode = input_mode
        self.on_success = on_success
        self.on_termination = on_termination
        if on_success not in ("continue", "pause") or on_termination not in ("pause", "reset"):
            raise ValueError("on_success: continue/pause; on_termination: pause/reset")
        self.max_steps = max_steps
        if fps <= 0 or input_mode not in ('native', 'ee_delta'):
            raise ValueError('fps must be positive and input_mode must be native or ee_delta')
        self.env = None
        self.pending = None
        self.cached = None
        self.blocked = False
        self.steps = 0
        self.episode = 0
        self.success = False
        adapter_type = action.get('adapter', 'vlastudio.deploy.robot.benchmark_adapters.CartesianAdapter')
        self.adapter = resolve(adapter_type)(action)
        self.arms = self.adapter.arms
        self.width = self.adapter.width
        self.native_type = self.adapter.native_type
        self.space = self.adapter.space
        self.last_action = None

    def connect(self):
        if self.env is None:
            config = dict(self.environment_config.get("args", {}))
            if self.task:
                config["task"] = self.task
            factory = resolve(self.environment_config["type"])
            self.env = factory(SimpleNamespace(**config))
            self.reset()
        return True

    def _cache(self, obs, reward=0, terminated=False, truncated=False, success=False):
        obs = asdict(obs) if is_dataclass(obs) else obs
        self.raw = obs
        data = {}
        for destination, source in self.observation_config.get("fields", {}).items():
            value = obs.get(source)
            if value is None:
                raise ValueError(f"Required observation {source!r} is missing")
            data[destination] = np.asarray(value, dtype=np.float32)
        source = self.observation_config.get("image", "image")
        if obs.get(source) is not None:
            images = np.asarray(obs[source])
            if self.observation_config.get("image_layout", "KCHW") == "KCHW":
                images = images.transpose(0, 2, 3, 1)
            cameras = self.observation_config.get("cameras", [])
            if len(cameras) != len(images):
                raise ValueError("Camera names must match the environment's image count")
            for camera, image in zip(cameras, images):
                data[camera] = image
        self.success = self.success or bool(success)
        data.update(sim_step=self.steps, sim_time=self.steps / self.fps,
                    episode_id=self.episode, reward=float(reward or 0),
                    success=self.success, terminated=bool(terminated), truncated=bool(truncated),
                    paused=self.blocked, raw_lang=obs.get("raw_lang", ""))
        if self.last_action is not None:
            data["executed_action"] = self.last_action.copy()
            data["action"] = self.last_action.copy()
        self.cached = data

    def reset(self):
        """Restore the configured benchmark task's initial scene."""
        self.pending = None
        self.blocked = False
        self.steps = 0
        self.success = False
        self.episode += 1
        self.last_action = None
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self._cache(obs)
        self.adapter.reset(self.env, self.raw)
        native_frequency = getattr(self.adapter, 'control_frequency', self.fps)
        if not np.isclose(native_frequency, self.fps):
            raise ValueError(f'Robot fps={self.fps} must match environment control frequency {native_frequency}')

    def process_action(self, message):
        return self.adapter.teleop_action(message, self.raw)

    def meta2act(self, mact):
        if mact["ctrl_space"] != self.space or mact["ctrl_type"] != self.native_type:
            raise ValueError("Policy action space/type must match the configured environment")
        return mact["action"]

    def publish_action(self, action):
        value = np.asarray(action, dtype=float).reshape(-1)
        if value.size != self.get_action_dim() or not np.all(np.isfinite(value)):
            raise ValueError("Invalid environment action shape or values")
        self.pending = value.copy()

    def get_action_dim(self):
        return self.arms * self.width

    def is_running(self):
        return self.env is not None

    def get_observation(self):
        return self.cached

    @staticmethod
    def obs2meta(data):
        result = {key: data[key] for key in ("state", "state_joint", "state_ee", "raw_lang") if key in data}
        images = [v.transpose(2, 0, 1) for v in data.values()
                  if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3]
        if images:
            result["image"] = np.stack(images)
        return result

    def _step(self):
        action = self.pending
        self.pending = None
        if action is None:
            action = self.adapter.idle_action()
        self.adapter.target = action.copy()
        result = self.env.step(self.adapter.environment_action(action))
        self.last_action = action
        self.steps += 1
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        success_key = self.environment_config.get("success_key", "success")
        success = bool(info.get(success_key, False)) if isinstance(info, dict) else False
        if self.environment_config.get("done_is_success", False):
            success, terminated = bool(terminated), False
        if self.max_steps and self.steps >= self.max_steps:
            truncated = True
        self.blocked = bool((success and self.on_success == "pause") or terminated or truncated)
        self._cache(obs, reward, terminated, truncated, success)
        if (terminated or truncated) and self.on_termination == "reset":
            self._reset_next = True

    def start(self):
        self.connect()
        self.shm = self.create_shm()
        self.is_running = True
        self._reset_next = False
        deadline = time.monotonic()
        try:
            while self.is_running:
                if self._reset_next:
                    self._reset_next = False
                    self.reset()
                if self.control_shm_name and self.control_shm is None:
                    try:
                        self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    except ValueError:
                        pass
                message = self.read_action()
                if not self.handle_control_message(message) and message and not self.blocked:
                    if "action" in message:
                        # Policy control messages carry native actions; teleop carries EE deltas.
                        if self.input_mode == "native":
                            self.publish_action(message["action"])
                        else:
                            self.publish_action(self.process_action(message)["action"])
                if not self.blocked:
                    self._step()
                self.write_data(self.cached)
                deadline += 1 / self.fps
                time.sleep(max(0, deadline - time.monotonic()))
                if time.monotonic() - deadline > 1 / self.fps:
                    deadline = time.monotonic()
        finally:
            self.close()

    def shutdown(self):
        self.is_running = False
        if self.env is not None:
            self.env.close()
            self.env = None

    def close(self):
        self.shutdown()
        if self.control_shm is not None:
            self.control_shm.destroy()
            self.control_shm = None
        super().close()
