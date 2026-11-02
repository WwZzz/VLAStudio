"""Environment/device adapters owned by the VLAStudio execution runtime.

These adapters perform one native publication when the runtime asks them to.
Chunk selection and publication timing belong to the runtime/ActionManager.
"""
from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from types import SimpleNamespace


def _namespace(value):
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _meta_observation(value):
    """Keep the native normalized observation contract at the backend boundary."""
    from vlastudio.benchmark.base import MetaObs, dict2meta
    if isinstance(value, MetaObs):
        return value
    if isinstance(value, Mapping):
        return dict2meta(dict(value), mtype="obs")
    raise TypeError("The environment must return a MetaObs or normalized observation mapping")


def _require_unused_channel(name):
    """Native writers replace same-name SHM: never replace another session's data."""
    from multiprocessing import shared_memory
    try:
        existing = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    existing.close()
    raise FileExistsError(f"Shared memory {name!r} is already in use; use distinct device names")


class SimulationBackend:
    mode = "sim"

    def __init__(self, config, session_id):
        self.config = copy.deepcopy(config)
        self.session_id = session_id
        self.device_shm_names = []
        self.device_configs = []
        self.env = None
        self._observation = None
        self._captured_at = 0.0

    def open(self):
        if self.env is not None:
            raise RuntimeError("Simulation backend is already open")
        from vlastudio.configs.loader import ConfigLoader
        from vlastudio.entrypoints.eval_sim import load_env_module

        source = self.config.get("config", self.config.get("env"))
        if source is None:
            raise ValueError("Simulation execution requires an environment config")
        loader = ConfigLoader(unknown_args=self.config.get("overrides", []))
        if isinstance(source, (Mapping, list)):
            rows = source.get("envs", source) if isinstance(source, Mapping) else source
            rows = rows if isinstance(rows, list) else [rows]
            normalized = []
            for row in rows:
                flat = loader.normalize_config(copy.deepcopy(row))
                for key, value in flat.get("args", {}).items():
                    flat.setdefault(key, value)
                normalized.append(_namespace(flat))
            choices = normalized
        else:
            loaded, _ = loader.load_env(str(source))
            choices = loaded if isinstance(loaded, list) else [loaded]
        if not choices:
            raise ValueError("Environment config contains no environments")
        if len(choices) > 1 and "env_index" not in self.config:
            raise ValueError("A multi-environment config requires an explicit env_index")
        index = self.config.get("env_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(choices):
            raise ValueError("env_index is outside the configured environments")
        selected = copy.deepcopy(choices[index])
        selected.rollout_index = self.config.get("rollout_index", 0)
        module, _, _ = load_env_module(selected)
        self.env = module.create_env(selected)
        try:
            initial = self.env.reset()
            # Gymnasium reset: (observation, info). A normalized observation
            # itself is never a two-tuple under VLAStudio's MetaObs contract.
            if isinstance(initial, tuple) and len(initial) == 2 and isinstance(initial[1], Mapping):
                initial = initial[0]
            self._observation = _meta_observation(initial)
            self._captured_at = time.perf_counter()
        except BaseException:
            self.close()
            raise

    def observe(self):
        # A paused simulator does not acquire a new physical state. Returning
        # the original timestamp makes this distinction visible to the runtime.
        return self._observation, self._captured_at

    def publish(self, step_action):
        if self.env is None:
            raise RuntimeError("Simulation backend is not open")
        import numpy as np
        action = step_action
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) != 1:
                raise ValueError("Single-environment execution requires exactly one MetaAction")
            action = action[0]
        # Preserve action, ctrl_type and ctrl_space for MetaEnv.meta2act().
        if not isinstance(action, Mapping):
            raise TypeError("Simulation execution expects a MetaAction mapping")
        result = self.env.step(action)
        if not isinstance(result, (list, tuple)) or len(result) not in (4, 5):
            raise TypeError("Environment step must return a four- or five-item result")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False
        self._observation = _meta_observation(observation)
        self._captured_at = time.perf_counter()
        return {
            "done": bool(done), "reward": reward, "info": info,
            "terminated": bool(terminated), "truncated": bool(truncated),
        }

    def close(self):
        env, self.env = self.env, None
        self._observation = None
        self._captured_at = 0.0
        if env is not None:
            close = getattr(env, "close", None)
            if close is not None:
                close()


class RobotBackend:
    mode = "robot"

    def __init__(self, config, session_id):
        self.config = copy.deepcopy(config)
        self.session_id = session_id
        self.device_shm_names = []
        self.device_configs = []
        self._processes = []
        self._channels = []
        self._control = None
        self._synchronizer = None
        self._obs2meta = None
        self._observation = None
        self._captured_at = 0.0
        self._opened = False
        self._owned_device_names = []

    def open(self):
        if self._opened:
            raise RuntimeError("Robot backend is already open")
        from vlastudio.configs.loader import ConfigLoader
        from vlastudio.deploy.base import create_obs2meta_func, is_robot_config
        from vlastudio.deploy.shm_utils import SharedMemoryChannel, SharedMemoryDataSynchronizer
        from vlastudio.deploy.utils import load_device_configs, start_devices

        source = self.config.get("config", self.config.get("robot"))
        if source is None:
            raise ValueError("Robot execution requires a device configuration")
        loader = ConfigLoader(unknown_args=self.config.get("overrides", []))
        if isinstance(source, (Mapping, list)):
            inline_source = copy.deepcopy(source)

            class InlineLoader:
                def load_robot(self, name):
                    return inline_source, "<inline robot config>"

            loader = InlineLoader()
            source = "<inline>"
        control_name = f"execution_{self.session_id}_control"
        args = SimpleNamespace(robot=str(source), teleop=None)
        configs, names, _, _ = load_device_configs(
            args, [], cfg_loader=loader, force_robot_control_shm=control_name)
        controlled = [item for item in configs if is_robot_config(item)]
        if len(controlled) != 1:
            raise ValueError("Execution currently requires exactly one controlled robot")
        if len(names) != len(set(names)) or control_name in names:
            raise ValueError("Every device/control shared-memory channel must have a unique name")
        for name in [control_name, *names]:
            _require_unused_channel(name)
        self.device_configs = configs
        self.device_shm_names = list(names)
        timeout = float(self.config.get("startup_timeout_s", 30.0))
        if timeout <= 0:
            raise ValueError("startup_timeout_s must be positive")
        self._opened = True
        try:
            self._control = SharedMemoryChannel(name=control_name, max_size_mb=1, is_writer=True)
            self._processes = start_devices(configs)
            self._owned_device_names = list(names)
            deadline = time.perf_counter() + timeout
            for name in names:
                while True:
                    if any(not process.is_alive() for process in self._processes):
                        raise RuntimeError("A robot/camera device exited during startup")
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("Timed out connecting execution device shared memory")
                    channel = SharedMemoryChannel(name=name, is_writer=False, timeout=min(0.1, remaining))
                    if channel.shm is not None:
                        self._channels.append((name, channel))
                        break
                    channel.destroy()
            self._synchronizer = SharedMemoryDataSynchronizer(
                shm_channels=self._channels,
                buffer_maxlen=int(self.config.get("sync_buffer_maxlen", 100)),
                max_tolerance_s=float(self.config.get("sync_max_tolerance_s", 0.05)),
            )
            self._obs2meta = create_obs2meta_func(configs)
            while self._observation is None:
                self.observe()
                if self._observation is not None:
                    break
                if any(not process.is_alive() for process in self._processes):
                    raise RuntimeError("A robot/camera device exited before its first observation")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Timed out waiting for an execution observation")
                time.sleep(0.001)
        except BaseException:
            self.close()
            raise

    def observe(self):
        if self._synchronizer is not None:
            frame = self._synchronizer.get_latest_frame_nonblocking(strict_new_only=True)
            if frame is not None:
                observation = self._obs2meta(frame)
                timestamps = [float(item.get("__timestamp__", item.get("timestamp", 0.0)))
                              for item in frame.values()]
                if observation is not None:
                    self._observation = observation
                    # Every participating device must pass a freshness barrier.
                    self._captured_at = min(timestamps) if timestamps else 0.0
        return self._observation, self._captured_at

    def publish(self, step_action):
        if self._control is None:
            raise RuntimeError("Robot backend is not open")
        import numpy as np
        from vlastudio.deploy.inference import _extract_step_action
        action = _extract_step_action(step_action)
        if action is None or np.asarray(action).size == 0:
            raise ValueError("Robot publication requires a nonempty native action")
        self._control.write({"action": np.asarray(action), "timestamp": time.perf_counter()})
        # This acknowledges publication only; physical completion/success is
        # decided by task management/evaluation rather than the robot transport.
        return {"done": None}

    def close(self):
        from vlastudio.deploy.shm_utils import cleanup_all_shm
        processes, self._processes = self._processes, []
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.perf_counter() + float(self.config.get("close_timeout_s", 5.0))
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.perf_counter()))
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
        for _, channel in self._channels:
            channel.destroy()
        self._channels = []
        if self._control is not None:
            self._control.destroy()
            self._control = None
        cleanup_all_shm(self._owned_device_names)
        self._owned_device_names = []
        self._synchronizer = None
        self._obs2meta = None
        self._observation = None
        self._captured_at = 0.0
        self._opened = False


def create_backend(config: dict, session_id: str):
    if not isinstance(config, dict):
        raise TypeError("Execution backend configuration must be a mapping")
    if config.get('type') == 'connected':
        from .target import AttachedBackend
        return AttachedBackend(config)
    mode = config.get("mode", config.get("backend", config.get("type")))
    if mode == "sim":
        return SimulationBackend(config, session_id)
    if mode == "robot":
        return RobotBackend(config, session_id)
    raise ValueError("Execution backend mode must be 'sim' or 'robot'")
