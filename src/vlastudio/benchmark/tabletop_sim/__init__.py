"""
Tabletop-Sim environment integration for ILStudio.

Tabletop-Sim (https://github.com/jellyho/Tabletop-Sim) is a ``dm_control`` based
bimanual tabletop manipulation simulator built on the ALOHA platform.  It provides
multiple tasks (e.g. dish drainer, handover box, ...) and supports three action
spaces: ``joint_pos`` (default, 14-dim for bimanual), ``ee_quat_pos`` and
``ee_6d_pos``.

Observation format (bimanual tasks)
-----------------------------------
``env.reset()`` / ``env.step(act)`` returns a ``dm_env.TimeStep`` with
``observation`` being a dict like::

    {
        'qpos':                ndarray(14,),
        'qvel':                ndarray(14,),
        'ee_pos':              ndarray(18,),    # 2 * (pos(3) + quat(4) + gripper(1))
        'ee_rpy_pos':          ndarray(14,),    # 2 * (pos(3) + rpy(3) + gripper(1))
        'ee_6d_pos':           ndarray(20,),    # 2 * (pos(3) + 6d(6) + gripper(1))
        'env_state':           ndarray(...,),
        'images': {
            'back':            ndarray(480, 640, 3),
            'wrist_left':      ndarray(240, 320, 3),
            'wrist_right':     ndarray(240, 320, 3),
        },
        'language_instruction': str,
    }

The matching LeRobot v3.0 datasets (e.g. ``jellyho/aloha_dish_drainer``) store
the ``back`` view under the key ``observation.images.agentview``.  When used in
configs, ``agentview`` is treated as an alias of ``back``.

Action space
------------
For ``joint_pos`` (default), the action is 14-dim:
``[left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]``
(gripper is normalized in [-1, 1]).
"""
from dataclasses import asdict
import os
import time
from multiprocessing import current_process
from typing import Optional

import numpy as np
import cv2
from loguru import logger

# Ensure off-screen rendering works on headless servers (CI, slurm, ...)
os.environ.setdefault("MUJOCO_GL", "egl")

import tabletop  # type: ignore
from tabletop.aloha_env import ALOHA_TASK_CONFIGS  # type: ignore
from tabletop.constants import DT as TABLETOP_DT  # type: ignore

from vlastudio.benchmark.base import MetaEnv, MetaAction, MetaObs


# Dataset-side aliases → simulator image key (see module docstring)
_CAMERA_ALIASES = {
    "agentview": "back",
    "observation.images.agentview": "back",
    "primary": "back",
    "top": "back",
    "observation.images.wrist_left": "wrist_left",
    "observation.images.wrist_right": "wrist_right",
    "wrist": "wrist_left",  # single-arm tasks expose only 'wrist'
}


def _resolve_sim_cam_key(name: str) -> str:
    return _CAMERA_ALIASES.get(name, name)


def create_env(config):
    """Factory used by ``vlastudio evaluate`` and ``configs.loader``."""
    return TabletopSimEnv(config)


class TabletopSimEnv(MetaEnv):
    """ILStudio MetaEnv wrapper around ``tabletop.env(...)``."""

    def __init__(self, config, *args, **kwargs):
        self.config = config

        # -------- task & action space --------
        self.task_name = getattr(config, "task", "aloha_dish_drainer")
        if self.task_name not in ALOHA_TASK_CONFIGS:
            raise ValueError(
                f"Unknown Tabletop-Sim task '{self.task_name}'. "
                f"Available tasks: {sorted(ALOHA_TASK_CONFIGS.keys())}"
            )
        self.action_space = getattr(config, "action_space", "joint_pos")
        assert self.action_space in ("joint_pos", "ee_quat_pos", "ee_6d_pos"), (
            f"Invalid action_space='{self.action_space}'. "
            f"Must be one of joint_pos / ee_quat_pos / ee_6d_pos"
        )

        # -------- control space / type (for MetaAction) --------
        # For Tabletop-Sim all action spaces are absolute target setpoints
        # consumed directly by the physics simulator.  We keep ``ctrl_space``
        # user-controllable so that downstream normalizers pick the right
        # statistics (e.g., state_joint vs state_ee).
        default_ctrl_space = "joint" if self.action_space == "joint_pos" else "ee"
        self.ctrl_space = getattr(config, "ctrl_space", default_ctrl_space)
        self.ctrl_type = getattr(config, "ctrl_type", "abs")

        # -------- cameras --------
        # Accept either raw simulator keys (``back``, ``wrist_left``, ``wrist_right``)
        # or dataset-side aliases (``agentview`` etc.).
        raw_cams = getattr(config, "camera_names", None)
        if raw_cams is None:
            raw_cams = ["agentview"]
        if isinstance(raw_cams, str):
            raw_cams = [raw_cams]
        self.camera_names_requested = list(raw_cams)
        self.camera_keys_sim = [_resolve_sim_cam_key(c) for c in raw_cams]

        # -------- image sizes --------
        image_size = getattr(config, "image_size", [640, 480])
        if isinstance(image_size, int):
            width, height = image_size, image_size
        else:
            width, height = image_size
        self.image_size = (int(width), int(height))

        # -------- horizon --------
        # ``max_timesteps`` is consumed by benchmark.utils.evaluate, but we also
        # expose it so external loggers can use it.
        ep_len_sec = ALOHA_TASK_CONFIGS[self.task_name].get("episode_len", 10)
        self.max_timesteps = int(getattr(config, "max_timesteps", int(ep_len_sec / TABLETOP_DT)))

        # -------- initial pose seeding --------
        self.benchmark_init_id: Optional[int] = getattr(config, "benchmark_init_id", None)

        env = self._build_env()
        super().__init__(env)
        self.raw_lang = ""

    # ------------------------------------------------------------------
    # Env life-cycle
    # ------------------------------------------------------------------
    def _build_env(self):
        env = tabletop.env(self.task_name, self.action_space)
        action_spec = env.action_spec()
        self.min_action = np.asarray(action_spec.minimum, dtype=np.float32)
        self.max_action = np.asarray(action_spec.maximum, dtype=np.float32)
        return env

    def reset(self):
        # Seed per-process (same pattern as the ALOHA env) so that parallel
        # SubprocVectorEnv workers don't all draw identical initial poses.
        pid = current_process().pid or 0
        seed = (pid * 1000 + time.time_ns()) % (2 ** 32)
        np.random.seed(seed)

        ts = self.env.reset()
        # Allow seeding from the pre-recorded benchmark info bank.
        if self.benchmark_init_id is not None and getattr(self.env.task, "benchmark_info", None) is not None:
            try:
                ts = self.env.task.benchmark_init(self.env.physics, int(self.benchmark_init_id))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"benchmark_init failed: {e}; falling back to random init")
        # Capture language instruction (may change per episode, e.g., BoxIntoPot)
        self.raw_lang = str(ts.observation.get("language_instruction", "") or "")
        self.prev_obs = self.obs2meta(ts.observation)
        return self.prev_obs

    def _clip_action_for_env(self, action: np.ndarray) -> np.ndarray:
        """
        Clip only where the simulator expects bounded inputs.

        IMPORTANT:
        - Tabletop-Sim internally expects gripper commands normalized to [-1, 1]
          (see `tabletop.constants.ALOHA_GRIPPER_UNNORMALIZE_FN`), but
          `dm_control`'s `action_spec()` exposes the actuator ctrlrange
          ([0.002, 0.037]) which is the *unnormalized* physical range.
          Therefore we must NOT clip gripper dims using `action_spec()`,
          otherwise grippers barely move.
        - For `ee_*` action spaces, `action_spec()` is not meaningful for the
          pose components either (they are not actuator controls), so we only
          clip grippers.
        """
        a = np.asarray(action, dtype=np.float32).copy()
        if a.ndim != 1:
            return a

        # Identify gripper indices by action_dim + action_space.
        gripper_idx = []
        if self.action_space == "joint_pos":
            if a.shape[0] == 14:
                gripper_idx = [6, 13]
            elif a.shape[0] == 7:
                gripper_idx = [6]
            else:
                gripper_idx = [a.shape[0] - 1]
        elif self.action_space == "ee_quat_pos":
            if a.shape[0] == 16:
                gripper_idx = [7, 15]
            elif a.shape[0] == 8:
                gripper_idx = [7]
            else:
                gripper_idx = [a.shape[0] - 1]
        elif self.action_space == "ee_6d_pos":
            if a.shape[0] == 20:
                gripper_idx = [9, 19]
            elif a.shape[0] == 10:
                gripper_idx = [9]
            else:
                gripper_idx = [a.shape[0] - 1]

        # Clip grippers to [-1, 1] (Tabletop-Sim convention).
        for gi in gripper_idx:
            if 0 <= gi < a.shape[0]:
                a[gi] = float(np.clip(a[gi], -1.0, 1.0))

        # For joint_pos, actuator bounds for arm joints are meaningful: clip them.
        if self.action_space == "joint_pos" and hasattr(self, "min_action") and hasattr(self, "max_action"):
            mn = np.asarray(self.min_action, dtype=np.float32)
            mx = np.asarray(self.max_action, dtype=np.float32)
            if mn.shape == a.shape and mx.shape == a.shape:
                # Do not use actuator ctrlrange for grippers.
                mask = np.ones_like(a, dtype=bool)
                for gi in gripper_idx:
                    if 0 <= gi < mask.shape[0]:
                        mask[gi] = False
                a[mask] = np.clip(a[mask], mn[mask], mx[mask])

        return a

    def step(self, *args, **kwargs):
        act = self.meta2act(*args, **kwargs)
        act = self._clip_action_for_env(act)
        ts = self.env.step(act)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        max_reward = getattr(self.env.task, "max_reward", 1) or 1
        done = bool(reward >= max_reward)
        info = {
            "success": done,
            "reward": reward,
            "max_reward": max_reward,
        }
        self.raw_lang = str(ts.observation.get("language_instruction", self.raw_lang) or self.raw_lang)
        self.prev_obs = self.obs2meta(ts.observation)
        results = [asdict(self.prev_obs), reward, done, info]
        return tuple(results)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MetaEnv interface
    # ------------------------------------------------------------------
    def meta2act(self, maction):
        """Convert a ``MetaAction`` (or plain ndarray) into a raw env action."""
        if isinstance(maction, MetaAction):
            action = maction.action
        elif isinstance(maction, dict) and "action" in maction:
            action = maction["action"]
        else:
            action = maction
        action = np.asarray(action, dtype=np.float32)
        # Vector envs may pass batched (1, A) - unwrap to (A,)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        return action

    def obs2meta(self, obs):
        # --- state (depends on control space) ---
        if self.ctrl_space == "joint":
            state_vec = np.asarray(obs["qpos"], dtype=np.float32)
        elif self.action_space == "ee_6d_pos":
            state_vec = np.asarray(obs["ee_6d_pos"], dtype=np.float32)
        elif self.action_space == "ee_quat_pos":
            state_vec = np.asarray(obs["ee_pos"], dtype=np.float32)
        else:
            state_vec = np.asarray(obs["qpos"], dtype=np.float32)

        # --- images ---
        width, height = self.image_size
        images = []
        available = obs.get("images", {}) or {}
        for cam_key in self.camera_keys_sim:
            if cam_key not in available:
                logger.warning(
                    f"Requested camera '{cam_key}' not available; "
                    f"available keys = {list(available.keys())}. Using zeros."
                )
                img = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img = available[cam_key]
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
            images.append(img)
        image_stack = np.stack(images).astype(np.uint8)  # (K, H, W, 3)
        image_stack = image_stack.transpose(0, 3, 1, 2)  # (K, 3, H, W)

        raw_lang = str(obs.get("language_instruction", self.raw_lang) or self.raw_lang)
        return MetaObs(
            state=state_vec,
            state_joint=np.asarray(obs["qpos"], dtype=np.float32),
            state_ee=np.asarray(obs.get("ee_6d_pos", obs.get("ee_pos", state_vec)), dtype=np.float32),
            image=image_stack,
            raw_lang=raw_lang,
        )
