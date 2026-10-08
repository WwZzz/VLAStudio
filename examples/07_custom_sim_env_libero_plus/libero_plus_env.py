"""External LIBERO-Plus adapter. Observations/actions stay in native units."""
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from loguru import logger
from vlastudio.benchmark.base import MetaEnv, MetaObs
from vlastudio.data_utils.pose_utils import quat2axisangle


def load_backend(require_plus=True):
    """Honor the explicit checkout; refuse silent fallback to vanilla LIBERO."""
    root = os.environ.get("LIBERO_ROOT")
    if root:
        root = Path(root).expanduser().resolve()
        if not (root / "libero/libero/__init__.py").is_file():
            raise ValueError(f"Invalid LIBERO_ROOT: {root}")
        sys.path.insert(0, str(root))
    package = importlib.import_module("libero.libero")
    package_root = Path(package.__file__).resolve().parent
    if root and package_root != root / "libero/libero":
        raise RuntimeError("A different LIBERO is already imported; start a fresh process.")
    classification = package_root / "benchmark/task_classification.json"
    if require_plus and not classification.is_file():
        raise RuntimeError(f"Expected LIBERO-Plus, imported {package_root}. Run setup.py. "
                           "Use require_plus: false only for a labeled vanilla control.")
    if not require_plus and classification.is_file():
        raise RuntimeError("Standard control requires vanilla LIBERO; start a fresh process with its source/config.")
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        path = Path(package.get_libero_path(key)).resolve()
        if key == "benchmark_root" and path != package_root:
            raise RuntimeError(f"Config points at {path}, imported {package_root}")
        if not path.is_dir():
            raise FileNotFoundError(f"Missing LIBERO {key}: {path}")
    return package, classification


def init_state_path(root, task):
    """Layout variants have separate states; visual variants share base states."""
    filename = task.init_states_file
    if "_add_" in filename or "_level" in filename:
        return Path(root) / "libero_newobj" / task.problem_folder / filename
    filename = re.sub(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+", "", filename)
    return Path(root) / task.problem_folder / filename


class LiberoPlusEnv(MetaEnv):
    def __init__(self, config):
        self.config = config
        self.ctrl_space = getattr(config, "ctrl_space", "ee")
        self.ctrl_type = getattr(config, "ctrl_type", "delta")
        if (self.ctrl_space, self.ctrl_type) != ("ee", "delta"):
            raise ValueError("This example requires ee/delta OSC_POSE actions.")
        package, classification = load_backend(getattr(config, "require_plus", True))
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        import torch

        suite_name, task_id = config.task.rsplit("_", 1)
        with redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[suite_name]()
        task_name = getattr(config, "task_name", None)
        if task_name:
            matches = [task for task in suite.tasks if task.name == task_name]
            if len(matches) != 1:
                raise ValueError(f"Expected one task named {task_name}, found {len(matches)}")
            task = matches[0]
        else:
            task = suite.get_task(int(task_id))
        self.task_name, self.raw_lang = task.name, task.language
        self.category = "standard"
        if classification.is_file():
            rows = json.loads(classification.read_text())[suite_name]
            row = next((row for row in rows if row["name"] == task.name), None)
            if row:
                self.category = row["category"]
        expected = getattr(config, "perturbation", None)
        if expected and expected != self.category:
            raise ValueError(f"Task category is {self.category}, expected {expected}")
        path = init_state_path(package.get_libero_path("init_states"), task)
        self.init_states = np.asarray(torch.load(path, weights_only=False))
        if self.init_states.ndim == 1:
            self.init_states = self.init_states[None, :]
        if not len(self.init_states):
            raise ValueError(f"No initial states in {path}")
        self.episode_index = int(getattr(config, "rollout_index", 0))
        self.init_state_offset = int(getattr(config, "init_state_offset", 0))
        self.seed = int(getattr(config, "seed", 0))
        self.num_steps_wait = int(getattr(config, "num_steps_wait", 10))
        if self.num_steps_wait < 0:
            raise ValueError("num_steps_wait must be nonnegative")
        self.use_wrist = bool(getattr(config, "use_wrist", True))
        self.image_transform = getattr(config, "image_transform", "rotate180")
        if self.image_transform not in ("identity", "rotate180"):
            raise ValueError("image_transform must be identity or rotate180")
        size = getattr(config, "image_size", [256, 256])
        height, width = (size, size) if isinstance(size, int) else size
        self.image_size = (width, height)
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(package.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=height, camera_widths=width, control_freq=20, hard_reset=True,
        )
        # Keep the standard task instruction for visual/physical variants.
        # Only language variants deliberately replace it with the BDDL rewrite.
        if self.category == "Language Instructions":
            self.raw_lang = env.language_instruction
        elif classification.is_file():
            base_name = re.sub(r"_(?:(?:language|view|light|table|tb|add)_-?\d|level_?-?\d).*$", "", task.name)
            self.raw_lang = benchmark.grab_language_from_filename(suite_name, base_name + ".bddl")
        super().__init__(env)
        logger.info("Backend={} task={} category={} cameras={} transform={}",
                    package.__file__, task.name, self.category,
                    2 if self.use_wrist else 1, self.image_transform)

    def reset(self):
        self.env.seed(self.seed + self.episode_index)
        self.env.reset()
        index = (self.init_state_offset + self.episode_index) % len(self.init_states)
        # reset randomizes placements: restore the official state AFTER reset.
        obs = self.env.set_init_state(self.init_states[index])
        if self.category == "Robot Initial States":
            # The shared state file also contains standard robot joints. Preserve
            # its object placements but reapply the named variant's robot pose.
            for robot in self.env.robots:
                robot.set_robot_joint_positions(robot.init_qpos)
                robot.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
            obs = self.env.set_init_state(self.env.get_sim_state())
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        logger.info("task={} episode={} init_state={}", self.task_name, self.episode_index, index)
        self.episode_index += 1
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs

    def obs2meta(self, obs):
        gripper = obs["robot0_gripper_qpos"]
        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), gripper]).astype(np.float32)
        keys = ["agentview_image"] + (["robot0_eye_in_hand_image"] if self.use_wrist else [])
        images = [obs[key] for key in keys]
        if self.image_transform == "rotate180":
            images = [img[::-1, ::-1] for img in images]
        return MetaObs(
            state=state, state_ee=state,
            state_joint=np.concatenate([obs["robot0_joint_pos"], gripper]).astype(np.float32),
            image=np.stack(images).transpose(0, 3, 1, 2).copy(), raw_lang=self.raw_lang,
        )

    def meta2act(self, maction):
        if (maction["ctrl_space"], maction["ctrl_type"]) != ("ee", "delta"):
            raise ValueError("Expected ee/delta policy output")
        action = np.asarray(maction["action"], dtype=np.float32).copy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite 7D action, got {action}")
        # Native LIBERO demonstrations already use [-1, 1] gripper actions.
        return action

    def step(self, action):
        obs, reward, _, info = self.env.step(self.meta2act(action))
        success = bool(self.env.check_success())
        self.prev_obs = self.obs2meta(obs)
        # VLAStudio's default evaluator treats done as success, not timeout.
        return asdict(self.prev_obs), reward, success, dict(info, success=success)


def create_env(config):
    return LiberoPlusEnv(config)
