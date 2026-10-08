"""LIBERO observations and reset protocol for the released FastWAM Base policy."""
from dataclasses import asdict
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from vlastudio.benchmark.base import MetaEnv, MetaObs
from vlastudio.data_utils.pose_utils import quat2axisangle


class FastWAMLiberoEnv(MetaEnv):
    def __init__(self, config):
        self.config = config
        suite_name, task_id = config.task.rsplit("_", 1)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(int(task_id))
        self.states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file, map_location="cpu", weights_only=False)
        self.index = int(getattr(config, "rollout_index", 0))
        if not 0 <= self.index < len(self.states):
            raise ValueError(f"No official initial state for rollout {self.index}")
        self.raw_lang = task.language
        self.ctrl_space, self.ctrl_type = "ee", "delta"
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256,
        )
        env.seed(int(getattr(config, "seed", 42)))
        super().__init__(env)

    def obs2meta(self, obs):
        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]]).astype(np.float32)
        images = np.stack([obs[key][::-1, ::-1] for key in ("agentview_image", "robot0_eye_in_hand_image")])
        return MetaObs(state=state, state_ee=state, image=images.transpose(0, 3, 1, 2), raw_lang=self.raw_lang)

    def reset(self):
        self.index = int(getattr(self.config, "rollout_index", 0))
        if not 0 <= self.index < len(self.states):
            raise ValueError(f"No official initial state for rollout {self.index}")
        self.env.reset()
        obs = self.env.set_init_state(self.states[self.index])
        for _ in range(30):
            obs, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs

    def meta2act(self, action):
        if action["ctrl_space"] != "ee" or action["ctrl_type"] != "delta":
            raise ValueError("FastWAM LIBERO requires native ee/delta actions")
        return np.asarray(action["action"], dtype=np.float32)

    def step(self, action):
        obs, reward, _, info = self.env.step(self.meta2act(action))
        self.prev_obs = self.obs2meta(obs)
        return asdict(self.prev_obs), reward, bool(self.env.check_success()), info
