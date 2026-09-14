"""
TD3 algorithm implementation based on the official reference:
https://arxiv.org/abs/1802.09477
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl.utils.action_utils as action_utils
from benchmark.base import MetaAction, MetaObs, MetaPolicy
from policy.mlp.mlp import MLPPolicy, MLPPolicyConfig
from rl.utils import polyak_update
from rl.utils.action_utils import ensure_action

from ..base import BaseAlgorithm


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    device: str = "cpu"
    state_key: str = "state"
    next_state_key: str = "next_state"
    action_key: str = "action"


class _IdentityStateNormalizer:
    def normalize_metaobs(self, mobs: MetaObs, ctrl_space: str):
        return mobs


class _IdentityActionNormalizer:
    def denormalize_metaact(self, mact: MetaAction):
        return mact


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Algorithm(BaseAlgorithm):
    """
    TD3 algorithm using MLPPolicy as actor and a twin-critic network.
    """

    def __init__(
        self,
        replay: Optional[Any],
        config: TD3Config,
        actor_config: Optional[MLPPolicyConfig] = None,
        meta_policy: Optional[MetaPolicy] = None,
        ensure_refine_fn= action_utils.tanh_action_to_space,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
        gripper_continuous: bool = False,
        **kwargs,
    ):
        if actor_config is None:
            actor_config = MLPPolicyConfig(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                chunk_size=1,
            )
        else:
            actor_config.chunk_size = 1

        self.device = torch.device(config.device)
        self._env = None
        self.config = config
        self.ensure_refine_fn = ensure_refine_fn
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.gripper_continuous = gripper_continuous

        self.actor = MLPPolicy(actor_config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)

        self.critic = Critic(config.state_dim, config.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.total_it = 0

        if meta_policy is None:
            meta_policy = MetaPolicy(
                self.actor,
                chunk_size=1,
                action_normalizer=_IdentityActionNormalizer(),
                state_normalizer=_IdentityStateNormalizer(),
                ctrl_space=ctrl_space,
                ctrl_type=ctrl_type,
            )

        super().__init__(meta_policy=meta_policy, replay=replay, **kwargs)

    def set_env(self, env: Any) -> None:
        """Attach an environment for action post-processing."""
        self._env = env

    def _actor_forward(self, state: torch.Tensor, model: Optional[MLPPolicy] = None) -> torch.Tensor:
        model = self.actor if model is None else model
        output = model(state)
        if isinstance(output, dict):
            action = output.get("action")
        else:
            action = output
        if action is None:
            raise ValueError("Actor output does not contain 'action'")
        if action.dim() == 3:
            action = action[:, 0, :]
        return action

    def _to_tensor(self, value, dtype=torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def _extract_batch_action(self, batch: Dict[str, Any]) -> Any:
        action = batch.get(self.config.action_key, None)
        if action is None:
            raise KeyError(f"Missing '{self.config.action_key}' in batch")
        if hasattr(action, "action"):
            return action.action
        if isinstance(action, dict) and "action" in action:
            return action["action"]
        return action

    def _extract_batch_state(self, batch: Dict[str, Any], key: str) -> Any:
        state = batch.get(key, None)
        if state is None:
            raise KeyError(f"Missing '{key}' in batch")
        return state

    def select_action(
        self,
        obs: Any,
        noise_scale: float = 0.0,
        env: Optional[Any] = None,
        **kwargs,
    ) -> MetaAction:
        if isinstance(obs, (list, np.ndarray)) and len(obs) > 0:
            first = obs[0] if isinstance(obs, list) else obs.flat[0]
            if hasattr(first, "__dataclass_fields__"):
                obs = self._organize_obs(obs)

        if hasattr(obs, self.config.state_key):
            state = getattr(obs, self.config.state_key)
        elif hasattr(obs, "state"):
            state = obs.state
        elif isinstance(obs, dict) and self.config.state_key in obs:
            state = obs[self.config.state_key]
        else:
            state = obs
        state_t = self._to_tensor(state)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            action = self._actor_forward(state_t)
            if noise_scale > 0:
                action = action + noise_scale * torch.randn_like(action)
            action = ensure_action(env or self._env, action, refine_fn=self.ensure_refine_fn)

        action_np = action.detach().cpu().numpy()
        return MetaAction(
            action=action_np,
            ctrl_space=self.ctrl_space,
            ctrl_type=self.ctrl_type,
            gripper_continuous=self.gripper_continuous,
        )

    def update(
        self,
        batch: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if batch is None:
            if self.replay is None:
                raise ValueError("Replay buffer is not set")
            batch_size = kwargs.get("batch_size", 256)
            if hasattr(self.replay, "sample_for_training"):
                batch = self.replay.sample_for_training(batch_size)
            else:
                raise ValueError("Replay buffer does not support sample_for_training")
            batch = self.replay.sample(batch_size)

        state = self._to_tensor(self._extract_batch_state(batch, self.config.state_key))
        next_state = self._to_tensor(self._extract_batch_state(batch, self.config.next_state_key))
        action = self._to_tensor(self._extract_batch_action(batch))
        reward = self._to_tensor(batch.get("reward"), dtype=torch.float32).unsqueeze(-1)
        done = self._to_tensor(batch.get("done"), dtype=torch.float32).unsqueeze(-1)
        truncated_raw = batch.get("truncated")
        if truncated_raw is None:
            truncated_raw = np.zeros_like(batch.get("done"))
        truncated = self._to_tensor(truncated_raw, dtype=torch.float32).unsqueeze(-1)

        terminal = done * (1.0 - truncated)
        not_done = 1.0 - terminal

        self.total_it += 1

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = self._actor_forward(next_state, model=self.actor_target)
            next_action = next_action + noise
            next_action = ensure_action(env or self._env, next_action, refine_fn=self.ensure_refine_fn)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.config.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.config.policy_freq == 0:
            actor_action = self._actor_forward(state)
            actor_action = ensure_action(env or self._env, actor_action, refine_fn=self.ensure_refine_fn)
            actor_loss = -self.critic.Q1(state, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.config.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.config.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": None if actor_loss is None else actor_loss.item(),
            "update_step": self.total_it,
        }

    def save(self, path: str, **kwargs) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(payload, path)

    def load(self, path: str, **kwargs) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

