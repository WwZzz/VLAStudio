from collections.abc import Iterable

import torch
import torch.nn as nn


ACTION_REGISTRY = {}


def register_action(name):
    def _wrap(cls):
        ACTION_REGISTRY[name.lower()] = cls
        return cls

    return _wrap


def build_action_space(name, **kwargs):
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


class BaseActionSpace(nn.Module):
    name = "base"
    dim_action = 0
    gripper_idx = ()

    def __init__(self):
        super().__init__()

    def compute_loss(self, pred, target):
        raise NotImplementedError

    def preprocess(self, proprio, action, mode="train"):
        return proprio, action

    def postprocess(self, action):
        return action


def _ensure_indices_valid(dim_action: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= dim_action]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim {dim_action}")


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        _ensure_indices_valid(pred.shape[-1], self.gripper_idx, "gripper_idx")
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE
        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action):
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        _ensure_indices_valid(pred.shape[-1], self.gripper_idx, "gripper_idx")
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE
        joints_idx = tuple(i for i in range(pred.shape[-1]) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE
        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action):
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    JOINTS_SCALE = 1.0

    def __init__(self, real_dim: int, max_dim: int):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x):
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                x = torch.cat([x, x.new_zeros(pad_shape)], dim=-1)
            else:
                x = x[..., : self.real_dim]
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        return torch.cat([x, x.new_zeros(pad_shape)], dim=-1)

    def _trim_to_real_dim(self, x):
        return x[..., : self.real_dim]

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        joints_loss = self.mse(pred[:, :, : self.real_dim], target[:, :, : self.real_dim]) * self.JOINTS_SCALE
        return {"joints_loss": joints_loss}

    def preprocess(self, proprio, action, mode="train"):
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action):
        return self._trim_to_real_dim(action)
