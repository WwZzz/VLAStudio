"""A compact external policy package for VLAStudio.

This example keeps the integration surface small while using a realistic visual
encoder: a torchvision ResNet backbone initialized from ImageNet weights.  Larger
custom policies can keep the same hooks and replace the backbone, state encoder,
or action head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig


class MyPolicyConfig(PretrainedConfig):
    model_type = "vlastudio_example_my_policy"

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        chunk_size: int = 50,
        hidden_dim: int = 512,
        vision_dim: int = 512,
        image_size: int = 224,
        use_images: bool = True,
        vision_backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_vision: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.vision_dim = vision_dim
        self.image_size = image_size
        self.use_images = use_images
        self.vision_backbone = vision_backbone
        self.pretrained = pretrained
        self.freeze_vision = freeze_vision


class MyPolicy(PreTrainedModel):
    """Pretrained-ResNet vision + state policy with a chunked action head."""

    config_class = MyPolicyConfig

    def __init__(self, config: MyPolicyConfig):
        super().__init__(config)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        if config.use_images:
            self.vision_encoder, resnet_dim = self._build_resnet_encoder(
                config.vision_backbone,
                bool(config.pretrained),
            )
            self.vision_projector = nn.Sequential(
                nn.Linear(resnet_dim, config.vision_dim),
                nn.LayerNorm(config.vision_dim),
                nn.SiLU(),
            )
            if config.freeze_vision:
                for parameter in self.vision_encoder.parameters():
                    parameter.requires_grad_(False)
            self.register_buffer(
                "image_mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
        fused_dim = config.hidden_dim + (config.vision_dim if config.use_images else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.query_embed = nn.Parameter(torch.randn(config.chunk_size, config.hidden_dim) * 0.02)
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.post_init()

    @staticmethod
    def _build_resnet_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

        name = backbone.lower()
        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        elif name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported vision_backbone={backbone!r}; use 'resnet18' or 'resnet34'.")
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_dim

    def _encode_images(self, image: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
        if not self.config.use_images:
            raise RuntimeError("Image encoding was requested while use_images=False")
        if image is None:
            return torch.zeros(batch_size, self.config.vision_dim, device=device)
        image = image.to(device=device, dtype=torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if image.ndim == 5:
            bsz, cameras, channels, height, width = image.shape
            image = image.reshape(bsz * cameras, channels, height, width)
            multi_camera = True
        elif image.ndim == 4:
            bsz = image.shape[0]
            cameras = 1
            multi_camera = False
        else:
            raise ValueError(f"Expected image shape [B,C,H,W] or [B,N,C,H,W], got {tuple(image.shape)}")
        image = F.interpolate(
            image,
            size=(int(self.config.image_size), int(self.config.image_size)),
            mode="bilinear",
            align_corners=False,
        )
        image = (image - self.image_mean.to(image.device)) / self.image_std.to(image.device)
        encoded = self.vision_projector(self.vision_encoder(image))
        if multi_camera:
            return encoded.reshape(bsz, cameras, -1).mean(dim=1)
        return encoded

    def forward(self, state, image=None, action=None, is_pad=None, **kwargs):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(dtype=torch.float32)
        state_features = self.state_encoder(state)
        features = [state_features]
        if self.config.use_images:
            features.append(self._encode_images(image, state.shape[0], state.device))
        fused = self.fusion(torch.cat(features, dim=-1))
        pred = self.action_head(fused[:, None, :] + self.query_embed[None, :, :])
        if action is None:
            return {"action": pred}
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
        action = action[..., : self.config.action_dim].to(device=pred.device, dtype=pred.dtype)
        pred = pred[:, : action.shape[1], :]
        loss = F.mse_loss(pred, action, reduction="none")
        if is_pad is not None:
            mask = (~is_pad[:, : pred.shape[1]].to(device=pred.device, dtype=torch.bool)).unsqueeze(-1)
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        else:
            loss = loss.mean()
        return {"loss": loss}

    def select_action(self, batch_obs):
        device = next(self.parameters()).device
        state = batch_obs["state"].to(device)
        image = batch_obs.get("image")
        if image is not None:
            image = image.to(device)
        with torch.no_grad():
            action = self.forward(state=state, image=image)["action"]
        return action.cpu().numpy()


@dataclass
class MyDataProcessor:
    state_dim: int
    use_images: bool = True

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(sample["state"], dtype=np.float32)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state.shape[-1]}")
        result = {"state": state}
        if "action" in sample:
            result["action"] = np.asarray(sample["action"], dtype=np.float32)
        if "is_pad" in sample:
            result["is_pad"] = np.asarray(sample["is_pad"], dtype=bool)
        elif "action" in result and result["action"].ndim >= 2:
            result["is_pad"] = np.zeros(result["action"].shape[0], dtype=bool)
        if self.use_images and "image" in sample:
            result["image"] = np.asarray(sample["image"], dtype=np.float32)
        return result


def _stack(values, *, dtype=torch.float32):
    tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
    return torch.stack(tensors, dim=0)


def data_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    result = {"state": _stack([item["state"] for item in batch])}
    if "action" in batch[0]:
        result["action"] = _stack([item["action"] for item in batch])
    if "is_pad" in batch[0]:
        result["is_pad"] = _stack([item["is_pad"] for item in batch], dtype=torch.bool)
    elif "action" in result and result["action"].ndim == 3:
        result["is_pad"] = torch.zeros(result["action"].shape[:2], dtype=torch.bool)
    if "image" in batch[0]:
        result["image"] = _stack([item["image"] for item in batch])
    return result


def load_model(args):
    if not getattr(args, "is_training", False):
        model = MyPolicy.from_pretrained(args.model_name_or_path)
    else:
        model_args = dict(getattr(args, "model_args", {}) or {})
        model = MyPolicy(MyPolicyConfig(**model_args))
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"model": model, "config": model.config}


def get_data_processor(args, model_components):
    config = model_components["model"].config
    return MyDataProcessor(state_dim=config.state_dim, use_images=config.use_images)


def get_data_collator(args, model_components):
    return data_collator


__all__ = [
    "MyPolicy",
    "MyPolicyConfig",
    "MyDataProcessor",
    "data_collator",
    "get_data_collator",
    "get_data_processor",
    "load_model",
]

