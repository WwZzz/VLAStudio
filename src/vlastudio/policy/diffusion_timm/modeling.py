"""
DiffusionUnetTimmPolicy - A flexible diffusion policy using timm vision encoders.
Adapted for ILStudio framework.
"""
import os
import math
import copy
from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import torchvision.transforms as transforms
from einops import rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from transformers import PreTrainedModel, PretrainedConfig
from loguru import logger


# ==================== Basic Building Blocks ====================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x: (B, C, T)
        cond: (B, cond_dim)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, C, 1)
        
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
            
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class TransformerForActionDiffusion(nn.Module):
    """Transformer decoder for action diffusion with global conditioning."""
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        global_cond_dim: int = None,
        n_layer: int = 8,
        n_head: int = 8,
        n_emb: int = 256,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()
        
        # Input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        # Project global condition to token(s)
        self.cond_proj = nn.Linear(global_cond_dim, n_emb) if global_cond_dim else None
        # Learnable position embedding for condition tokens (time + cond)
        max_cond_tokens = 2  # time token + condition token
        self.cond_pos_emb = nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )
        
        # Output head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        self.action_horizon = action_horizon
        self.n_emb = n_emb
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info(f"TransformerForActionDiffusion: n_layer={n_layer}, n_head={n_head}, n_emb={n_emb}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, sample, timestep, global_cond=None):
        """
        sample: (B, T, input_dim) action sequence
        timestep: (B,) or int
        global_cond: (B, D) observation features
        """
        B = sample.shape[0]
        
        # Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)
        time_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_emb)
        
        # Condition embedding
        if global_cond is not None and self.cond_proj is not None:
            cond_emb = self.cond_proj(global_cond).unsqueeze(1)  # (B, 1, n_emb)
            cond_emb = torch.cat([cond_emb, time_emb], dim=1)  # (B, 2, n_emb)
        else:
            cond_emb = time_emb  # (B, 1, n_emb)
        
        tc = cond_emb.shape[1]
        cond_emb = cond_emb + self.cond_pos_emb[:, :tc, :]
        
        # Input embedding
        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        t = input_emb.shape[1]
        input_emb = input_emb + self.pos_emb[:, :t, :]
        
        # Transformer decode
        x = self.decoder(tgt=input_emb, memory=cond_emb)
        x = self.ln_f(x)
        x = self.head(x)  # (B, T, output_dim)
        
        return x


class ConditionalUnet1D(nn.Module):
    """1D U-Net with global conditioning for diffusion."""
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, global_cond=None):
        """
        sample: (B, T, C) action sequence
        timestep: (B,) or int
        global_cond: (B, D) observation features
        """
        # (B, T, C) -> (B, C, T)
        sample = sample.transpose(1, 2)

        # Timestep encoding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        return x


# ==================== Vision Encoder ====================

class TimmObsEncoder(nn.Module):
    """Simplified timm-based observation encoder for ILStudio."""
    def __init__(self,
            model_name: str = 'resnet18',
            pretrained: bool = True,
            frozen: bool = False,
            feature_dim: int = 64,
            num_cameras: int = 1,
            state_dim: int = 0,
            image_size: tuple = (256, 256),
            share_rgb_model: bool = False,
            imagenet_norm: bool = True,
            feature_aggregation: str = 'spatial_softmax',
            num_kp: int = 32,
        ):
        super().__init__()
        
        self.model_name = model_name
        self.num_cameras = num_cameras
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.imagenet_norm = imagenet_norm
        
        # Determine if this is a ViT-like model
        is_vit = 'vit' in model_name.lower() or 'dino' in model_name.lower() or 'swin' in model_name.lower()
        
        # For ViT models, use max dimension of image_size (ViT expects square input)
        if is_vit:
            vit_img_size = max(image_size)
            logger.info(f"ViT model detected, using image_size={vit_img_size} (max of {image_size})")
        
        # Create backbone models
        backbones = []
        pools = []
        linears = []
        
        for cam_id in range(num_cameras):
            # Create timm model
            if 'resnet' in model_name.lower() or 'convnext' in model_name.lower():
                backbone = timm.create_model(
                    model_name, pretrained=pretrained, 
                    global_pool='', num_classes=0
                )
                # Get feature dimension
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, image_size[0], image_size[1])
                    feat = backbone(dummy)
                    backbone_dim = feat.shape[1]
                    feat_h, feat_w = feat.shape[2], feat.shape[3]
            else:
                # ViT and similar models need fixed input size
                backbone = timm.create_model(
                    model_name, pretrained=pretrained,
                    global_pool='', num_classes=0,
                    img_size=vit_img_size
                )
                backbone_dim = backbone.num_features
                # For ViT, patch size determines feature map size
                patch_size = getattr(backbone, 'patch_embed', None)
                if patch_size is not None and hasattr(patch_size, 'patch_size'):
                    ps = patch_size.patch_size[0] if isinstance(patch_size.patch_size, tuple) else patch_size.patch_size
                    feat_h = feat_w = vit_img_size // ps
                else:
                    feat_h = feat_w = vit_img_size // 14  # Default for DINOv2
            
            if frozen:
                for param in backbone.parameters():
                    param.requires_grad = False
            
            # Replace BatchNorm with GroupNorm for better small batch performance
            backbone = self._replace_bn_with_gn(backbone)
            
            if share_rgb_model and cam_id > 0:
                backbones.append(backbones[0])
            else:
                backbones.append(backbone)
            
            # Feature pooling - handle different model types
            if is_vit:
                # ViT outputs (B, N_tokens, D), use Identity as placeholder
                # Actual pooling handled in forward()
                pool = nn.Identity()
                pool_out_dim = backbone_dim
            else:
                # CNN models output (B, C, H, W)
                if feature_aggregation == 'spatial_softmax':
                    pool = SpatialSoftmax(
                        input_shape=(backbone_dim, feat_h, feat_w),
                        num_kp=num_kp
                    )
                    pool_out_dim = num_kp * 2
                elif feature_aggregation == 'avg':
                    pool = nn.AdaptiveAvgPool2d(1)
                    pool_out_dim = backbone_dim
                else:
                    pool = nn.Identity()
                    pool_out_dim = backbone_dim * feat_h * feat_w
            
            pools.append(pool)
            linears.append(nn.Linear(pool_out_dim, feature_dim))
        
        self.is_vit = is_vit
        self.vit_pool_type = feature_aggregation if is_vit else None
        
        self.backbones = nn.ModuleList(backbones)
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)
        
        # ImageNet normalization
        if imagenet_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Calculate output dimension
        self._output_dim = feature_dim * num_cameras + state_dim
        
        logger.info(f"TimmObsEncoder: {model_name}, output_dim={self._output_dim}")

    def _replace_bn_with_gn(self, module, features_per_group=16):
        """Replace BatchNorm with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = max(1, child.num_features // features_per_group)
                setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
            else:
                self._replace_bn_with_gn(child, features_per_group)
        return module

    def forward(self, obs_dict):
        """
        obs_dict: {
            'image': (B, N, C, H, W) - N cameras
            'state': (B, D) - optional state
        }
        Returns: (B, output_dim) feature vector
        """
        image = obs_dict['image']  # (B, N, C, H, W)
        B, N = image.shape[:2]
        
        features = []
        for cam_id in range(N):
            cam_img = image[:, cam_id]  # (B, C, H, W)
            
            # Normalize if needed
            if self.imagenet_norm:
                cam_img = (cam_img - self.mean) / self.std
            
            # Extract features
            feat = self.backbones[min(cam_id, len(self.backbones) - 1)](cam_img)
            
            # Handle different model types
            if self.is_vit:
                # ViT output: (B, N_tokens, D) where first token is CLS
                if self.vit_pool_type == 'cls':
                    feat = feat[:, 0]  # Use CLS token: (B, D)
                else:
                    # Mean over all tokens (or exclude CLS with feat[:, 1:])
                    feat = feat.mean(dim=1)  # (B, D)
            else:
                # CNN output: (B, C, H, W)
                feat = self.pools[min(cam_id, len(self.pools) - 1)](feat)
                feat = feat.flatten(1)
            
            feat = self.linears[min(cam_id, len(self.linears) - 1)](feat)
            features.append(feat)
        
        # Concatenate camera features
        result = torch.cat(features, dim=-1)
        
        # Add state if present
        if 'state' in obs_dict and self.state_dim > 0:
            state = obs_dict['state']
            if len(state.shape) == 3:
                state = state.squeeze(1)  # (B, 1, D) -> (B, D)
            result = torch.cat([result, state], dim=-1)
        
        return result

    def output_shape(self):
        return (self._output_dim,)


class SpatialSoftmax(nn.Module):
    """Spatial Softmax for extracting keypoints from feature maps."""
    def __init__(self, input_shape, num_kp=32, temperature=1.0):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature
        
        C, H, W = input_shape
        self.conv = nn.Conv2d(C, num_kp, kernel_size=1)
        
        # Cache for position grids (will be created dynamically)
        self._cached_pos_x = None
        self._cached_pos_y = None
        self._cached_shape = None

    def _get_pos_grids(self, H, W, device):
        """Get or create position grids for given spatial dimensions."""
        if self._cached_shape != (H, W) or self._cached_pos_x is None:
            pos_x, pos_y = torch.meshgrid(
                torch.linspace(-1, 1, W, device=device),
                torch.linspace(-1, 1, H, device=device),
                indexing='xy'
            )
            self._cached_pos_x = pos_x.reshape(1, 1, H, W)
            self._cached_pos_y = pos_y.reshape(1, 1, H, W)
            self._cached_shape = (H, W)
        return self._cached_pos_x, self._cached_pos_y

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, num_kp * 2)
        """
        # Reduce channels to num_kp
        x = self.conv(x)  # (B, num_kp, H, W)
        B, K, H, W = x.shape
        
        # Get position grids (dynamically sized)
        pos_x, pos_y = self._get_pos_grids(H, W, x.device)
        
        # Spatial softmax
        x = x.view(B, K, -1)  # (B, K, H*W)
        x = F.softmax(x / self.temperature, dim=-1)
        x = x.view(B, K, H, W)
        
        # Compute expected coordinates
        expected_x = torch.sum(x * pos_x, dim=[2, 3])  # (B, K)
        expected_y = torch.sum(x * pos_y, dim=[2, 3])  # (B, K)
        
        # Concatenate
        return torch.cat([expected_x, expected_y], dim=-1)  # (B, K*2)


# ==================== Config and Model ====================

class DiffusionUnetTimmConfig(PretrainedConfig):
    """Configuration for DiffusionUnetTimmPolicy."""
    model_type = "diffusion_unet_timm"
    
    def __init__(
        self,
        # Vision encoder
        vision_model_name: str = 'resnet18',
        vision_pretrained: bool = True,
        vision_frozen: bool = False,
        vision_feature_dim: int = 64,
        feature_aggregation: str = 'spatial_softmax',
        num_kp: int = 32,
        imagenet_norm: bool = True,
        share_rgb_model: bool = False,
        backbone_image_size: tuple = None,  # For ViT models that need fixed input size
        # Data dimensions
        action_dim: int = 7,
        state_dim: int = 7,
        num_cameras: int = 1,
        image_size: tuple = (256, 256),
        # Diffusion
        chunk_size: int = 16,  # action_horizon in original code
        num_inference_steps: int = 10,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'squaredcos_cap_v2',
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        # Backbone type: 'unet' or 'transformer'
        backbone_type: str = 'unet',
        # UNet specific
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # Transformer specific
        n_layer: int = 8,
        n_head: int = 8,
        n_emb: int = 256,
        p_drop_attn: float = 0.1,
        # Training
        input_pertub: float = 0.1,
        train_diffusion_n_samples: int = 1,
        ema_power: float = 0.75,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Vision
        self.vision_model_name = vision_model_name
        self.vision_pretrained = vision_pretrained
        self.vision_frozen = vision_frozen
        self.vision_feature_dim = vision_feature_dim
        self.feature_aggregation = feature_aggregation
        self.num_kp = num_kp
        self.imagenet_norm = imagenet_norm
        self.share_rgb_model = share_rgb_model
        self.backbone_image_size = backbone_image_size
        # Data
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_cameras = num_cameras
        self.image_size = image_size
        # Diffusion
        self.chunk_size = chunk_size
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        # Backbone type
        self.backbone_type = backbone_type
        # UNet
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        # Transformer
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.p_drop_attn = p_drop_attn
        # Training
        self.input_pertub = input_pertub
        self.train_diffusion_n_samples = train_diffusion_n_samples
        self.ema_power = ema_power


class DiffusionUnetTimmModel(PreTrainedModel):
    """Diffusion Policy with Timm vision encoder for ILStudio.
    
    Supports both UNet and Transformer backbones via config.backbone_type.
    """
    config_class = DiffusionUnetTimmConfig
    
    def __init__(self, config: DiffusionUnetTimmConfig):
        super().__init__(config)
        
        # Build observation encoder
        backbone_image_size = config.backbone_image_size
        if backbone_image_size is not None and not isinstance(backbone_image_size, tuple):
            if isinstance(backbone_image_size, int):
                backbone_image_size = (backbone_image_size, backbone_image_size)
            else:
                backbone_image_size = tuple(backbone_image_size)
        
        self.obs_encoder = TimmObsEncoder(
            model_name=config.vision_model_name,
            pretrained=config.vision_pretrained,
            frozen=config.vision_frozen,
            feature_dim=config.vision_feature_dim,
            num_cameras=config.num_cameras,
            state_dim=config.state_dim,
            image_size=tuple(config.image_size),
            share_rgb_model=config.share_rgb_model,
            imagenet_norm=config.imagenet_norm,
            feature_aggregation=config.feature_aggregation,
            num_kp=config.num_kp,
        )
        
        # Get observation feature dimension
        obs_feature_dim = np.prod(self.obs_encoder.output_shape())
        
        # Build diffusion model based on backbone_type
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'transformer':
            self.model = TransformerForActionDiffusion(
                input_dim=config.action_dim,
                output_dim=config.action_dim,
                action_horizon=config.chunk_size,
                global_cond_dim=obs_feature_dim,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_emb=config.n_emb,
                p_drop_attn=config.p_drop_attn,
            )
            logger.info(f"Using Transformer backbone: n_layer={config.n_layer}, n_head={config.n_head}, n_emb={config.n_emb}")
        else:  # Default: unet
            self.model = ConditionalUnet1D(
                input_dim=config.action_dim,
                global_cond_dim=obs_feature_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                down_dims=config.down_dims,
                kernel_size=config.kernel_size,
                n_groups=config.n_groups,
                cond_predict_scale=config.cond_predict_scale
            )
            logger.info(f"Using UNet backbone: down_dims={config.down_dims}")
        
        # Build noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
            set_alpha_to_one=True,
            steps_offset=0,
        )
        
        # EMA model
        self.ema = EMAModel(self.parameters(), power=config.ema_power)
        
        # Store config values
        self.action_dim = config.action_dim
        self.chunk_size = config.chunk_size
        self.input_pertub = config.input_pertub
        self.train_diffusion_n_samples = config.train_diffusion_n_samples
        
        logger.info(f"DiffusionUnetTimmModel initialized: backbone={config.backbone_type}, "
                   f"action_dim={config.action_dim}, chunk_size={config.chunk_size}, obs_feature_dim={obs_feature_dim}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Training: input qpos, image, actions -> loss
        Inference: input qpos, image -> action sequence
        
        Args:
            qpos: (B, D) or (B, 1, D) state/qpos
            image: (B, N, C, H, W) multi-camera images
            actions: (B, T, A) action sequence (training only)
            is_pad: (B, T) padding mask (training only)
        """
        # Ensure image is normalized to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Prepare observation dict
        obs_dict = {'image': image}
        if qpos is not None:
            if len(qpos.shape) == 3:
                qpos = qpos.squeeze(1)
            obs_dict['state'] = qpos
        
        B = image.shape[0]
        
        # Extract observation features
        obs_cond = self.obs_encoder(obs_dict)
        
        if actions is not None:  # Training mode
            # Train on multiple diffusion samples per obs if configured
            if self.train_diffusion_n_samples != 1:
                obs_cond = torch.repeat_interleave(
                    obs_cond, repeats=self.train_diffusion_n_samples, dim=0)
                actions = torch.repeat_interleave(
                    actions, repeats=self.train_diffusion_n_samples, dim=0)
                if is_pad is not None:
                    is_pad = torch.repeat_interleave(
                        is_pad, repeats=self.train_diffusion_n_samples, dim=0)
            
            # Add noise
            noise = torch.randn_like(actions)
            # Input perturbation
            noise_new = noise + self.input_pertub * torch.randn_like(actions)
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (actions.shape[0],), device=actions.device
            ).long()
            
            noisy_actions = self.noise_scheduler.add_noise(actions, noise_new, timesteps)
            
            # Predict noise
            noise_pred = self.model(noisy_actions, timesteps, global_cond=obs_cond)
            
            # Calculate loss
            if self.config.prediction_type == 'epsilon':
                target = noise
            else:
                target = actions
            
            loss = F.mse_loss(noise_pred, target, reduction='none')
            if is_pad is not None:
                loss = (loss * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum() / self.action_dim
            else:
                loss = loss.mean()
            
            return {'loss': loss}
        
        else:  # Inference mode
            noisy_action = torch.randn(
                (B, self.chunk_size, self.action_dim), 
                device=obs_cond.device, dtype=obs_cond.dtype
            )
            
            self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.model(noisy_action, t, global_cond=obs_cond)
                noisy_action = self.noise_scheduler.step(
                    noise_pred, t, noisy_action
                ).prev_sample
            
            return noisy_action

    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Dict with 'image' and 'qpos' keys
        
        Returns:
            Action predictions (B, T, A)
        """
        device = next(self.parameters()).device
        
        image = batch_obs['image'].to(device)
        qpos = batch_obs.get('qpos', batch_obs.get('state', None))
        if qpos is not None:
            qpos = qpos.to(device)
        
        with torch.no_grad():
            action = self.forward(qpos, image, None, None)
        
        return action

    def save_pretrained(self, save_directory, state_dict=None, *args, **kwargs):
        """Save model weights and configuration."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save weights
        if state_dict is None:
            state_dict = self.state_dict()
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, model_path)
        
        # Save EMA
        if self.ema is not None:
            ema_path = os.path.join(save_directory, "ema_model.bin")
            torch.save(self.ema.state_dict(), ema_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load pretrained model."""
        config = DiffusionUnetTimmConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        model = cls(config)
        
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        ema_path = os.path.join(pretrained_model_name_or_path, "ema_model.bin")
        if os.path.exists(ema_path) and model.ema is not None:
            ema_state_dict = torch.load(ema_path, map_location="cpu")
            model.ema.load_state_dict(ema_state_dict)
        
        return model

