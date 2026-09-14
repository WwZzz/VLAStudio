from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
try:
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
except:
    # For compatibility
    RTCConfig = None
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from lerobot.utils.constants import ACTION
from collections import deque
from torch import Tensor
import torch
import numpy as np

class SmolVLAPolicyConfig(PretrainedConfig):
    def __init__(self, 
            model_type="smolvla_policy",
            n_obs_steps: int = 1,
            chunk_size: int = 50,
            n_action_steps: int = 50,
            action_dim: int = 14,
            state_dim: int = 14,
            max_state_dim: int = 32,
            max_action_dim: int = 32,
            resize_imgs_with_padding: tuple[int, int] = (512, 512),
            empty_cameras: int = 0,
            adapt_to_pi_aloha: bool = False,
            use_delta_joint_actions_aloha: bool = False,
            tokenizer_max_length: int = 48,
            num_steps: int = 10,
            use_cache: bool = True,
            freeze_vision_encoder: bool = True,
            train_expert_only: bool = True,
            train_state_proj: bool = True,
            optimizer_lr: float = 1e-4,
            optimizer_betas: tuple[float, float] = (0.9, 0.95),
            optimizer_eps: float = 1e-8,
            optimizer_weight_decay: float = 1e-10,
            optimizer_grad_clip_norm: float = 10,
            scheduler_warmup_steps: int = 1_000,
            scheduler_decay_steps: int = 30_000,
            scheduler_decay_lr: float = 2.5e-6,
            vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights: bool = True,
            add_image_special_tokens: bool = False,
            attention_mode: str = "cross_attn",
            prefix_length: int = -1,
            pad_language_to: str = "longest",
            num_expert_layers: int = -1,
            num_vlm_layers: int = 16,
            self_attn_every_n_layers: int = 2,
            expert_width_multiplier: float = 0.75,
            min_period: float = 4e-3,
            max_period: float = 4.0,
            device: str = 'cuda',
            rtc_config: RTCConfig = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.empty_cameras = empty_cameras
        self.adapt_to_pi_aloha = adapt_to_pi_aloha
        self.use_delta_joint_actions_aloha = use_delta_joint_actions_aloha
        self.tokenizer_max_length = tokenizer_max_length
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_state_proj = train_state_proj
        self.vlm_model_name = vlm_model_name
        self.load_vlm_weights = load_vlm_weights
        self.add_image_special_tokens = add_image_special_tokens
        self.attention_mode = attention_mode
        self.prefix_length = prefix_length
        self.pad_language_to = pad_language_to
        self.num_expert_layers = num_expert_layers
        self.num_vlm_layers = num_vlm_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.expert_width_multiplier = expert_width_multiplier
        self.min_period = min_period
        self.max_period = max_period    
        self.device = device
        self.rtc_config = rtc_config


class SmolVLAPolicy(PreTrainedModel):
    """
    A custom model containing SmolVLM model and Policy Head.
    """
    # Associate model class with our custom configuration class
    config_class = SmolVLAPolicyConfig

    def __init__(self, config: SmolVLAPolicyConfig):
        super().__init__(config)
        self.config = config
        self.model = VLAFlowMatching(config)
        self.reset()
    
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
    
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions=None, is_pad=None, noise=None, time=None) -> dict[str, Tensor]:
        # Use self.training to determine mode: eval mode (False) -> inference, train mode (True) -> training
        if not self.training:
            actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            return {"action": actions}
        losses = self.model(images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks,
                            state=state, actions=actions, noise=noise, time=time)
        # losses = (losses * ~is_pad.unsqueeze(-1)) # cannot use any mask to denoise-based policy
        losses = losses[:, :, : self.config.max_action_dim]
        return {"loss": losses.mean()}
    
    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Action predictions
        """
        # Move batch to device
        device = next(self.parameters()).device
        for k, v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(device)
            elif isinstance(v, list):
                batch_obs[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
        
        # Forward pass
        action = self.forward(**batch_obs)['action']
        action = action[:, :, :self.model.config.action_dim]
        return action
