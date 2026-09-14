import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from .multimodal_encoder.siglip_encoder import SiglipVisionTower
from .multimodal_encoder.t5_encoder import T5Embedder
from .rdt.model import RDT
from modeling import RDTRunner
# =============================================================================
# Step 1: Create custom Config class
# =============================================================================
class RDTConfig(PretrainedConfig):
    # Custom model type name, important for auto classes like `AutoModel`
    model_type = "rdt"

    def __init__(
        self,
        pretrained_text_encoder_name_or_path: str = "",
        pretrained_vision_encoder_name_or_path: str = "",
        tokenizer_max_length: int = 512,
        # vlm_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        output_dim: int = 256,
        horizon: int = 16,
        hidden_size: int = 256,
        depth: int = 12,
        num_heads: int = 8,
        max_lang_cond_len: int = 256,
        img_cond_len: int = 256,
        lang_pos_embed_config: str = "",
        img_pos_embed_config: str = "",
        num_train_timesteps: int= 100,
        num_inference_timesteps: int = 10,
        beta_schedule: int = 0.9,
        prediction_type: str = "epsilon", # or sample
        clip_sample: str = "mlp",
        # adaptor config
        lang_adaptor: str= "mlp",
        lang_token_dim: int = 1024,
        img_adaptor: str = 'mlp',
        img_token_dim: int = 256,
        state_adaptor: str = 'mlp',
        state_token_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_text_encoder_name_or_path = pretrained_text_encoder_name_or_path
        self.pretrained_vision_encoder_name_or_path = pretrained_vision_encoder_name_or_path
        self.tokenizer_max_length = tokenizer_max_length
        self.output_dim = output_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Adaptor configurations
        self.lang_adaptor = lang_adaptor
        self.lang_token_dim = lang_token_dim
        self.img_adaptor = img_adaptor
        self.img_token_dim = img_token_dim
        self.state_adaptor = state_adaptor
        self.state_token_dim = state_token_dim



# =============================================================================
# Step 2: Create custom Model class
# =============================================================================
class RDTPolicy(PreTrainedModel):
    """
    A custom model containing Qwen2-VL model and Policy Head.
    """
    # Associate model class with our custom configuration class
    config_class = RDTConfig

    def __init__(self, config: RDTConfig):
        super().__init__(config)
        self.config = config
        # 1. Load vision encoder
        self.vision_encoder  = SiglipVisionTower.from_pretrained(config.pretrained_vision_encoder_name_or_path)
        self.image_processor = self.vision_encoder.image_processor
        # 2. Load text encoder
        text_embedder = T5Embedder(
            from_pretrained=config.pretrained_text_encoder_name_or_path, 
            model_max_length=tokenizer_max_length
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        
        # 3. Define RDT
        self.model = RDT(
            output_dim=config.action_dim,
            horizon=config.pred_horizon,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            max_lang_cond_len=config.max_lang_cond_len,
            img_cond_len=config.img_cond_len,
            lang_pos_embed_config=None,
            img_pos_embed_config=None,
            dtype=torch.bfloat16,
        )
        # 4. Define connector
        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config.lang_adaptor, 
            in_features=config.lang_token_dim, 
            out_features=config.hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config.img_adaptor, 
            in_features=config.img_token_dim, 
            out_features=config.hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config.state_adaptor, 
            in_features=config.state_token_dim * 2,    # state + state mask (indicator)
            out_features=config.hidden_size
        )
        
        # 5. Define diffusion's noise scheduler 
        # Create the noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps = config.num_train_timesteps,
            beta_schedule = config.beta_schedule,
            prediction_type = config.prediction_type,
            clip_sample = config.clip_sample,
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps = config.num_train_timesteps,
            beta_schedule = config.beta_schedule,
            prediction_type = config.prediction_type,
        )
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_timesteps = config.num_inference_timesteps
        self.prediction_type = config.prediction_type

        self.pred_horizon = config.pred_horizon
        self.action_dim = config.action_dim

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask
        return noisy_action
    
    def forward(
        self, 
        lang_tokens, 
        lang_attn_mask, 
        img_tokens, 
        state_tokens, 
        action_gt, 
        action_mask, 
        ctrl_freqs
    ) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result
        pred = self.model(state_action_traj, ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
    
    def select_action(self, obs):
        # process data
        device = next(self.parameters()).device  # Get model's device
        obs = {k:torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k,v in obs.items()}
        obs['image'] = obs['image']/255.0
        
        # TODO: Implement RDT-specific inference logic
        # This is a placeholder implementation
        # The actual implementation would depend on RDT's specific architecture
        batch_size = obs['state'].shape[0]
        action_dim = getattr(self.config, 'action_dim', 14)
        chunk_size = getattr(self.config, 'chunk_size', 50)
        
        # Placeholder: return random actions with correct shape
        action = torch.randn(batch_size, chunk_size, action_dim, device=device)
        return action