"""
OpenVLA-OFT Policy Model for ILStudio.

This module implements the OpenVLA-OFT (Optimized Fine-Tuning) policy model,
which uses continuous action heads (L1 regression or diffusion) instead of
discrete action tokens for improved performance and efficiency.

Key features:
- Continuous action prediction via L1 regression head
- Optional diffusion-based action prediction
- Support for proprioceptive state input
- Support for multiple camera inputs
- FiLM for language-conditioned visual features
- LoRA fine-tuning support
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image

# Add openvla-oft to path
OPENVLA_OFT_PATH = os.path.join(os.path.dirname(__file__), 'openvla-oft')
if OPENVLA_OFT_PATH not in sys.path:
    sys.path.insert(0, OPENVLA_OFT_PATH)

# Import from openvla-oft
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
import prismatic.extern.hf.modeling_prismatic as modeling_prismatic_module
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.projectors import ProprioProjector, NoisyActionProjector
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.vla.action_tokenizer import ActionTokenizer
import prismatic.vla.constants as vla_constants


# Default constants - these can be overridden by config
DEFAULT_NUM_ACTIONS_CHUNK = 8
DEFAULT_ACTION_DIM = 7
DEFAULT_PROPRIO_DIM = 8
OPENVLA_IMAGE_SIZE = 224


def _set_vla_constants(action_dim: int, chunk_size: int, state_dim: int):
    """
    Dynamically set VLA constants in all relevant modules.
    This is necessary because openvla-oft's action heads and modeling modules
    use global constants that are copied at import time.
    """
    # Set in the constants module
    vla_constants.ACTION_DIM = action_dim
    vla_constants.NUM_ACTIONS_CHUNK = chunk_size
    vla_constants.PROPRIO_DIM = state_dim
    
    # Also patch the copies in modeling_prismatic module
    # These are copied at import time, so we need to update them directly
    modeling_prismatic_module.ACTION_DIM = action_dim
    modeling_prismatic_module.NUM_ACTIONS_CHUNK = chunk_size


class OpenVLAOFTConfig(PretrainedConfig):
    """
    Configuration class for OpenVLA-OFT Policy.
    """
    model_type = "openvla_oft"
    
    def __init__(
        self,
        # Model architecture
        pretrained_checkpoint: str = "openvla/openvla-7b",
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        num_diffusion_steps_train: int = 50,
        num_diffusion_steps_inference: int = 50,
        use_film: bool = False,
        num_images_in_input: int = 2,
        use_proprio: bool = True,
        center_crop: bool = True,
        # Action parameters
        action_dim: int = 7,
        state_dim: int = 8,
        chunk_size: int = 8,  # Action chunk size (open-loop steps)
        # Training mode
        training_mode: str = "lora",  # "lora" or "full"
        lora_rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        use_quantization: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        # Task parameters
        camera_names: List[str] = None,
        # Inference
        num_open_loop_steps: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Model architecture
        self.pretrained_checkpoint = pretrained_checkpoint
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.num_diffusion_steps_inference = num_diffusion_steps_inference
        self.use_film = use_film
        self.num_images_in_input = num_images_in_input
        self.use_proprio = use_proprio
        self.center_crop = center_crop
        
        # Action parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_actions_chunk = chunk_size
        self.chunk_size = chunk_size
        
        # Training mode
        self.training_mode = training_mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Task parameters
        self.camera_names = camera_names if camera_names is not None else ["primary"]
        
        # Inference
        self.num_open_loop_steps = num_open_loop_steps


class OpenVLAOFTPolicy(PreTrainedModel):
    """
    OpenVLA-OFT Policy model for continuous action prediction.
    
    This model wraps the base OpenVLA model and adds:
    - L1 regression action head for continuous action prediction
    - Optional proprioceptive state projection
    - Optional diffusion-based action generation
    - FiLM for language-conditioned visual features
    """
    config_class = OpenVLAOFTConfig
    
    def __init__(self, config: OpenVLAOFTConfig):
        super().__init__(config)
        self.config = config
        
        # CRITICAL: Set VLA constants BEFORE importing action heads
        # openvla-oft's action heads use global constants from prismatic.vla.constants
        # These constants are read at module import time, so we must set them first
        # and force reload the action_heads module to pick up the new values
        _set_vla_constants(
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            state_dim=config.state_dim,
        )
        
        # Force reload action_heads module to pick up new constants
        import importlib
        import prismatic.models.action_heads as action_heads_module
        importlib.reload(action_heads_module)
        L1RegressionActionHead = action_heads_module.L1RegressionActionHead
        DiffusionActionHead = action_heads_module.DiffusionActionHead
        
        # Load processor directly using openvla-oft's PrismaticProcessor
        # We bypass Auto classes to avoid conflicts with policy/openvla's registration
        self.processor = PrismaticProcessor.from_pretrained(
            config.pretrained_checkpoint, 
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        
        # Load base VLA model directly using openvla-oft's OpenVLAForActionPrediction
        # This ensures we get the extended PrismaticVisionBackbone with set_num_images_in_input,
        # proprio support, and other openvla-oft specific features
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            config.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid SDPA issues
        )
        
        # Set number of images in model input
        # Using openvla-oft's extended PrismaticVisionBackbone
        self.vla.vision_backbone.set_num_images_in_input(config.num_images_in_input)
        
        # Get LLM dimension for projectors and action heads
        self.llm_dim = self.vla.llm_dim
        
        # Initialize action head with correct dimensions
        # Note: The action heads internally use vla_constants (ACTION_DIM, NUM_ACTIONS_CHUNK)
        # for reshape operations, which we've set above via _set_vla_constants
        # Input: [batch, chunk_size * action_dim, llm_dim] -> reshape to [batch, chunk_size, action_dim * llm_dim]
        # Output: [batch, chunk_size, action_dim]
        self.action_head = None
        if config.use_l1_regression:
            self.action_head = L1RegressionActionHead(
                input_dim=self.llm_dim,
                hidden_dim=self.llm_dim,
                action_dim=config.action_dim,  # Per-timestep action dimension
            ).to(torch.bfloat16)
        elif config.use_diffusion:
            self.action_head = DiffusionActionHead(
                input_dim=self.llm_dim,
                hidden_dim=self.llm_dim,
                action_dim=config.action_dim,  # Per-timestep action dimension
                num_diffusion_steps_train=config.num_diffusion_steps_train,
            ).to(torch.bfloat16)
        
        # Initialize proprio projector if using proprioceptive state
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=self.llm_dim,
                proprio_dim=config.state_dim,
            ).to(torch.bfloat16)
        
        # Initialize noisy action projector if using diffusion
        self.noisy_action_projector = None
        if config.use_diffusion:
            self.noisy_action_projector = NoisyActionProjector(llm_dim=self.llm_dim).to(torch.bfloat16)
        
        # FiLM setup
        if config.use_film:
            self.vla.vision_backbone = FiLMedPrismaticVisionBackbone(
                vision_backbone=self.vla.vision_backbone,
                llm_dim=self.llm_dim,
            )
        
        # Note: Normalization is handled externally by ILStudio's unified normalization pipeline
        # Do NOT add internal normalization here
    
    def _get_num_patches(self) -> int:
        """Get total number of vision patches including additional embeddings."""
        num_patches = (
            self.vla.vision_backbone.get_num_patches() 
            * self.vla.vision_backbone.get_num_images_in_input()
        )
        if self.config.use_proprio:
            num_patches += 1  # Add proprio embedding
        if self.config.use_diffusion:
            num_patches += 1  # Add diffusion timestep embedding
        return num_patches
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            pixel_values: Image inputs [B, num_images, C, H, W]
            input_ids: Tokenized text inputs [B, seq_len]
            labels: Target token IDs for loss computation [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            proprio: Proprioceptive state [B, proprio_dim]
            actions: Ground truth actions [B, chunk_size, action_dim]
        
        Returns:
            Dictionary containing 'loss' and optionally other metrics
        """
        # Remove num_items_in_batch if present (from trainer)
        kwargs.pop('num_items_in_batch', None)
        
        device = pixel_values.device
        batch_size = pixel_values.shape[0]
        
        # Prepare diffusion inputs if using diffusion
        noisy_actions = None
        diffusion_timestep_embeddings = None
        if self.config.use_diffusion and actions is not None:
            noisy_dict = self.action_head.sample_noisy_actions(actions.to(device).to(torch.bfloat16))
            noise = noisy_dict["noise"]
            noisy_actions = noisy_dict["noisy_actions"]
            diffusion_timestep_embeddings = noisy_dict["diffusion_timestep_embeddings"]
        
        # Proprio is already normalized by ILStudio's external normalization pipeline
        proprio_input = None
        if proprio is not None and self.config.use_proprio:
            proprio_input = proprio.to(device).to(torch.bfloat16)
        
        # VLA forward pass using openvla-oft's extended forward with proprio/diffusion support
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                pixel_values=pixel_values.to(torch.bfloat16).to(device),
                labels=labels,
                output_hidden_states=True,
                proprio=proprio_input if self.config.use_proprio else None,
                proprio_projector=self.proprio_projector if self.config.use_proprio else None,
                noisy_actions=noisy_actions if self.config.use_diffusion else None,
                noisy_action_projector=self.noisy_action_projector if self.config.use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if self.config.use_diffusion else None,
                use_film=self.config.use_film,
            )
        
        # Compute loss using continuous action head
        if (self.config.use_l1_regression or self.config.use_diffusion) and actions is not None:
            num_patches = self._get_num_patches()
            ground_truth_actions = actions.to(device).to(torch.bfloat16)
            
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            
            # Get hidden states for text portion (after vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            
            # Get action masks
            ground_truth_token_ids = labels[:, 1:].to(device) if labels is not None else None
            
            # Extract action hidden states
            # For simplicity, we take the last chunk_size * action_dim hidden states
            num_action_tokens = self.config.num_actions_chunk * self.config.action_dim
            actions_hidden_states = text_hidden_states[:, -num_action_tokens:].to(torch.bfloat16)
            
            if self.config.use_l1_regression:
                predicted_actions = self.action_head.predict_action(actions_hidden_states)
                loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
            elif self.config.use_diffusion:
                noise_pred = self.action_head.predict_noise(actions_hidden_states)
                noise_pred = noise_pred.reshape(noise.shape)
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            
            return {'loss': loss}
        
        # Fall back to standard cross-entropy loss if no continuous action head
        return {'loss': output.loss}
    
    def get_input_embeddings(self):
        return self.vla.get_input_embeddings()
    
    @torch.inference_mode()
    def select_action(self, batch_obs: Dict[str, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Predict action from processed batch observation.
        
        Args:
            batch_obs: Dictionary containing:
                - pixel_values: Image inputs [B, C, H, W] or [B, num_images*C, H, W]
                - input_ids: Tokenized text [B, seq_len]
                - attention_mask: Attention mask [B, seq_len]
                - proprio (optional): Proprioceptive state [B, proprio_dim]
        
        Returns:
            Action predictions as numpy array [B, chunk_size, action_dim]
        """
        # CRITICAL: Ensure VLA constants are set correctly for this config
        # This patches both vla_constants and modeling_prismatic module-level copies
        _set_vla_constants(
            action_dim=self.config.action_dim,
            chunk_size=self.config.chunk_size,
            state_dim=self.config.state_dim,
        )
        
        device = next(self.vla.parameters()).device
        
        # Move inputs to device
        pixel_values = batch_obs['pixel_values'].to(device, dtype=torch.bfloat16)
        input_ids = batch_obs['input_ids'].to(device)
        attention_mask = batch_obs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids)
        
        batch_size = pixel_values.shape[0]
        
        # Handle proprio - already normalized by ILStudio's external normalization pipeline
        proprio = None
        if self.config.use_proprio and 'proprio' in batch_obs:
            proprio = batch_obs['proprio'].to(device).to(torch.bfloat16)
        
        # Use VLA's predict_action method for continuous action prediction
        if self.config.use_l1_regression or self.config.use_diffusion:
            # Process each sample in the batch
            all_actions = []
            
            for i in range(batch_size):
                sample_input_ids = input_ids[i:i+1]
                sample_attention_mask = attention_mask[i:i+1]
                sample_pixel_values = pixel_values[i:i+1]
                sample_proprio = proprio[i:i+1] if proprio is not None else None
                
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Use VLA's predict_action which handles action token preparation
                    # skip_unnorm=True because ILStudio handles normalization externally
                    normalized_actions, _ = self.vla.predict_action(
                        input_ids=sample_input_ids,
                        pixel_values=sample_pixel_values,
                        attention_mask=sample_attention_mask,
                        proprio=sample_proprio.float().cpu().numpy() if sample_proprio is not None else None,
                        proprio_projector=self.proprio_projector if self.config.use_proprio else None,
                        action_head=self.action_head,
                        noisy_action_projector=self.noisy_action_projector if self.config.use_diffusion else None,
                        use_film=self.config.use_film,
                        skip_unnorm=True,
                    )
                
                # Convert to numpy float32 if tensor
                if isinstance(normalized_actions, torch.Tensor):
                    normalized_actions = normalized_actions.float().cpu().numpy()
                elif isinstance(normalized_actions, np.ndarray):
                    normalized_actions = normalized_actions.astype(np.float32)
                
                all_actions.append(normalized_actions)
            
            # Stack actions: shape [B, chunk_size, action_dim]
            actions = np.stack(all_actions, axis=0).astype(np.float32)
        else:
            # Fall back to discrete token prediction
            generated_ids = self.vla.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=self.config.action_dim,
            )
            predicted_token_ids = generated_ids[:, -self.config.action_dim:].cpu().numpy()
            actions = np.array([
                self.action_tokenizer.decode_token_ids_to_actions(pids) 
                for pids in predicted_token_ids
            ])
            # Reshape to [B, 1, action_dim] since discrete mode doesn't produce chunks
            actions = actions[:, np.newaxis, :]
        
        # Note: Action unnormalization is handled by ILStudio's external normalization pipeline
        return actions
    
    def _run_diffusion_sampling(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run diffusion sampling to generate actions."""
        device = pixel_values.device
        batch_size = pixel_values.shape[0]
        
        # Sample random noise as starting point
        noise = torch.randn(
            size=(batch_size, self.config.num_actions_chunk, self.config.action_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Set diffusion timesteps
        self.action_head.noise_scheduler.set_timesteps(self.config.num_diffusion_steps_inference)
        
        # Reverse diffusion
        curr_noisy_actions = noise
        num_patches = self._get_num_patches()
        
        for t in self.action_head.noise_scheduler.timesteps:
            timesteps = torch.tensor([t], device=device).repeat(batch_size)
            diffusion_timestep_embeddings = (
                self.action_head.time_encoder(timesteps)
                .to(curr_noisy_actions.dtype)
                .unsqueeze(1)
            )
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=None,
                    output_hidden_states=True,
                    proprio=proprio if self.config.use_proprio else None,
                    proprio_projector=self.proprio_projector if self.config.use_proprio else None,
                    noisy_actions=curr_noisy_actions,
                    noisy_action_projector=self.noisy_action_projector,
                    diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                    use_film=self.config.use_film,
                )
                
                last_hidden_states = output.hidden_states[-1]
                text_hidden_states = last_hidden_states[:, num_patches:-1]
                num_action_tokens = self.config.num_actions_chunk * self.config.action_dim
                actions_hidden_states = text_hidden_states[:, -num_action_tokens:].to(torch.bfloat16)
                noise_pred = self.action_head.predict_noise(actions_hidden_states)
            
            curr_noisy_actions = self.action_head.noise_scheduler.step(
                noise_pred, t, curr_noisy_actions
            ).prev_sample
        
        return curr_noisy_actions

