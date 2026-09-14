"""
Qwen-OFT (OpenVLA-OFT style) Model Implementation for ILStudio.

A lightweight implementation that uses action special tokens to parallelly predict 
continuous actions conditioned on multi-view images plus a language instruction.

Key Points:
  - Supports both Qwen2.5-VL and Qwen3-VL vision-language backbones
  - Injects action special tokens into the VLM
  - Continuous action prediction via L1 regression over the action token hidden states

Reference: 
  - OpenVLA-OFT: https://github.com/moojink/openvla-oft
  - StarVLA QwenOFT implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from PIL import Image
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoProcessor

from loguru import logger


# =============================================================================
# VLM Type Detection and Loading Utilities
# =============================================================================

def detect_qwen_version(model_name_or_path: str) -> str:
    """
    Detect Qwen VLM version from model name/path.
    
    Returns:
        'qwen2.5' or 'qwen3'
    """
    model_name_lower = model_name_or_path.lower()
    if 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower:
        return 'qwen3'
    elif 'qwen2.5' in model_name_lower or 'qwen2_5' in model_name_lower or 'qwen-2.5' in model_name_lower:
        return 'qwen2.5'
    elif 'qwen2' in model_name_lower:
        return 'qwen2.5'  # Default Qwen2 series to 2.5
    else:
        # Default to Qwen2.5 for backward compatibility
        logger.warning(f"Could not detect Qwen version from '{model_name_or_path}', defaulting to Qwen2.5")
        return 'qwen2.5'


def load_qwen_vlm(model_name_or_path: str, qwen_version: str = None, use_flash_attn: bool = None, **kwargs):
    """
    Load Qwen VLM model based on version.
    
    Args:
        model_name_or_path: Path to model
        qwen_version: 'qwen2.5' or 'qwen3', auto-detected if None
        use_flash_attn: Whether to use flash_attention_2 (auto-detect if None)
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Tuple of (model, qwen_version)
    """
    if qwen_version is None:
        qwen_version = detect_qwen_version(model_name_or_path)
    
    # Auto-detect flash attention availability
    if use_flash_attn is None:
        try:
            import flash_attn
            use_flash_attn = True
            logger.info("Flash Attention 2 detected, will use it for acceleration")
        except ImportError:
            use_flash_attn = False
            logger.info("Flash Attention 2 not available, using default attention")
    
    logger.info(f"Loading Qwen VLM (version: {qwen_version}) from: {model_name_or_path}")
    
    # Build common kwargs
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        **kwargs
    }
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    
    if qwen_version == 'qwen3':
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )
        # Qwen3 hidden_size is in text_config
        if not hasattr(model.config, 'hidden_size') or model.config.hidden_size is None:
            model.config.hidden_size = model.config.text_config.hidden_size
    else:  # qwen2.5
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )
    
    return model, qwen_version


# =============================================================================
# MLP ResNet Action Head (L1 Regression)
# =============================================================================

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([
            MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)
        ])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns:
            actions: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)
        actions = x.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)


# =============================================================================
# Model Configuration
# =============================================================================

class QwenOFTConfig(PretrainedConfig):
    """
    Configuration for Qwen-OFT model.
    
    Args:
        vlm_model_name_or_path: Path to Qwen VLM model (supports Qwen2.5-VL and Qwen3-VL)
        qwen_version: VLM version ('qwen2.5' or 'qwen3'), auto-detected if None
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        chunk_size: Action prediction horizon (chunk length)
        action_hidden_dim: Hidden dimension for action head (auto-set from VLM)
        action_head_hidden_mult: Multiplier for action head hidden dimension
        action_head_num_blocks: Number of MLP ResNet blocks
        action_token: Special token for action prediction
    """
    model_type = "qwen_oft"

    def __init__(
        self,
        vlm_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        qwen_version: str = None,  # 'qwen2.5' or 'qwen3', auto-detect if None
        action_dim: int = 7,
        state_dim: int = 7,
        chunk_size: int = 16,
        action_hidden_dim: int = None,  # Will be set from VLM hidden size
        action_head_hidden_mult: int = 2,
        action_head_num_blocks: int = 2,
        action_token: str = "🔍",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_model_name_or_path = vlm_model_name_or_path
        # Auto-detect Qwen version if not specified
        self.qwen_version = qwen_version or detect_qwen_version(vlm_model_name_or_path)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_hidden_dim = action_hidden_dim
        self.action_head_hidden_mult = action_head_hidden_mult
        self.action_head_num_blocks = action_head_num_blocks
        self.action_token = action_token


# =============================================================================
# Main Model
# =============================================================================

class QwenOFTForPolicy(PreTrainedModel):
    """
    Qwen-OFT model for continuous action prediction.
    
    Supports both Qwen2.5-VL and Qwen3-VL as backbone and predicts actions 
    via L1 regression on special action token hidden states.
    """
    config_class = QwenOFTConfig

    def __init__(self, config: QwenOFTConfig):
        super().__init__(config)
        self.config = config
        
        # Load Qwen VLM backbone (auto-detect version)
        self.vlm, self.qwen_version = load_qwen_vlm(
            config.vlm_model_name_or_path,
            qwen_version=config.qwen_version
        )
        
        # Store version in config for consistency
        config.qwen_version = self.qwen_version
        
        # Get hidden size from VLM
        vlm_hidden_size = self.vlm.config.hidden_size
        if config.action_hidden_dim is None:
            config.action_hidden_dim = vlm_hidden_size
        
        # Build action head
        self.action_head = L1RegressionActionHead(
            input_dim=config.action_hidden_dim,
            hidden_dim=config.action_hidden_dim * config.action_head_hidden_mult,
            action_dim=config.action_dim,
            num_blocks=config.action_head_num_blocks,
        )
        
        # Action token setup
        self.action_token = config.action_token
        self.chunk_size = config.chunk_size
        
        # L1 loss
        self.l1_loss = nn.L1Loss()
        
        # Will be set after processor is loaded
        self.action_token_id = None
        self.multimodal_processor = None
        self.tokenizer = None
        
        logger.info(f"QwenOFT initialized with {self.qwen_version} backbone, hidden_size={vlm_hidden_size}")

    def set_processor(self, processor):
        """Set the multimodal processor and extract action token ID."""
        self.multimodal_processor = processor
        self.tokenizer = processor.tokenizer
        # Get action token ID
        self.action_token_id = self.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"][0]
        logger.info(f"Action token '{self.action_token}' -> ID: {self.action_token_id}")

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_requires_grad(self, training_args):
        """Set trainable parameters based on training arguments."""
        # Freeze vision tower if specified
        if hasattr(training_args, 'freeze_vision_tower') and training_args.freeze_vision_tower:
            for p in self.vlm.visual.parameters():
                p.requires_grad = False
            logger.info("Vision tower frozen")
        
        # Freeze language model if specified
        if hasattr(training_args, 'freeze_backbone') and training_args.freeze_backbone:
            for p in self.vlm.model.parameters():
                p.requires_grad = False
            logger.info("Language model backbone frozen")
        
        # Action head always trainable
        self.action_head.requires_grad_(True)

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id: int = None,
    ) -> torch.Tensor:
        """
        Extract action token embeddings from hidden states.
        
        Args:
            last_hidden: (B, L, H) - Last layer hidden states
            input_ids: (B, L) - Input token IDs
            action_token_id: Action token ID to match
            
        Returns:
            action_queries: (B, chunk_size, H)
        """
        if action_token_id is None:
            action_token_id = self.action_token_id
        if action_token_id is None:
            raise ValueError("action_token_id not set. Call set_processor first.")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Create mask for action tokens
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        # Check if we have enough action tokens
        counts = mask.sum(dim=1)
        if (counts < self.chunk_size).any():
            insufficient = (counts < self.chunk_size).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Samples {insufficient} have insufficient action tokens "
                f"(need {self.chunk_size}, got {counts.tolist()})"
            )

        # Get position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        # Get last chunk_size positions (topk gives largest values)
        topk_pos = masked_pos.topk(k=self.chunk_size, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values  # Sort in temporal order

        # Gather embeddings
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        
        return action_queries

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Returns:
            Dict with 'loss' and 'action_loss' keys.
        """
        # Forward through VLM
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        # Predict actions
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, self.action_token_id
            )
            pred_actions = self.action_head(action_queries)

            # Compute loss if actions provided
            if actions is not None:
                # Take last chunk_size actions as target
                if actions.dim() == 3:
                    actions_target = actions[:, -self.chunk_size:, :]
                else:
                    actions_target = actions
                
                # Apply padding mask if provided
                if is_pad is not None:
                    is_pad_chunk = is_pad[:, -self.chunk_size:]
                    # Mask out padded positions
                    valid_mask = ~is_pad_chunk
                    if valid_mask.any():
                        action_loss = self.l1_loss(
                            pred_actions[valid_mask],
                            actions_target[valid_mask]
                        )
                    else:
                        action_loss = torch.tensor(0.0, device=pred_actions.device)
                else:
                    action_loss = self.l1_loss(pred_actions, actions_target)
            else:
                action_loss = torch.tensor(0.0, device=pred_actions.device)

        return {
            'loss': action_loss,
            'action_loss': action_loss,
            'pred_actions': pred_actions,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate actions for inference.
        
        Returns:
            Dict with 'pred_actions' as numpy array.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, self.action_token_id
            )
            pred_actions = self.action_head(action_queries)

        return {'pred_actions': pred_actions.detach().cpu().numpy()}

    def select_action(self, batch_obs: Dict[str, Any]) -> np.ndarray:
        """
        Select action from observation batch (inference interface).
        
        Args:
            batch_obs: Processed and collated batch observation
            
        Returns:
            Action predictions as numpy array
        """
        # Move batch to device
        for k, v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(self.device)
        
        result = self.generate(**batch_obs)
        return result['pred_actions']

