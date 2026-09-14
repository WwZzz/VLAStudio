import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from .policy import ConditionalUnet1D
from .data_utils import Qwen2VLAProcess, Qwen2VLADataCollatorForSupervisedDataset
from .fusion import ActionProjector, FiLM
# =============================================================================
# Step 1: Create custom Config class
# =============================================================================
class QwenVLPolicyConfig(PretrainedConfig):
    """
    This is our custom model configuration class. It inherits from PretrainedConfig,
    making it compatible with the Hugging Face ecosystem.
    Parameters:
        vlm_model_name_or_path (str, optional):
            Path to base VLM model or Hugging Face Hub name.
            For example: "Qwen/Qwen2-VL-1.5B-Instruct".
        policy_input_size (int, optional):
            Input dimension of Policy Head. This is usually the same as VLM's hidden_size.
        policy_hidden_size (int, optional):
            Dimension of Policy Head's intermediate hidden layer.
        policy_output_size (int, optional):
            Final output dimension of Policy Head.
        **kwargs:
            Other parameters passed to parent class PretrainedConfig.
    """
    # Custom model type name, important for auto classes like `AutoModel`
    model_type = "qwen_vl_with_policy_head"

    def __init__(
        self,
        vlm_model_name_or_path: str = "Qwen/Qwen2-VL-1.5B-Instruct",
        policy_action_dim: int = 7,
        policy_state_dim: int = 7,
        policy_cond_dim: int = 1536,
        policy_prediction_horizon: int = 16,
        policy_diffusion_step_embed_dim: int = 256,
        policy_down_dims: List[int] = [256, 512, 1024],
        policy_kernel_size: int = 5,
        policy_n_groups: int = 8,
        policy_noise_samples: int = 1,
        policy_num_inference_timesteps: int = 10,
        policy_num_train_timesteps: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_model_name_or_path = vlm_model_name_or_path
        self.policy_action_dim = policy_action_dim
        self.policy_cond_dim = policy_cond_dim
        self.policy_diffusion_step_embed_dim = policy_diffusion_step_embed_dim
        self.policy_down_dims = policy_down_dims
        self.policy_kernel_size = policy_kernel_size
        self.policy_n_groups = policy_n_groups
        self.policy_state_dim = policy_state_dim
        self.policy_prediction_horizon = policy_prediction_horizon
        self.policy_noise_samples = policy_noise_samples
        self.policy_num_inference_timesteps = policy_num_inference_timesteps
        self.policy_num_train_timesteps = policy_num_train_timesteps



# =============================================================================
# Step 2: Create custom Model class
# =============================================================================
class QwenVLForPolicy(PreTrainedModel):
    """
    A custom model containing Qwen2-VL model and Policy Head.
    """
    # Associate model class with our custom configuration class
    config_class = QwenVLPolicyConfig

    def __init__(self, config: QwenVLPolicyConfig):
        super().__init__(config)
        self.config = config
        # 1. Load VLM model
        # Use AutoModelForCausalLM for loading, more general and robust
        print(f"Initializing VLM from base path: {config.vlm_model_name_or_path}")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name_or_path,
            torch_dtype=torch.bfloat16, # Automatically select appropriate precision
            trust_remote_code=True # Qwen-VL model requires this parameter
        )
        # 2. Define Policy Head
        policy_config_dict = {k[7:]:v for k,v in self.config.to_dict().items() if 'policy_' in k}
        self.policy_head = ConditionalUnet1D(**policy_config_dict)
        
        # 3. Define fusion module
        config.hidden_size = config.policy_cond_dim
        self.action_proj = ActionProjector(config.hidden_size, config.hidden_size)
        self.reasoning_proj = ActionProjector(config.hidden_size, config.hidden_size)
        self.reasoning_film = FiLM(feature_dim=config.hidden_size, condition_dim=config.hidden_size)
    
    def set_requires_grad(self, training_args):
        if not training_args.lora_enable:
            # Set vision model freezing
            self.vlm.visual.requires_grad_(True) # set to true first
            self.config.freeze_vision_tower = training_args.freeze_vision_tower
            if training_args.freeze_vision_tower:
                for n,p in self.vlm.visual.named_parameters():
                    p.requires_grad = False
            if not training_args.freeze_backbone:# Try to set fine-tuning lm_head
                try:
                    self.vlm.lm_head.requires_grad_(True) 
                except Exception as e:
                    print(e)
        # Set policy_head to require gradients
        self.policy_head.requires_grad_(True)
        self.action_proj.requires_grad_(True)
        self.reasoning_proj.requires_grad_(True)
        self.reasoning_film.requires_grad_(True)
    
    def film_forward(self, labels, input_ids, hidden_states):
        """
        Perform the forward pass for the film module.
        """
        inputs_index = labels[:, :] == -100
        inputs_index = inputs_index.int()

        xor_array = torch.bitwise_xor(inputs_index[:, :-1], inputs_index[:, 1:])
        indexs = torch.argmax((xor_array != 0).float(), dim=1)
        input_embeddings = []
        reasoning_embeddings = []
        identity = []
        for i in range(indexs.shape[0]):
            end = indexs[i] + 1
            temp = input_ids[i] == 151643  # pad token id for qwen2_vl, because qwen2_vl uses left padding
            start = sum(temp.int()) # Remove padding
            input_embeddings.append(self.action_proj(hidden_states[i, start:end, :]))
            identity.append(torch.mean(hidden_states[i, start:end, :], dim=0))
            reasoning_embeddings.append(self.reasoning_proj(hidden_states[i, end:, :]))
        input_embeddings = torch.cat(input_embeddings, dim=0)
        reasoning_embeddings = torch.cat(reasoning_embeddings, dim=0)
        identity = torch.stack(identity)
        action_conds = self.reasoning_film(input_embeddings, reasoning_embeddings).unsqueeze(1)
        action_conds = action_conds + identity.unsqueeze(1)
        return action_conds
    
    def forward(
        self,
        # QwenVL Input
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Policy Input
        actions: Optional[torch.LongTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        is_pad: bool = False,
        **kwargs,
    ):
        """
        Forward propagation logic.

        Returns:
            A dictionary containing policy_logits.
        """
        # Pass input to VLM
        # We need the last layer hidden states from VLM
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True,  # Ensure VLM returns hidden_states
            # return_dict=True,
        )
        # Extract last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        # Use film to extract subtask-related information
        action_conds = self.film_forward(labels=labels, input_ids=input_ids, hidden_states=hidden_states)
        policy_outputs = self.policy_head(actions, action_conds, states, is_pad)
        # Calculate action loss
        return {
            'llm_loss': outputs['loss'],
            'action_loss': policy_outputs['loss'],
            'loss': outputs['loss'] + policy_outputs['loss'],
        }

    
    @torch.no_grad()
    def generate(self,
        # QwenVL Input
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Policy Input
        actions: Optional[torch.LongTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        is_pad: bool = False,
        **kwargs,
    ):
        outputs = self.vlm.generate(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            num_beams=1,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=self.tokenizer.pad_token_id,  # Pad token
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        # get hidden states
        last_hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
        # get token ids 
        output_ids = outputs.sequences
        # constuct labels
        bs = states.shape[0]
        labels_input = torch.ones((bs, input_ids.shape[1])) * (-100)
        labels_output = torch.ones((bs, output_ids.shape[1] - input_ids.shape[1]))
        labels = torch.cat([labels_input, labels_output], dim=1)
        action_conds = self.film_forward(labels=labels, input_ids=output_ids, hidden_states=last_hidden_states)
        policy_outputs = self.policy_head(actions, action_conds, states, is_pad)
        return policy_outputs    
        
    
    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Action predictions
        """
        # Move batch to device
        for k, v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(self.device)
        
        # Ensure states are in bfloat16
        if 'states' in batch_obs:
            batch_obs['states'] = batch_obs['states'].to(dtype=torch.bfloat16)
        
        # Generate actions
        action = self.generate(**batch_obs)
        return action

