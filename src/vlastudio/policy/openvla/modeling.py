import os
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Any, List, Union
import warnings
from PIL import Image
import sys
sys.path.append(os.path.dirname(__file__))
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import LlamaTokenizerFast

IGNORE_INDEX = -100

class OpenConfig(PretrainedConfig):
    """
    Simplified configuration class for OpenVLA Policy.
    """
    def __init__(
        self,
        # Training mode parameters
        training_mode="lora",  # "lora" or "full"
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=False,
        # Model parameters
        max_length=2048,
        # Task parameters
        state_dim=14,
        action_dim=14,
        camera_names=["primary"],
        pretrained_weight_path: str="openvla/openvla-7b",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Training mode
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        
        # Model parameters
        self.max_length = max_length
        
        # Task parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.camera_names = camera_names if camera_names is not None else []
        self.pretrained_weight_path = pretrained_weight_path
        

class OpenPolicy(PreTrainedModel):
    """
    Simplified OpenVLA Policy model for robot action prediction.
    """
    config_class = OpenConfig
    
    def __init__(self, config: OpenConfig):
        super().__init__(config)
        self.config = config
        # Initialize model components (will be loaded from pretrained)
        self.processor =  AutoProcessor.from_pretrained(config.pretrained_weight_path, trust_remote_code=True)
        # Initialize tokenizer
        self.tokenizer = self.processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        # Initialize model
        quantization_config = None
        if self.config.use_quantization:
            assert self.config.training_mode == "lora", "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_quant_type="nf4"
            )
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.pretrained_weight_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid SDPA issues
        )
        
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of the OpenVLA model.
        """
        if self.model is None:
            raise ValueError("Model components not loaded. Call load_pretrained_components first.")
        # Remove num_items_in_batch if present (from trainer)
        kwargs.pop('num_items_in_batch', None)
        output = self.model(
            pixel_values=pixel_values.to(torch.bfloat16),
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        # action_logits = output.logits[:, self.model.vision_backbone.featurizer.patch_embed.num_patches : -1]
        # action_preds = action_logits.argmax(dim=2)
        # action_gt = labels[:, 1:].to(action_preds.device)
        # mask = action_gt > self.action_tokenizer.action_token_begin_idx
        # correct_preds = (action_preds == action_gt) & mask
        # action_accuracy = correct_preds.sum().float() / mask.sum().float()
        # continuous_actions_pred = torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
        # continuous_actions_gt = torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
        # action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
        return {'loss': output.loss,}
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def select_action(self, batch_obs, **kwargs) -> np.ndarray:
        """
        Predict action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Action predictions as numpy array
        """
        # The batch_obs should already contain processed inputs from data_processor
        device = next(self.model.parameters()).device
        
        # Move inputs to device
        inputs = {k: v.to(device, dtype=torch.bfloat16) if k == 'pixel_values' else v.to(device) 
                 for k, v in batch_obs.items() if isinstance(v, torch.Tensor)}
        
        bs = inputs['input_ids'].shape[0]
        
        # Ensure proper token formatting (add token 29871 if needed)
        if not torch.all(inputs['input_ids'][:, -1] == 29871):
            inputs['input_ids'] = torch.cat(
                (inputs['input_ids'], torch.full((bs, 1), 29871, dtype=torch.long, device=device)), 
                dim=1
            )
            inputs['attention_mask'] = torch.cat(
                (inputs['attention_mask'], torch.ones((bs, 1), dtype=torch.long, device=device)), 
                dim=1
            )
        
        # Split batch for generation (model processes one at a time)
        inputs_list = [{k: v[i:i+1] for k, v in inputs.items()} for i in range(bs)]
        
        # Run VLA inference
        generated_ids = [self.model.generate(**inp, max_new_tokens=self.config.action_dim) 
                        for inp in inputs_list]
        
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = [gids[0, -self.config.action_dim:].cpu().numpy() 
                                      for gids in generated_ids]
        normalized_actions = [self.action_tokenizer.decode_token_ids_to_actions(pids) 
                             for pids in predicted_action_token_ids]
        action = np.stack(normalized_actions)
        
        return action[:, np.newaxis, :]

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
