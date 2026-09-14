import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from octo.model.octo_model_pt import OctoModelPt
from octo.model.components.action_heads_pt import L1ActionHeadPt
from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt, ImageTokenizerPt, LanguageTokenizerPt
from octo.model.octo_model_pt import OctoModelPt, _np2pt
from octo.utils.spec import ModuleSpec
from typing import Tuple, List, Dict, Any
class OctoConfig(PretrainedConfig):
    def __init__(
        self, 
        jax_checkpoint: str = 'hf://rail-berkeley/octo-small-1.5',
        action_dim: int = 14,
        state_dim: int = 14,
        chunk_size: int = 50,
        use_wrist: bool = True,
        use_proprio: bool = True,
        last_hidden_dim: int = 384,
        image_size: List[int] = [256, 256],
        wrist_image_size: List[int] = [128, 128],
        num_language_tokens: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.jax_checkpoint = jax_checkpoint
        self.action_dim = action_dim
        self.action_horizon = chunk_size
        self.state_dim = state_dim
        self.use_wrist = use_wrist
        self.use_proprio = use_proprio
        self.input_dim = last_hidden_dim 
        self.image_size = image_size
        self.wrist_image_size = wrist_image_size
        self.num_language_tokens = num_language_tokens
        
class OctoPolicy(PreTrainedModel):
    config_class = OctoConfig
    
    def __init__(self, config: OctoConfig):
        super().__init__(config)       
        self.model, self.text_processor = self.init_model()
        self._eval_task = None
        
    def init_model(self): 
        meta = OctoModelPt.load_config_and_meta_from_jax(self.config.jax_checkpoint)
        text_processor = meta['text_processor']
        bs = 1
        h,w = self.config.image_size
        num_tokens_dict = {'primary': h, 'wrist': h, 'language': self.config.num_language_tokens, 'action': 1}
        pad_mask_dict = {k:torch.ones(bs,1).bool() for k in ['timestep', 'image_primary', 'image_wrist', 'proprio']}
        meta['example_batch']['observation']['image_primary'] = torch.randint(0, 255, (bs, 1, 3, h, w))
        meta['example_batch']['observation']['timestep'] = torch.randint(0, 1000, (bs, 1))
        meta['example_batch']['observation']['timestep_pad_mask'] = torch.ones(bs,1).bool()
        meta['example_batch']['observation']['task_completed'] = torch.zeros(bs, 1, self.config.action_horizon).bool()
        meta['example_batch']['task']['language_instruction']['input_ids'] = meta['example_batch']['task']['language_instruction']['input_ids'].long()
        meta['example_batch']['task']['language_instruction']['attention_mask'] = meta['example_batch']['task']['language_instruction']['attention_mask'].long()
        meta['example_batch']['task']['pad_mask_dict'] = {'language_instruction': torch.ones(bs,).bool()} 
        if 'image_primary' in meta['example_batch']['task']: meta['example_batch']['task'].pop('image_primary')
        if 'image_wrist' in meta['example_batch']['task']: meta['example_batch']['task'].pop('image_wrist')
        if 'timestep' in meta['example_batch']['task']: meta['example_batch']['task'].pop('timestep')
        meta['example_batch']['action'] = torch.randn(bs, 1, self.config.action_horizon, self.config.action_dim)
        meta['example_batch']['action_pad_mask'] = torch.ones_like(meta['example_batch']['action']).bool()
        if not self.config.use_wrist:
            del meta["config"]["model"]["observation_tokenizers"]["wrist"]
            num_tokens_dict.pop('wrist')
            del meta['example_batch']['observation']['image_wrist']
            pad_mask_dict.pop('image_wrist')
        else:
            wrist_h, wrist_w = self.config.wrist_image_size
            meta['example_batch']['observation']['image_wrist'] = torch.randint(0, 255, (bs, 1, 3, wrist_h, wrist_w))
            num_tokens_dict['wrist'] = int(wrist_h//16 * wrist_w//16)
        if self.config.use_proprio:
            meta["config"]["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
                LowdimObsTokenizerPt,
                n_bins=256,
                bin_type="normal",
                low=-2.0,
                high=2.0,
                obs_keys=["proprio"],
            )
            num_tokens_dict['proprio'] = self.config.state_dim
            meta['example_batch']['observation']['proprio'] = torch.randn(bs, 1, self.config.state_dim)
        else:
            pad_mask_dict.pop('proprio')
        # we have to explicitly specify number of tokens for proprio and observations
        meta["config"]["model"]["num_tokens_dict"] = num_tokens_dict
        meta["example_batch"]["observation"]["pad_mask_dict"] = pad_mask_dict
        
        # Fully override the old action head with a new one (for smaller changes, you can use update_config)
        meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
            L1ActionHeadPt,
            input_dim=self.config.input_dim,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            readout_key="readout_action",
        )
        # Load checkpoint
        model = OctoModelPt.from_config(
            **meta,
            verbose=True,
        )
        _, _ = model.load_weights_from_jax(
            self.config.jax_checkpoint,
            skip_keys_regex= '.*hf_model',
        )
        return model, text_processor
    
    def forward(self, observation, task, action=None,  action_pad_mask=None):
        _, head_outputs = self.model(
            observations=observation, 
            tasks=task, 
            timestep_pad_mask=observation['timestep_pad_mask'], 
            action_pad_mask=action_pad_mask,
            gt_actions=action,
            train=True, 
            verbose=False,
            save_attention_mask=True,
        )
        loss = head_outputs['action'][0]
        return {'loss': loss}
    
    def select_action(self, batch_obs, **kwargs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Action predictions
        """
        device = next(self.parameters()).device
        
        # Create task for evaluation if not already done
        if self._eval_task is None:
            # Extract raw_lang from batch_obs
            raw_lang = batch_obs.get('raw_lang', [])
            self._eval_task = self.model.create_tasks(texts=raw_lang, device=device)
        
        # Extract observation from batch
        observation = batch_obs['observation'] if 'observation' in batch_obs else batch_obs
        
        # Move observation to device
        observation = self.data_processor._pt2dev(observation, device)
        
        # Sample actions
        actions = self.model.sample_actions(
            observation, 
            self._eval_task, 
            unnormalization_statistics=None,
            generator=torch.Generator(device).manual_seed(0),    
        )
        return actions

    def reset(self):
        self._eval_task = None
        
        
        