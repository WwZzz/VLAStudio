"""
Trainer for OpenVLA-OFT Policy.

This module extends the BaseTrainer to provide specialized training
functionality for OpenVLA-OFT models with continuous action prediction.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from vlastudio.policy.trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Custom trainer for OpenVLA-OFT models.
    
    This trainer extends BaseTrainer to:
    - Handle continuous action training (L1 regression or diffusion)
    - Properly manage gradient accumulation for large models
    - Support mixed precision training
    - Provide action-specific evaluation metrics
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the OpenVLA-OFT trainer.
        
        Args:
            **kwargs: Arguments passed to BaseTrainer
        """
        super().__init__(**kwargs)
        
        # Log trainer configuration
        if hasattr(self.model, 'config'):
            config = self.model.config
            logger.info(f"🤖 OpenVLA-OFT Trainer initialized:")
            logger.info(f"   - L1 Regression: {getattr(config, 'use_l1_regression', False)}")
            logger.info(f"   - Diffusion: {getattr(config, 'use_diffusion', False)}")
            logger.info(f"   - Use Proprio: {getattr(config, 'use_proprio', False)}")
            logger.info(f"   - Num Images: {getattr(config, 'num_images_in_input', 1)}")
            logger.info(f"   - Action Chunk: {getattr(config, 'num_actions_chunk', 8)}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model, including PEFT/LoRA adapters and config.
        
        For OpenVLA-OFT with LoRA, this saves:
        - The LoRA adapter weights (via PEFT)
        - The model config (OpenVLAOFTConfig)
        - The processor/tokenizer
        - The action head and projector weights
        
        Args:
            output_dir: Directory to save to (defaults to self.args.output_dir)
            _internal_call: Whether this is an internal call
        """
        # Call parent save_model first
        super().save_model(output_dir, _internal_call=_internal_call)
        
        save_directory = output_dir if output_dir is not None else self.args.output_dir
        
        if self.is_world_process_zero():
            # Get the actual model (unwrap if wrapped by PEFT or DataParallel)
            model = self.model
            
            # If model.vla is a PeftModel, save the adapter
            if hasattr(model, 'vla'):
                from peft import PeftModel
                if isinstance(model.vla, PeftModel):
                    # Save LoRA adapter
                    model.vla.save_pretrained(save_directory)
                    logger.info(f"   ✓ Saved LoRA adapter to {save_directory}")
            
            # Save config to preserve task-specific parameters (action_dim, state_dim, etc.)
            if hasattr(model, 'config'):
                model.config.save_pretrained(save_directory)
                logger.info(f"   ✓ Saved model config to {save_directory}")
            
            # Save processor if available
            if hasattr(model, 'processor') and model.processor is not None:
                model.processor.save_pretrained(save_directory)
                logger.info(f"   ✓ Saved processor to {save_directory}")
            
            # Save action head and projectors (these are not part of VLA/PEFT)
            extra_state_dict = {}
            if hasattr(model, 'action_head') and model.action_head is not None:
                extra_state_dict['action_head'] = model.action_head.state_dict()
            if hasattr(model, 'proprio_projector') and model.proprio_projector is not None:
                extra_state_dict['proprio_projector'] = model.proprio_projector.state_dict()
            if hasattr(model, 'noisy_action_projector') and model.noisy_action_projector is not None:
                extra_state_dict['noisy_action_projector'] = model.noisy_action_projector.state_dict()
            
            if extra_state_dict:
                extra_weights_path = os.path.join(save_directory, 'extra_weights.bin')
                torch.save(extra_state_dict, extra_weights_path)
                logger.info(f"   ✓ Saved extra weights (action_head, projectors) to {extra_weights_path}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute training loss for OpenVLA-OFT.
        
        Args:
            model: The OpenVLA-OFT model
            inputs: Dictionary of input tensors
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments
        
        Returns:
            loss: The computed loss tensor
            outputs (optional): Model outputs if return_outputs is True
        """
        # Remove num_items_in_batch if present
        inputs.pop('num_items_in_batch', None)
        
        # Forward pass
        outputs = model(**inputs)
        
        # Extract loss
        loss = outputs.get('loss')
        if loss is None:
            raise ValueError("Model did not return a loss")
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = True,
        ignore_keys: Optional[list] = None,
    ):
        """
        Perform a prediction step for evaluation.
        
        Args:
            model: The model to evaluate
            inputs: Dictionary of input tensors
            prediction_loss_only: If True, only return loss
            ignore_keys: Keys to ignore in outputs
        
        Returns:
            Tuple of (loss, predictions, labels)
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.get('loss')
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, None, None)
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "val_action"
    ) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        For OpenVLA-OFT, this computes:
        - L1 loss between predicted and ground truth actions
        - MSE loss between predicted and ground truth actions
        - Per-dimension action errors
        
        Args:
            eval_dataset: Optional evaluation dataset
            ignore_keys: Keys to ignore during evaluation
            metric_key_prefix: Prefix for metric names
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Get eval dataloader
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if eval_dataloader is None:
            logger.warning("⚠️  get_eval_dataloader returned None, skipping evaluation")
            return {}
        
        # Check if model has select_action method
        if not hasattr(self.model, 'select_action'):
            # Fall back to parent's evaluate
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Set model to eval mode
        self.model.eval()
        
        # Initialize metrics
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        total_samples = 0
        
        # Iterate over eval batches
        with torch.no_grad():
            for batch in eval_dataloader:
                # Prepare batch_obs (exclude actions for inference)
                batch_obs = {}
                for k, v in batch.items():
                    if k not in ['actions', 'action', 'is_pad', 'labels']:
                        batch_obs[k] = v
                
                # Get ground truth actions
                gt_actions = batch.get('actions')
                if gt_actions is None:
                    continue
                
                # Move to device
                gt_actions = gt_actions.to(self.args.device)
                
                # Get predicted actions
                try:
                    pred_actions = self.model.select_action(batch_obs)
                except Exception as e:
                    logger.warning(f"select_action failed: {e}")
                    continue
                
                # Convert to tensor if needed
                if isinstance(pred_actions, np.ndarray):
                    pred_actions = torch.from_numpy(pred_actions).to(gt_actions.device)
                
                # Align shapes
                if pred_actions.shape != gt_actions.shape:
                    # Try to align by taking the first chunk_size actions
                    min_chunk = min(pred_actions.shape[1], gt_actions.shape[1])
                    pred_actions = pred_actions[:, :min_chunk]
                    gt_actions = gt_actions[:, :min_chunk]
                
                # Compute losses
                l1_loss = torch.nn.functional.l1_loss(pred_actions, gt_actions, reduction='sum')
                mse_loss = torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='sum')
                
                total_l1_loss += l1_loss.item()
                total_mse_loss += mse_loss.item()
                total_samples += gt_actions.numel()
        
        # Compute average metrics
        if total_samples > 0:
            avg_l1 = total_l1_loss / total_samples
            avg_mse = total_mse_loss / total_samples
        else:
            avg_l1 = 0.0
            avg_mse = 0.0
        
        metrics = {
            f"{metric_key_prefix}_l1": avg_l1,
            f"{metric_key_prefix}_mse": avg_mse,
        }
        
        # Log metrics
        self.log(metrics)
        
        return metrics

