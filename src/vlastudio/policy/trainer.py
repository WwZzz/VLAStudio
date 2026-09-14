# import transformers
import re
import shutil
from pathlib import Path

from transformers.trainer import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
import numpy as np
from loguru import logger


class BaseTrainer(Trainer):
    def __init__(self, *, train_loader=None, eval_loader=None, **kwargs):
        # If no eval dataset/loader is provided, force evaluation off to avoid HF Trainer errors
        training_args = kwargs.get('args', None)
        if (
            training_args is not None
            and getattr(training_args, 'save_latest_checkpoint_only', False)
            and getattr(training_args, 'load_best_model_at_end', False)
        ):
            raise ValueError(
                "save_latest_checkpoint_only cannot be combined with "
                "load_best_model_at_end because older best checkpoints are "
                "intentionally removed"
            )
        no_eval_inputs = eval_loader is None and (kwargs.get('eval_dataset', None) is None)
        if no_eval_inputs and training_args is not None:
            if hasattr(training_args, 'eval_strategy'):
                training_args.eval_strategy = "no"
            if hasattr(training_args, 'evaluation_strategy'):
                training_args.evaluation_strategy = "no"
            if hasattr(training_args, 'do_eval'):
                training_args.do_eval = False
            # Disable best-model tracking which requires evaluation
            if hasattr(training_args, 'load_best_model_at_end'):
                training_args.load_best_model_at_end = False

        # If eval_loader is provided but eval_dataset is not, set a dummy eval_dataset
        # This is needed because Trainer checks eval_dataset in its initialization
        if eval_loader is not None and 'eval_dataset' not in kwargs:
            # Create a dummy dataset to signal we have eval data
            # Use a list to make it behave like a real dataset
            class DummyEvalDataset:
                def __len__(self):
                    # Return a reasonable length - we'll use the actual loader later
                    return 100
                def __getitem__(self, idx):
                    # This won't actually be called since we override get_eval_dataloader
                    return {}
            kwargs['eval_dataset'] = DummyEvalDataset()
        
        super().__init__(**kwargs)
        self._train_loader = train_loader
        self._eval_loader  = eval_loader

        # Log eval configuration after initialization
        if hasattr(self.args, 'do_eval') and self.args.do_eval:
            eval_strategy = getattr(self.args, 'eval_strategy', None)
            logger.info(f"🔍 Evaluation configured: do_eval={self.args.do_eval}, eval_steps={getattr(self.args, 'eval_steps', None)}, eval_strategy={eval_strategy}, eval_loader={'provided' if eval_loader is not None else 'None'}")

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        """Keep exactly the newest checkpoint when explicitly requested.

        Transformers calls this method after the new checkpoint's model,
        optimizer, scheduler, RNG state, and trainer state have been saved.
        The default path is left untouched unless the project-level switch is
        enabled.
        """
        if not getattr(self.args, 'save_latest_checkpoint_only', False):
            return super()._rotate_checkpoints(
                use_mtime=use_mtime,
                output_dir=output_dir,
            )

        checkpoint_root = Path(output_dir or self.args.output_dir).resolve()
        checkpoints = self._sorted_checkpoints(
            use_mtime=use_mtime,
            output_dir=str(checkpoint_root),
        )
        if len(checkpoints) <= 1:
            return

        checkpoint_pattern = re.compile(
            rf"{re.escape(PREFIX_CHECKPOINT_DIR)}-[0-9]+"
        )
        for checkpoint in checkpoints[:-1]:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_symlink():
                raise RuntimeError(
                    f"Refusing to delete symlinked checkpoint: {checkpoint_path}"
                )
            resolved_checkpoint = checkpoint_path.resolve()
            if (
                resolved_checkpoint.parent != checkpoint_root
                or checkpoint_pattern.fullmatch(resolved_checkpoint.name) is None
                or not resolved_checkpoint.is_dir()
            ):
                raise RuntimeError(
                    "Refusing to delete unexpected checkpoint path: "
                    f"{resolved_checkpoint}"
                )
            logger.info(
                f"Deleting older checkpoint [{resolved_checkpoint}] because "
                "save_latest_checkpoint_only=True"
            )
            try:
                shutil.rmtree(resolved_checkpoint)
            except OSError as exc:
                raise RuntimeError(
                    "Failed to delete older checkpoint after the newest "
                    f"checkpoint was saved: {resolved_checkpoint}"
                ) from exc

    def get_train_dataloader(self):
        if self._train_loader is None:
            raise ValueError("You passed train_loader=None")
        
        # Check if the underlying dataset is RLDS
        # RLDS datasets use tensorflow and don't work well with accelerator.prepare
        is_rlds = self._is_rlds_dataloader(self._train_loader)
        
        if is_rlds:
            # For RLDS datasets, return the loader without accelerator.prepare
            # This is because tensorflow datasets handle distributed training internally
            return self._train_loader
        else:
            # For regular PyTorch datasets, use accelerator.prepare
            # accelerator.prepare() might return a tuple if multiple items are passed
            # Since we're only passing one loader, extract it if it's a tuple
            prepared = self.accelerator.prepare(self._train_loader)
            if isinstance(prepared, tuple):
                prepared = prepared[0]
            return prepared
    
    def _is_rlds_dataloader(self, dataloader):
        """Check if a DataLoader wraps an RLDS dataset"""
        try:
            # Try to access the underlying dataset
            if hasattr(dataloader, 'dataset'):
                dataset = dataloader.dataset
                
                # Check for RLDS dataset indicators
                # RLDS datasets are typically wrapped in IterableDataset with dlimp.DLataset
                if hasattr(dataset, 'dataset'):
                    import dlimp as dl
                    if isinstance(dataset.dataset, dl.DLataset):
                        return True
                
                # Also check if it's directly a DLataset
                import dlimp as dl
                if isinstance(dataset, dl.DLataset):
                    return True
            
            return False
        except (ImportError, AttributeError):
            # If we can't determine, assume it's not RLDS (safer default for PyTorch)
            return False

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Get evaluation dataloader.
        
        Priority:
        1. Use self._eval_loader if it exists (passed during initialization)
        2. Fall back to parent's get_eval_dataloader if eval_dataset is provided
        3. Return None only if neither is available
        """
        # If we have an eval_loader from initialization, use it
        if self._eval_loader is not None:
            is_rlds = self._is_rlds_dataloader(self._eval_loader)
            if is_rlds:
                return self._eval_loader
            else:
                # accelerator.prepare() might return a tuple if multiple items are passed
                # Since we're only passing one loader, extract it if it's a tuple
                prepared = self.accelerator.prepare(self._eval_loader)
                if isinstance(prepared, tuple):
                    prepared = prepared[0]
                return prepared
        
        # Fall back to parent's implementation if eval_dataset is provided
        if eval_dataset is not None:
            return super().get_eval_dataloader(eval_dataset)
        
        # Return None only if we have neither eval_loader nor eval_dataset
        return None

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """
        Override to add logging and ensure evaluation runs.
        Modified to support different transformers versions dynamically using *args and **kwargs.
        """
        step = self.state.global_step if hasattr(self.state, 'global_step') else 'unknown'
        
        if hasattr(self.args, 'do_eval') and self.args.do_eval:
            eval_strategy = getattr(self.args, 'eval_strategy', None)
            if eval_strategy is None:
                eval_strategy = getattr(self.args, 'evaluation_strategy', 'no')

            if str(eval_strategy).lower() != "no":
                try:
                    self.get_eval_dataloader()
                except Exception as e:
                    logger.warning(f"⚠️  Step {step}: Error getting eval_dataloader in _maybe_log_save_evaluate: {e}")
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="val_action"):
        """
        Default evaluation method that computes MSE and MAE between predicted actions
        and ground truth actions using model.select_action.
        
        For each batch:
        1. Uses model.select_action(batch_obs) to get predicted actions
        2. Compares with batch_obs['actions'] (or batch_obs['action'])
        3. Computes MSE and MAE metrics
        """
        # Get eval dataloader
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if eval_dataloader is None:
            logger.warning("⚠️  get_eval_dataloader returned None, skipping evaluation")
            return {}
        
        # Check if model has select_action method
        if not hasattr(self.model, 'select_action'):
            # Fall back to parent's evaluate if no select_action method
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Set model to eval mode
        self.model.eval()
        
        # Initialize metrics
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        # Iterate over eval batches
        with torch.no_grad():
            for batch in eval_dataloader:
                # Prepare batch_obs (copy to avoid modifying original)
                # Exclude actions and is_pad from batch_obs for select_action (inference mode)
                batch_obs = {}
                for k, v in batch.items():
                    if k not in ['actions', 'action', 'is_pad']:
                        batch_obs[k] = v
                # Get ground truth actions
                gt_actions = batch.get('actions')
                if gt_actions is None:
                    gt_actions = batch.get('action')
                if gt_actions is None:
                    continue
                
                # For models with max_action_dim padding (like smolvla), extract only the actual action dimensions
                # gt_actions might be [B, T, max_action_dim] but we only want [B, T, action_dim]
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'action_dim'):
                    action_dim = self.model.config.action_dim
                    if gt_actions.shape[-1] > action_dim:
                        gt_actions = gt_actions[..., :action_dim]
                
                # Get is_pad mask if available (to exclude padded timesteps)
                is_pad = batch.get('is_pad')
                # Get predicted actions using select_action
                try:
                    pred_actions = self.model.select_action(batch_obs)
                except Exception as e:
                    # If select_action fails, skip this batch
                    continue
                # Convert to torch.Tensor if needed
                if isinstance(pred_actions, np.ndarray):
                    pred_actions = torch.from_numpy(pred_actions).to(gt_actions.device)
                elif not isinstance(pred_actions, torch.Tensor):
                    pred_actions = torch.tensor(pred_actions, device=gt_actions.device)
                
                # Ensure pred_actions and gt_actions are on same device
                pred_actions = pred_actions.to(gt_actions.device)
                
                # Ensure same dtype
                if pred_actions.dtype != gt_actions.dtype:
                    pred_actions = pred_actions.to(gt_actions.dtype)
                
                # Handle different shapes - align dimensions
                # pred_actions might be [B, T, A] or [B, 1, A] or [B, A]
                # gt_actions might be [B, T, A] or [B, A]
                pred_shape = pred_actions.shape
                gt_shape = gt_actions.shape
                
                # If pred_actions has extra dimension, take first timestep or squeeze
                if len(pred_shape) == 3 and len(gt_shape) == 3:
                    # Both are [B, T, A], use all timesteps
                    pass
                elif len(pred_shape) == 3 and len(gt_shape) == 2:
                    # pred is [B, T, A], gt is [B, A] - take first timestep of pred
                    pred_actions = pred_actions[:, 0, :]
                elif len(pred_shape) == 2 and len(gt_shape) == 3:
                    # pred is [B, A], gt is [B, T, A] - take first timestep of gt
                    gt_actions = gt_actions[:, 0, :]
                elif len(pred_shape) == 3 and pred_shape[1] == 1:
                    # pred is [B, 1, A], squeeze middle dimension
                    pred_actions = pred_actions.squeeze(1)
                elif len(gt_shape) == 3 and gt_shape[1] == 1:
                    # gt is [B, 1, A], squeeze middle dimension
                    gt_actions = gt_actions.squeeze(1)
                
                # Ensure final shapes match
                if pred_actions.shape != gt_actions.shape:
                    # Try to align by taking first timestep if both have time dimension
                    if len(pred_actions.shape) == 3 and len(gt_actions.shape) == 3:
                        min_t = min(pred_actions.shape[1], gt_actions.shape[1])
                        pred_actions = pred_actions[:, :min_t, :]
                        gt_actions = gt_actions[:, :min_t, :]
                    else:
                        # Skip if shapes don't match
                        continue
                
                # Ensure shapes still match
                if pred_actions.shape != gt_actions.shape:
                    # If shapes don't match, try to flatten both
                    pred_actions = pred_actions.flatten()
                    gt_actions = gt_actions.flatten()
                    if pred_actions.shape != gt_actions.shape:
                        continue
                
                # Compute MSE and MAE with reduction='none' first
                # This matches the ACT training code style
                all_mse = torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='none')
                all_mae = torch.nn.functional.l1_loss(pred_actions, gt_actions, reduction='none')
                
                # Apply is_pad mask if available (align with ACT's training code)
                if is_pad is not None:
                    # is_pad is typically [B, T] for ACT
                    if len(is_pad.shape) == 2 and len(pred_actions.shape) == 3:
                        # [B, T] and [B, T, A] - use multiplicative mask like in ACT
                        # ~is_pad: True for valid data, False for padding
                        mask = ~is_pad.unsqueeze(-1)  # [B, T, 1]
                        # Expand mask to match all_mse shape [B, T, A]
                        mask = mask.expand_as(all_mse)
                        # Count valid elements (all valid action values)
                        num_valid = mask.sum().item()
                        if num_valid > 0:
                            # Compute mean only over valid elements (not including padding in denominator)
                            mse = (all_mse * mask).sum() / num_valid
                            mae = (all_mae * mask).sum() / num_valid
                            num_action_values = num_valid
                        else:
                            # No valid elements in this batch, skip it
                            continue
                    elif len(is_pad.shape) == 1:
                        # [B] - mask entire samples
                        mask = ~is_pad  # [B]
                        if len(pred_actions.shape) == 3:
                            # Expand mask to [B, T, A]
                            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(pred_actions)
                        elif len(pred_actions.shape) == 2:
                            # Expand mask to [B, A]
                            mask = mask.unsqueeze(1).expand_as(pred_actions)
                        # Count valid elements
                        num_valid = mask.sum().item()
                        if num_valid > 0:
                            mse = (all_mse * mask).sum() / num_valid
                            mae = (all_mae * mask).sum() / num_valid
                            num_action_values = num_valid
                        else:
                            continue
                    else:
                        # Other cases: use regular mean
                        mse = all_mse.mean()
                        mae = all_mae.mean()
                        num_action_values = pred_actions.numel()
                else:
                    # No mask: use regular mean
                    mse = all_mse.mean()
                    mae = all_mae.mean()
                    num_action_values = pred_actions.numel()
                
                total_mse += mse.item() * num_action_values
                total_mae += mae.item() * num_action_values
                total_samples += num_action_values
        
        # Compute average metrics
        if total_samples > 0:
            avg_mse = total_mse / total_samples
            avg_mae = total_mae / total_samples
        else:
            avg_mse = 0.0
            avg_mae = 0.0
        
        # Return metrics in the format expected by transformers.Trainer
        metrics = {
            f"{metric_key_prefix}_mse": avg_mse,
            f"{metric_key_prefix}_mae": avg_mae,
        }
        
        # Log metrics
        self.log(metrics)
        
        return metrics
