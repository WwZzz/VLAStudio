# Optional: Export Trainer if custom training logic needed
from vlastudio.policy.trainer import BaseTrainer
from safetensors.torch import save_file, load_file
import os

from loguru import logger

class Trainer(BaseTrainer):
    """Qwen-OFT Trainer with custom loss computation."""
    
    EXTRA_FILE = "extra_trainable.safetensors"
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # Log metrics
        logging_steps = self.args.logging_steps
        if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
            log_dict = {}
            if "action_loss" in outputs:
                log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
            if log_dict:
                self.log(log_dict)
        
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with extra trainable parameters."""
        output_dir = output_dir or self.args.output_dir
        super().save_model(output_dir, _internal_call)
        self.model.config.save_pretrained(output_dir)
        
        if not self.is_world_process_zero():
            return
        
        # Get non-LoRA trainable parameters
        trainable_keys = [
            n for n, p in self.accelerator.unwrap_model(self.model).named_parameters()
            if "lora_" not in n and p.requires_grad
        ]
        
        if self.is_deepspeed_enabled:
            if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict()
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
                state_dict = clone_tensors_for_torch_save(
                    self.accelerator.unwrap_model(self.deepspeed).state_dict()
                )
        else:
            state_dict = self.model.state_dict()
        
        extra_state = {k: v for k, v in state_dict.items() if k in trainable_keys}
        if extra_state:
            os.makedirs(output_dir, exist_ok=True)
            save_file(extra_state, os.path.join(output_dir, self.EXTRA_FILE))
