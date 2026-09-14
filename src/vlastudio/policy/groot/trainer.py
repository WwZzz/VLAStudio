from safetensors.torch import save_file
import os
from vlastudio.policy.trainer import BaseTrainer

class Trainer(BaseTrainer):
    """GR00T Trainer with custom loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        # Log metrics
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model checkpoint."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        self.model.config.save_pretrained(output_dir)
        
        # Save model weights
        if self.is_world_process_zero():
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
            
            # Clone tensors to avoid shared memory issue with safetensors
            # GR00T's Eagle model shares embed_tokens.weight and lm_head.weight
            state_dict = {k: v.clone() for k, v in state_dict.items()}
            save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
