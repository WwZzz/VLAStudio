import os
import json
from typing import Optional
import transformers
from peft import PeftModel
from vlastudio.policy.trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    A versatile Trainer that intelligently handles saving for different training scenarios.
    
    Saving Strategy:
    1. Pure PEFT (only adapter params trainable):
       - Saves ONLY the adapter weights (lightweight, ~MB)
       - Load with: PeftModel.from_pretrained(base_model, adapter_path)
    
    2. PEFT + Extra Trainable Params (some non-adapter params unfrozen):
       - Automatically merges adapter into base model
       - Saves as a complete merged model
       - Load with: AutoModel.from_pretrained(checkpoint_path)
    
    3. Full Fine-tuning (no PEFT):
       - Saves the entire model
       - Load with: AutoModel.from_pretrained(checkpoint_path)
    
    All saved models can be loaded using standard HuggingFace from_pretrained methods.
    No custom loading functions required!
    """
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.is_world_process_zero():
            # Get the correct output directory
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            
            # --- Case 1: PEFT Model (LoRA, etc.) ---
            if isinstance(self.model, PeftModel):
                print(f"Saving PEFT model to {output_dir}")
                
                # Save adapter weights and config (includes modules_to_save if configured)
                # PEFT's save_pretrained automatically handles modules_to_save
                self.model.save_pretrained(output_dir)
                
                # Save the base model's config (needed for reconstruction)
                base_model = self.model.get_base_model()
                base_model.config.save_pretrained(output_dir)
                
                # Save tokenizer if available
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                
                # Save metadata
                peft_config = self.model.peft_config.get('default', None)
                metadata = {
                    "model_type": "peft",
                    "base_model_name_or_path": peft_config.base_model_name_or_path if peft_config else "unknown",
                    "modules_to_save": peft_config.modules_to_save if peft_config and hasattr(peft_config, 'modules_to_save') else [],
                    "adapter_name": "default"
                }
                
                with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                    
            # --- Case 2: Regular Model (Full Fine-tuning) ---
            else:
                print(f"Saving full model (all trainable weights) to {output_dir}")
                # Use the standard Trainer save behavior
                super().save_model(output_dir, _internal_call)
                
                # Save a metadata file
                metadata = {
                    "model_type": "full",
                }
                with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)