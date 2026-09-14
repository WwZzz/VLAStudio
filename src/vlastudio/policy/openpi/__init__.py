from .modeling import OpenPiPolicyConfig, OpenPiPolicy
import openpi.models.tokenizer as _tokenizer
from .data_utils import OpenPiProcessor, OpenPiCollator
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, TaskType
import torch
import os
from .trainer import Trainer

def find_all_linear_names(model, lora_module=[]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any([k in name for k in lora_module]) and isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def load_model(args):
    if not args.is_training:
        import json
        checkpoint_path = args.model_name_or_path
        # Check if this is a PEFT checkpoint or a full model checkpoint
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
        # Determine checkpoint type
        is_peft_checkpoint = os.path.exists(adapter_config_path)
        if is_peft_checkpoint:
            # Case 1: PEFT checkpoint - load base model + adapter
            print(f"Loading PEFT checkpoint from {checkpoint_path}")
            
            # Read config.json to get base model configuration
            config_path = os.path.join(checkpoint_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
            
            # Load base model with the saved config
            config = OpenPiPolicyConfig.from_pretrained(checkpoint_path)
            base_model = OpenPiPolicy(config=config)
            
            # Load PEFT adapter (this will also load modules_to_save)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            print(f"Loaded base model with PEFT adapter from {checkpoint_path}")
            # Set data processor and collator
            model.model.data_processor = OpenPiProcessor(
                max_token_len=model.config.max_token_len, 
                model_action_dim=model.config.max_action_dim, 
                discrete_state_input=model.config.discrete_state_input
            )
            model.model.data_collator = OpenPiCollator()
            
        else:
            # Case 2: Full model checkpoint - direct load
            print(f"Loading full model checkpoint from {checkpoint_path}")
            model = OpenPiPolicy.from_pretrained(checkpoint_path)
            
            # Set data processor and collator
            model.data_processor = OpenPiProcessor(
                max_token_len=model.config.max_token_len, 
                model_action_dim=model.config.max_action_dim, 
                discrete_state_input=model.config.discrete_state_input
            )
            model.data_collator = OpenPiCollator()
    else:
        model_args = getattr(args, 'model_args', {})
        config = OpenPiPolicyConfig(**model_args) 
        model = OpenPiPolicy(config=config)
        if config.lora_module:
            # Collect module names (not parameter names) that need to be fully trained
            # modules_to_save should contain module names that are trainable but don't use LoRA
            modules_to_save_list = ['model.action_in_proj', 'model.action_out_proj', 'model.state_proj', 'model.action_time_mlp_in', 'model.action_time_mlp_out']
            if not config.freeze_vision_tower:
                modules_to_save_list.append('model.paligemma_with_expert.paligemma.model.vision_tower')
            # Get all module names that are the target LoRA modules
            target_modules = find_all_linear_names(model, config.lora_module)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules, 
                modules_to_save=modules_to_save_list,
            )
            model = get_peft_model(model, lora_config)
            # Set Gradient
            for n,p in model.named_parameters():
                if 'lora' in n or 'adapter' in n: continue
                elif any([k in n for k in modules_to_save_list]):
                    assert all([k not in n for k in target_modules])
                    p.requires_grad = True

            print("\nTrainable parameters summary:")
            model.print_trainable_parameters()
        elif config.freeze_vision_tower:
            for n,p in model.named_parameters():
                if 'vision_tower' in n:
                    p.requires_grad = False

        
    model.to('cuda')
    return {'model': model}

def get_data_processor(args, model_components):
    return OpenPiProcessor(max_token_len=args.max_token_len, model_action_dim=args.max_action_dim, discrete_state_input=args.discrete_state_input)

def get_data_collator(args, model_components):
    return OpenPiCollator()


