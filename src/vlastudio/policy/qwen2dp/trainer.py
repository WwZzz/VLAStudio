import transformers.trainer 
import os
import json
from functools import partial
from peft import PeftModel, PeftConfig
import warnings
from safetensors.torch import save_file, load_file
import re
import torch
from typing import Optional
from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)
from vlastudio.policy.trainer import BaseTrainer

# Ignore duplicate UserWarnings, only show specific warnings once
warnings.simplefilter("once", UserWarning)

def _lora_decay_fn(name, param, decay_param_names):
    return ".lora_" in name and name in decay_param_names

def _lora_nodecay_fn(name, param, decay_param_names):
    return ".lora_" in name and name not in decay_param_names

def _full_decay_fn(name, param, decay_param_names):
    return ".lora_" not in name and name in decay_param_names

def _full_nodecay_fn(name, param, decay_param_names):
    return ".lora_" not in name and name not in decay_param_names


class Trainer(BaseTrainer):
    
    EXTRA_FILE = "extra_trainable.safetensors"
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        logging_steps = self.args.logging_steps
        if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
            log_dict = {}
            if "action_loss" in outputs:
                log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
            if "llm_loss" in outputs:
                log_dict["llm_loss"] = outputs["llm_loss"].detach().cpu().item()
            if log_dict:
                self.log(log_dict)
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call=False):
        """override: Save PEFT first, then save non-PEFT trainable parameters"""
        output_dir = output_dir or self.args.output_dir
        super().save_model(output_dir, _internal_call)
        self.model.config.save_pretrained(output_dir)
        if not self.is_world_process_zero(): return
        trainable_keys = [n for n,p in self.accelerator.unwrap_model(self.model).named_parameters() if "lora_" not in n and p.requires_grad]
        if self.is_deepspeed_enabled:
            if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                # ZeRO-3: Must first enable "stage3_gather_16bit_weights_on_model_save": true in ds_config
                state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict()
            else:
                # ZeRO-0/1/2: Weights are complete on each card, just clone
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
                state_dict = clone_tensors_for_torch_save(
                    self.accelerator.unwrap_model(self.deepspeed).state_dict()
                )
        else:
            # Non-DeepSpeed: Get directly
            state_dict = self.model.state_dict()
        # 4. Filter out parameters that are "non-LoRA and requires_grad=True"
        extra_state = {
            k: v for k, v in state_dict.items()
            if k in trainable_keys
        }
        if extra_state:
            os.makedirs(output_dir, exist_ok=True)
            save_file(extra_state, os.path.join(output_dir, self.EXTRA_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """override: Load PEFT first, then load non-PEFT weights"""
        super()._load_from_checkpoint(resume_from_checkpoint)
        extra_path = os.path.join(resume_from_checkpoint, self.EXTRA_FILE)
        trainable_keys = [n for n,p in self.accelerator.unwrap_model(self.model).named_parameters() if "lora_" not in n and p.requires_grad]
        if os.path.exists(extra_path):
            extra_state = load_file(extra_path, device="cpu")
            missing, unexpected = self.accelerator.unwrap_model(self.model).load_state_dict(extra_state, strict=False)
            missing = [k for k in missing if k in trainable_keys]
            if missing:
                self.control.should_training_stop = True
                raise RuntimeError(f"Missing non-PEFT keys: {missing}")
    
    def create_optimizer(self):
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        if self.optimizer is not None: return self.optimizer
        # --------------- 1. Collect all trainable parameters ---------------
        decay_param_names = set(
            get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        )
        decay_param_names = {n for n in decay_param_names if "bias" not in n} # bias does not decay

        lora_decay = []
        lora_nodecay = []
        full_decay = []
        full_nodecay = []

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_lora = ".lora_" in name                # Match LoRA
            is_no_decay = name not in decay_param_names
            if is_lora:
                (lora_nodecay if is_no_decay else lora_decay).append(param)
            else:
                (full_nodecay if is_no_decay else full_decay).append(param)

        # --------------- 2. Construct param_groups ---------------
        lora_lr = getattr(self.args, "lora_lr", 1e-4)
        full_lr = getattr(self.args, "learning_rate", 1e-4)

        param_groups = []
        if lora_decay:
            param_groups.append(
                {"params": lora_decay, "lr": lora_lr, "weight_decay": self.args.weight_decay}
            )
        if lora_nodecay:
            param_groups.append(
                {"params": lora_nodecay, "lr": lora_lr, "weight_decay": 0.0}
            )
        if full_decay:
            param_groups.append(
                {"params": full_decay, "lr": full_lr, "weight_decay": self.args.weight_decay}
            )
        if full_nodecay:
            param_groups.append(
                {"params": full_nodecay, "lr": full_lr, "weight_decay": 0.0}
            )

        if not param_groups:
            raise ValueError("No trainable parameters found!")

        # --------------- 3. DeepSpeed injection ---------------
        if self.is_deepspeed_enabled:
            # Write grouping information directly into deepspeed_config
            ds_config = self.args.deepspeed_config if hasattr(self.args, 'deepspeed_config') else self.args.deepspeed
            if isinstance(ds_config, str):
                with open(ds_config, "r", encoding="utf-8") as f:
                    ds_config = json.load(f)

            lora_re = re.compile(r"\.lora_")
            
            ds_groups = []
            if lora_decay:
                ds_groups.append({
                    "name": "lora_decay",
                    "params": {"lr": lora_lr, "weight_decay": self.args.weight_decay},
                    "params_fn": partial(_lora_decay_fn, decay_param_names=decay_param_names),
                })
            if lora_nodecay:
                ds_groups.append({
                    "name": "lora_nodecay",
                    "params": {"lr": lora_lr, "weight_decay": 0.0},
                    "params_fn": partial(_lora_nodecay_fn, decay_param_names=decay_param_names),
                })
            if full_decay:
                ds_groups.append({
                    "name": "full_decay",
                    "params": {"lr": full_lr, "weight_decay": self.args.weight_decay},
                    "params_fn": partial(_full_decay_fn, decay_param_names=decay_param_names),
                })
            if full_nodecay:
                ds_groups.append({
                    "name": "full_nodecay",
                    "params": {"lr": full_lr, "weight_decay": 0.0},
                    "params_fn": partial(_full_nodecay_fn, decay_param_names=decay_param_names),
                })

            ds_config.setdefault("optimizer", {})
            ds_config["optimizer"]["param_groups"] = ds_groups
            self.args.deepspeed_config = ds_config

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        optimizer_kwargs.pop("lr", None)  # Already specified in groups
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer