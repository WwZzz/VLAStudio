import json
import math
import os
from typing import Optional

import torch
from vlastudio.policy.trainer import BaseTrainer
from torch.optim.lr_scheduler import LambdaLR


def _get_xvla_config(model):
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.get_base_model().config
    except ImportError:
        pass
    return model.config


class Trainer(BaseTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            config = _get_xvla_config(self.model)
            if hasattr(self.model, "get_optim_params"):
                params = self.model.get_optim_params()
            else:
                params = dict(self.model.named_parameters())
            assert isinstance(params, dict), "XVLA optimizer requires named parameters."
            vlm_group, soft_prompt_group, lora_group, other_group = [], [], [], []
            for name, p in params.items():
                if not p.requires_grad:
                    continue
                if "lora" in name.lower() or "adapter" in name.lower():
                    lora_group.append(p)
                elif "vlm" in name.lower():
                    vlm_group.append(p)
                elif "soft_prompt" in name.lower():
                    soft_prompt_group.append(p)
                else:
                    other_group.append(p)
            soft_prompt_lr = config.optimizer_lr * config.optimizer_soft_prompt_lr_scale
            if config.optimizer_soft_prompt_warmup_lr_scale is not None:
                soft_prompt_lr = config.optimizer_lr * config.optimizer_soft_prompt_warmup_lr_scale
            param_groups = [
                {
                    "params": vlm_group,
                    "lr": config.optimizer_lr * 0.1,
                    "weight_decay": config.optimizer_weight_decay * 0.1,
                    "name": "vlm",
                },
                {
                    "params": lora_group,
                    "lr": config.optimizer_lr,
                    "weight_decay": config.optimizer_weight_decay,
                    "name": "lora",
                },
                {
                    "params": soft_prompt_group,
                    "lr": soft_prompt_lr,
                    "weight_decay": config.optimizer_weight_decay,
                    "name": "soft_prompts",
                },
                {
                    "params": other_group,
                    "lr": config.optimizer_lr,
                    "weight_decay": config.optimizer_weight_decay,
                    "name": "other",
                },
            ]
            param_groups = [g for g in param_groups if g["params"]]
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=tuple(config.optimizer_betas),
                eps=config.optimizer_eps,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            config = _get_xvla_config(self.model)
            actual_warmup_steps = config.scheduler_warmup_steps
            actual_decay_steps = config.scheduler_decay_steps
            if num_training_steps < config.scheduler_decay_steps:
                scale_factor = num_training_steps / config.scheduler_decay_steps
                actual_warmup_steps = int(config.scheduler_warmup_steps * scale_factor)
                actual_decay_steps = num_training_steps

            def lr_lambda(current_step):
                if current_step < actual_warmup_steps:
                    if current_step <= 0:
                        return 1 / (actual_warmup_steps + 1)
                    frac = 1 - current_step / actual_warmup_steps
                    return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = config.scheduler_decay_lr / config.optimizer_lr
                return (1 - alpha) * cosine_decay + alpha

            self.lr_scheduler = LambdaLR(self.optimizer if optimizer is None else optimizer, lr_lambda, -1)
        return self.lr_scheduler

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.is_world_process_zero():
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            from peft import PeftModel
            is_peft = isinstance(self.model, PeftModel)
        except ImportError:
            is_peft = False

        if is_peft:
            print(f"Saving PEFT XVLA model to {output_dir}")
            self.model.save_pretrained(output_dir)
            base_model = self.model.get_base_model()
            base_model.config.save_pretrained(output_dir)

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            peft_config = self.model.peft_config.get("default", None)
            metadata = {
                "model_type": "peft",
                "base_model_name_or_path": (
                    peft_config.base_model_name_or_path if peft_config else "unknown"
                ),
                "modules_to_save": (
                    peft_config.modules_to_save
                    if peft_config and hasattr(peft_config, "modules_to_save")
                    else []
                ),
                "adapter_name": "default",
            }
            with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            print(f"Saving full XVLA model to {output_dir}")
            super().save_model(output_dir, _internal_call)
            metadata = {"model_type": "full"}
            with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
