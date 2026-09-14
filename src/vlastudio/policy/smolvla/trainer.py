import transformers
import torch 
from vlastudio.policy.trainer import BaseTrainer
from transformers.optimization import get_cosine_schedule_with_warmup

def create_smolvla_scheduler(optimizer, num_training_steps, args=None):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    # SmolVLA scheduler parameters (can be overridden via args)
    num_warmup_steps = getattr(args, 'scheduler_warmup_steps', 1_000)
    num_decay_steps = getattr(args, 'scheduler_decay_steps', 30_000)
    peak_lr = getattr(args, 'scheduler_peak_lr', 1e-4)
    decay_lr = getattr(args, 'scheduler_decay_lr', 2.5e-6)
    
    # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
    actual_warmup_steps = num_warmup_steps
    actual_decay_steps = num_decay_steps
    
    if num_training_steps < num_decay_steps:
        scale_factor = num_training_steps / num_decay_steps
        actual_warmup_steps = int(num_warmup_steps * scale_factor)
        actual_decay_steps = num_training_steps
        print(f"Auto-scaling LR scheduler: warmup={actual_warmup_steps}, decay={actual_decay_steps}")
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < actual_warmup_steps:
            if current_step <= 0:
                return 1 / (actual_warmup_steps + 1)
            frac = 1 - current_step / actual_warmup_steps
            return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
        
        # Cosine decay phase
        step = min(current_step, actual_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
        alpha = decay_lr / peak_lr
        decayed = (1 - alpha) * cosine_decay + alpha
        return decayed
    
    return LambdaLR(optimizer, lr_lambda, -1)

class Trainer(BaseTrainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = create_smolvla_scheduler(
                self.optimizer if optimizer is None else optimizer,
                num_training_steps,
                args=self.args
            )
        return self.lr_scheduler