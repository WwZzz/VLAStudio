import transformers
import torch 
from vlastudio.policy.trainer import BaseTrainer

class Trainer(BaseTrainer):
    def create_optimizer(self):
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.learning_rate*0.1,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.args.learning_rate, 
            betas=[self.args.adam_beta1, self.args.adam_beta2], 
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )
        return self.optimizer
    
    def training_step(self, *args, **kwargs):
        """Extended training step, update EMA parameters"""
        loss = super().training_step(*args, **kwargs)
        if hasattr(self.model, "ema") and self.model.ema is not None:
            self.model.ema.step(self.model.parameters())  # Use model's `ema` object to update weights
        return loss
    
    def evaluate(self, *args, **kwargs):
        """Use EMA parameters during evaluation"""
        using_ema = hasattr(self.model, "ema") and self.model.ema is not None
        if using_ema:
            # Save original model parameters and switch to EMA parameters
            self.model.ema.store(self.model.parameters())  # Backup current model parameters
            self.model.ema.copy_to(self.model.parameters())  # Load EMA parameters into model
        # Execute default evaluation logic
        results = super().evaluate(*args, **kwargs)
        if using_ema:
            # Restore original model parameters
            self.model.ema.restore(self.model.parameters())
        return results