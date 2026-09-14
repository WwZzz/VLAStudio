import transformers
import torch 
from vlastudio.policy.trainer import BaseTrainer
class Trainer(BaseTrainer):
    def create_optimizer(self):
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.model.config.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return self.optimizer