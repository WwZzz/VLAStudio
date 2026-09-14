"""Small external dataset for a CPU packaging smoke test."""
import numpy as np
import torch

class ToyDataset:
    def __init__(self, size=8):
        self.size = size
        self.ctrl_space = "joint"
        self.ctrl_type = "abs"
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError(index)
        return {"state": np.array([index / 8, 0.5], dtype=np.float32),
                "action": np.array([[index / 8, 0.5]], dtype=np.float32), "is_pad": torch.zeros(1, dtype=torch.bool)}
    def extract_all(self, keys):
        return {key: np.concatenate([self[i][key].reshape(-1, 2) for i in range(self.size)]) for key in keys}
