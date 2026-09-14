"""Synthetic ALOHA-shaped data loaded only inside the managed ACT environment."""
from pathlib import Path

import numpy as np
import torch


class AlohaSmokeDataset:
    def __init__(self, size=8, chunk_size=50, dataset_dir=None):
        self.size = int(size)
        self.chunk_size = int(chunk_size)
        self.dataset_dir = str(Path(dataset_dir or ".").expanduser().resolve())
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        self.ctrl_space = "joint"
        self.ctrl_type = "abs"

    def get_dataset_dir(self):
        return self.dataset_dir

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError(index)
        rng = np.random.default_rng(index)
        return {
            "state": rng.normal(size=14).astype(np.float32),
            "action": rng.normal(size=(self.chunk_size, 14)).astype(np.float32),
            "image": torch.from_numpy(
                rng.integers(0, 256, size=(1, 3, 64, 64), dtype=np.uint8)
            ),
            "is_pad": torch.zeros(self.chunk_size, dtype=torch.bool),
            "raw_lang": "Transfer the red cube from the right arm to the left arm.",
            "reasoning": "",
        }

    def extract_all(self, keys):
        result = {}
        for key in keys:
            values = [np.asarray(self[index][key]) for index in range(self.size)]
            result[key] = np.concatenate([value.reshape(-1, 14) for value in values])
        return result
