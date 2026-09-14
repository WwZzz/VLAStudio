from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np

class DAQUARDataset(Dataset):
    def __init__(self, dataset_path: str=None, split='train'):
        self.dataset_path = dataset_path
        self.dataset = load_dataset("Luo-wj/DAQUAR", data_dir=dataset_path, split=split)
        self.dataset_id = "DAQUAR"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        images = np.stack([np.array(item['images'][i]) for i in range(len(item['images']))]).transpose(0, 3, 1, 2)
        conversation = [
            {"from": "human", "value": item['problem']},
            {"from": "gpt", "value": item['answer']},
        ]
        raw_lang = ""
        return {
            "image": torch.from_numpy(images),
            "raw_lang": raw_lang,
            "reasoning": {
                "conversation": conversation,
                "meta": {},
                "is_vqa": True,
            },
        }


