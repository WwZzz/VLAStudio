
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Any, List, Union
import warnings
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

IGNORE_INDEX = -100
    
class OpenVLACollator:
    def __init__(self, tokenizer, dtype=torch.bfloat16):
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
        )
    
    def __call__(self, instances):
        return self.collator(instances)

class OpenVLAProcessor:
    """Simplified data processor for OpenVLA training."""
    
    def __init__(self, tokenizer, image_transform):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = PurePromptBuilder
        self.action_tokenizer = ActionTokenizer(tokenizer)
        self.predict_stop_token = True
    
    def __call__(self, sample):
        """Process a single sample."""
        action = sample.get('action', None)
        if action is not None:
            action = action[0]
        # Handle image format: (num_cameras, C, H, W) -> take first camera
        image_tensor = sample['image'][0]  # Take first camera (primary)
        # Convert from tensor to PIL Image
        image_array = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(image_array)
        lang = sample['raw_lang']
        
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        if action is not None:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if action is not None:
            labels[: -(len(action) + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX
        # Note: Removed dataset_name field to avoid None values causing concatenation errors in accelerate
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        