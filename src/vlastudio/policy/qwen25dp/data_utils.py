from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms as transforms
import torch
from qwen_vl_utils import process_vision_info
from qwen_vl_utils import *
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers
import gc
import os

class QwenVLAProcess:
    def __init__(
            self,
            tokenizer=None,
            max_seq_len=512,
            multimodal_processor=None,
            camera_names=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.camera_names = camera_names
        self.multimodal_processor = multimodal_processor

    def preprocess_image(self, image, size=224):
        # Model has been trained to handle images of different aspects ratios
        # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
        # options are helpful to improve quality in some tasks.
        image = np.asarray(image)
        if image.ndim == 2:  # Convert image without last channel into greyscale.
            image = np.stack((image,) * 3, axis=-1)
        image = image[..., :3]  # Remove alpha layer.
        assert image.shape[-1] == 3
        image_pil = to_pil_image(image)

        # Step 2: Define the resize transformation
        resize_transform = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)

        # Step 3: Apply the resize transformation
        image_resized_pil = resize_transform(image_pil)

        # Step 4: Convert back to tensor if needed
        image_resized = to_tensor(image_resized_pil)
        return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

    def qwen2_image_preprocess(self, each, camera_name):
        ele = {
        }
        each = Image.fromarray(each.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
        ele['image'] = each
        if 'wrist' in camera_name:
            ele['resized_height'] = each.height
            ele['resized_width'] = each.width
        each = fetch_image(ele)
        return torch.from_numpy(np.array(each))

    def __call__(self, sample, use_reasoning=True):
        # Process model input part
        video = False
        messages = self.datastruct_droid2llava(sample, video=video)

        data_dict = dict(
            messages=messages,
            images=None
        )

        image_data = torch.chunk(sample['image'], sample['image'].shape[0], 0)
        if self.camera_names is not None:
            images_list = []
            for i, each in enumerate(image_data):
                img_pil = self.qwen2_image_preprocess(each, self.camera_names[i])
                images_list.append(img_pil)
            image_data = images_list
        else:
            images_list = [imgi for imgi in image_data]
        video_inputs = None
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_labels = torch.ones_like(model_inputs['input_ids']) * (-100)
        answer = '<|im_end|>'
        # answer = sample['reasoning']
        # Start processing output part
        # Tokenize label text, convert to vocabulary id
        output_text = self.tokenizer(answer, padding=True, return_tensors="pt")
        output_labels = output_text['input_ids'].long()
        # Concatenate input and output text ids as final text
        model_inputs['input_ids'] = torch.cat((model_inputs['input_ids'], output_text['input_ids']), dim=-1).long()
        # Concatenate attention masks of both texts
        model_inputs['attention_mask'] = torch.cat((model_inputs['attention_mask'], output_text['attention_mask']), dim=-1).bool()
        # Remember both labels, ignore_idx is -100 for no loss calculation, remaining answer needs loss calculation, so shift later
        labels = torch.cat((input_labels, output_labels), dim=-1)
        data_dict['state'] = sample.get('state', None)
        data_dict['action'] = sample.get('action', None)
        data_dict['is_pad'] = sample.get('is_pad', None)
        data_dict['labels'] = labels
        data_dict['image_data'] = image_data
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict

    def datastruct_droid2llava(self, sample, video=False):
        len_image = sample['image'].shape[0]

        messages = [
            {
                "role": "user",
                "content": [],
            },
        ]

        for i in range(len_image):
            if video:
                messages[0]['content'].append({
                    "type": "video",
                    "video": None,
                })
            else:
                messages[0]['content'].append({
                            "type": "image",
                            "image": None,
                        })
        messages[0]['content'].append({"type": "text", "text": f""})
        messages[0]['content'][-1]['text'] = sample['raw_lang']

        return messages
    
class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        processed_sample = self.processor(sample)
        return processed_sample
    

@dataclass
class QwenVLADataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    multimodal_processor: transformers.AutoProcessor=None
    computed_type: torch.dtype=None
    tokenizer: transformers.AutoTokenizer=None
    video: bool=False
    dtype: torch.dtype=torch.float32

    # @profile
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.flip(instance['input_ids'].squeeze(0), dims=[0]) for instance in instances]
        labels = [torch.flip(instance['labels'].squeeze(0), dims=[0]) for instance in instances]
        if self.video:
            video_grid_thw = torch.stack([instances['video_grid_thw'] for instances in instances])
            pixel_values_videos = torch.stack([instances['pixel_values_videos'] for instances in instances])
            pixel_values = None
            image_grid_thw=None
        else:
            image_grid_thw = torch.stack([instances['image_grid_thw'] for instances in instances])
            pixel_values = torch.stack([instances['pixel_values'] for instances in instances])
            pixel_values_videos = None
            video_grid_thw = None

        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        labels = torch.flip(labels, dims=[1])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        input_ids = torch.flip(input_ids, dims=[1])
        b = input_ids.shape[0]
        if self.video:
            video_grid_thw = video_grid_thw.reshape(b * video_grid_thw.shape[1], video_grid_thw.shape[2])
            pixel_values_videos = pixel_values_videos.reshape(b * pixel_values_videos.shape[1], pixel_values_videos.shape[2])

        else:
            image_grid_thw = image_grid_thw.reshape(b * image_grid_thw.shape[1], image_grid_thw.shape[2])
            pixel_values = pixel_values.reshape(b * pixel_values.shape[1], pixel_values.shape[2])

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        if isinstance(instances[0]['action'], list):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
        elif isinstance(instances[0]['action'], np.ndarray):
            actions = torch.tensor(np.stack([instance['action'] for instance in instances]))
        elif isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.stack([instance['action'] for instance in instances])
        else:
            actions = None
        
        if isinstance(instances[0]['state'], list):
            states = torch.tensor(np.array([instance['state'] for instance in instances]))
        elif isinstance(instances[0]['state'], np.ndarray):
            actions = torch.tensor(np.stack([instance['state'] for instance in instances]))
        elif isinstance(instances[0]['state'], torch.Tensor):
            states = torch.stack([instance['state'] for instance in instances])
        elif instances[0]['state'] is not None:
            states = None
        is_pad_all = torch.stack([instance['is_pad'] for instance in instances]) if instances[0]['is_pad'] is not None else None

        assert len(attention_mask.shape) == 2, "Attention mask shape should be (batch_size, seq_len)"
        #exit(0)
        batch = dict(
            input_ids=input_ids.long(),
            attention_mask=attention_mask,
            labels=labels,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            actions=actions.to(dtype=self.dtype),
            states=states.to(dtype=self.dtype),
            video_grid_thw=video_grid_thw,
            pixel_values=pixel_values,
            is_pad=is_pad_all
        )
        del input_ids
        del attention_mask
        del labels
        del pixel_values_videos
        del pixel_values
        del actions
        del states
        del video_grid_thw
        del image_grid_thw
        del is_pad_all
        gc.collect()
        return batch
