import torch
import numpy as np
import torch.nn.functional as F  # noqa: N812


class SmolVLAProcess:
    def __call__(self, sample):
        # process language
        if not sample['raw_lang'].endswith("\n"):
            sample['raw_lang'] = sample['raw_lang'] + "\n"
        # tokenized_prompt = self.tokenizer(
        #     sample['raw_lang'],
        #     max_length=self.max_length,
        #     truncation=self.truncation,
        #     padding=self.padding,
        #     padding_side=self.padding_side,
        #     return_tensors="pt",
        # )
        if sample['image'].max()>1.0:
            image_data = sample['image'] / 255.0 # k,c,h,w
        # organize data
        data_dict = {}
        data_dict['state'] = sample.get('state', None)
        data_dict['action'] = sample.get('action', None)
        data_dict['is_pad'] = sample.get('is_pad', None)
        data_dict['image'] = image_data
        data_dict['raw_lang'] = sample['raw_lang']
        # for k, v in tokenized_prompt.items():
        #     data_dict[k] = v
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)
        return data_dict

class SmolVLADataCollator:
    def __init__(self, tokenizer, max_state_dim:int=32, max_action_dim:int=32, resize_imgs_with_padding:tuple[int, int]=(512, 512), max_length=512, padding_side="right", padding="max_length", truncation=True):
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side
        self.padding = padding
        self.truncation = truncation

    def pad_vector(self,vector, new_dim):
        """Can be (batch_size x sequence_length x features_dimension)
        or (batch_size x features_dimension)
        """
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector

    # def resize_with_pad(self, img, width, height, pad_value=-1):
    #     # assume no-op when width height fits already
    #     if img.ndim != 4:
    #         raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    #     cur_height, cur_width = img.shape[2:]

    #     ratio = max(cur_width / width, cur_height / height)
    #     resized_height = int(cur_height / ratio)
    #     resized_width = int(cur_width / ratio)
        
    #     resized_img = F.interpolate(
    #         img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    #     )

    #     delta_h = max(0, height - resized_height)
    #     delta_w = max(0, width - resized_width)

    #     pad_top = delta_h // 2
    #     pad_bottom = delta_h - pad_top
        
    #     pad_left = delta_w // 2
    #     pad_right = delta_w - pad_left

    #     padded_img = F.pad(
    #         resized_img, 
    #         (pad_left, pad_right, pad_top, pad_bottom), 
    #         value=pad_value
    #     )
    #     return padded_img
    
    def resize_with_pad(self, img, width, height, pad_value=-1):
        # assume no-op when width height fits already
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def __call__(self, instances):
        # process state and action
        if not isinstance(instances[0]['state'], torch.Tensor):
            states = torch.tensor(np.array([instance['state'] for instance in instances]))
        else:
            states = torch.stack([instance['state'] for instance in instances])
        states = self.pad_vector(states, self.max_state_dim)
        if instances[0]['action'] is not None:
            if not isinstance(instances[0]['action'], torch.Tensor):
                actions = torch.tensor(np.array([instance['action'] for instance in instances]))
            else:
                actions = torch.stack([instance['action'] for instance in instances])
            actions = self.pad_vector(actions, self.max_action_dim)
            is_pad_all = torch.stack([instance['is_pad'] for instance in instances])
        else:
            actions = None
            is_pad_all = None
        
        # process image
        num_images = instances[0]['image'].shape[0]
        bs = len(instances)
        images = [torch.stack([ins['image'][img_idx] for ins in instances]) for img_idx in range(num_images)]
        images = [self.resize_with_pad(img, self.resize_imgs_with_padding[0], self.resize_imgs_with_padding[1], pad_value=0) for img in images]
        images = [img * 2.0 - 1.0 for img in images]
        img_masks = [torch.ones(bs, dtype=torch.bool) for _ in range(num_images)]
        tokenized_prompt = self.tokenizer(
            [ins['raw_lang'] for ins in instances],
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt.input_ids
        lang_masks = tokenized_prompt.attention_mask.bool()
        batch = dict(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=states,
            actions=actions,
            is_pad=is_pad_all
        )
        return batch