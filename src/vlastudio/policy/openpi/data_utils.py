import torch
import numpy as np
import openpi.models.model
from PIL import Image
import openpi.models.tokenizer as _tokenizer
from loguru import logger
# openpi.models.model.Observation

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im if images.max()>1.0 else (im*255).astype(np.uint8)), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}

class OpenPiProcessor:
    def __init__(self, max_token_len:int=48, image_size=[224,224], discrete_state_input: bool = False, model_action_dim: int=32, pi05: bool = False):
        self.image_size = image_size
        self.tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
        self.discrete_state_input = discrete_state_input
        self.model_action_dim = model_action_dim
        self.pi05 = pi05

    def __call__(self, sample):
        # process image (k c h w)
        if isinstance(sample['image'], torch.Tensor):
            img = sample['image'].numpy()
        elif isinstance(sample['image'], np.ndarray):
            img = sample['image']
        else:
            raise ValueError("img must be torch.Tensor or np.ndarray")
        img = img.transpose(0,2,3,1) # kchw -> khwc
        img = resize_with_pad(img, height=self.image_size[0], width=self.image_size[1])
        img_keys = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        num_imgs = min(len(img_keys), img.shape[0])
        images = {img_keys[i]:img[i] if i<num_imgs else np.zeros_like(img[0]) for i in range(3)}
        image_masks = {img_keys[i]: i<num_imgs for i in range(3)}
        # padding state
        state = sample['state']
        state = pad_to_dim(state, self.model_action_dim, axis=-1)
        prompt = sample['raw_lang']
        if not isinstance(prompt, str):
            prompt = prompt.item()
        # process language and tokenize data
        if self.pi05 or self.discrete_state_input:
            if (state := sample.get("state", None)) is None:
                raise ValueError("State is required.")
            tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        else:
            tokens, token_masks = self.tokenizer.tokenize(prompt, None)
        lanugage_dict = {"tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
        # process action
        if "action" in sample:
            action = pad_to_dim(sample["action"], self.model_action_dim, axis=-1)
        else:
            action = None
        
        data_dict = {
            'images': images,
            'image_masks': image_masks,
            'state': state,
            'actions': action,
            'is_pad': sample.get('is_pad', None),
            **lanugage_dict,
        }
        return data_dict
    
    
class OpenPiCollator:
    def __init__(self, device=None):
        self.device = 'cuda' if device is None else device
        
    def __call__(self, instances):
        # process image
        image_keys = list(instances[0]['images'].keys())
        images = {k:torch.from_numpy(np.stack([instance['images'][k] for instance in instances]).transpose(0,3,1,2).astype(np.float32)/255.*2.0-1.0) for k in image_keys}
        image_masks = {k:torch.from_numpy(np.array([instance['image_masks'][k] for instance in instances])) for k in image_keys}
        
        # process state and action
        if not isinstance(instances[0]['state'], torch.Tensor):
            states = torch.tensor(np.array([instance['state'] for instance in instances]))
        else:
            states = torch.stack([instance['state'] for instance in instances])
        if instances[0]['actions'] is not None:
            if not isinstance(instances[0]['actions'], torch.Tensor):
                actions = torch.tensor(np.array([instance['actions'] for instance in instances]))
            else:
                actions = torch.stack([instance['actions'] for instance in instances])
            is_pad_all = torch.stack([instance['is_pad'] for instance in instances])
        else:
            actions = None
            is_pad_all = None
        # process language
        tokenized_prompt = torch.from_numpy(np.stack([instance["tokenized_prompt"] for instance in instances]))
        tokenized_prompt_mask = torch.from_numpy(np.stack([instance["tokenized_prompt_mask"] for instance in instances]))
        observation = dict(
            image=images,
            image_mask=image_masks, 
            state=states.to(torch.float64), 
            tokenized_prompt=tokenized_prompt, 
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        return {'observation': observation, 'actions': actions, "is_pad": is_pad_all}
    
        