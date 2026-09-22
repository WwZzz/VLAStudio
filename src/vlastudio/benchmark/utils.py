import numpy as np
from dataclasses import asdict
import torch
import time
import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from pathlib import Path
from collections import deque
import cv2
from loguru import logger


class SequentialVectorEnv:
    """
    Simple sequential vector environment wrapper (no multiprocessing).
    Useful for environments that have issues with daemon processes.
    """
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.envs = [fn() for fn in env_fns]
        self.env_num = len(self.envs)
    
    def _stack_obs(self, obs_list):
        """Stack observations, handling both numpy arrays and MetaObs objects."""
        if len(obs_list) == 0:
            return None
        
        # Check if first observation is a MetaObs object
        first_obs = obs_list[0]
        if hasattr(first_obs, '__dataclass_fields__'):  # MetaObs is a dataclass
            # Return as a regular numpy array (not object dtype) to be compatible with organize_obs
            # SubprocVectorEnv returns np.array where each element is a MetaObs
            return np.array(obs_list)
        else:
            # Regular numpy array observations
            return np.stack(obs_list)
    
    def reset(self, id=None):
        if id is None:
            obs_list = [env.reset() for env in self.envs]
            return self._stack_obs(obs_list)
        else:
            if np.isscalar(id):
                return self.envs[id].reset()
            else:
                obs_list = [self.envs[i].reset() for i in id]
                return self._stack_obs(obs_list)
    
    def step(self, action, id=None):
        if id is None:
            results = [env.step(act) for env, act in zip(self.envs, action)]
            obs = self._stack_obs([r[0] for r in results])
            rew = np.array([r[1] for r in results])
            done = np.array([r[2] for r in results])
            info = [r[3] for r in results]
            return obs, rew, done, info
        else:
            if np.isscalar(id):
                return self.envs[id].step(action)
            else:
                results = [self.envs[i].step(act) for i, act in zip(id, action)]
                obs = self._stack_obs([r[0] for r in results])
                rew = np.array([r[1] for r in results])
                done = np.array([r[2] for r in results])
                info = [r[3] for r in results]
                return obs, rew, done, info
    
    def __len__(self):
        return self.env_num
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()


def get_images_from_metaobs(mobs): 
    images = mobs.image
    return [Image.fromarray(img.transpose(1,2,0)) for img in images]

def resize_with_pad(img, width, height, pad_value=-1, interpolation=cv2.INTER_LINEAR):
    """Resize (N,C,H,W) to (N,C,height,width) with aspect ratio preserved via padding.
    - Supports torch.Tensor or numpy.ndarray inputs
    - Pads on left and top (image aligned to bottom-right), matching existing behavior
    """
    is_torch = isinstance(img, torch.Tensor)
    device = img.device if is_torch else None
    dtype = img.dtype if is_torch else None
    arr = img.detach().cpu().numpy() if is_torch else img
    if arr.ndim != 4:
        raise ValueError(f"(n,c,h,w) expected, but {tuple(arr.shape)}")
    
    n, c, h, w = arr.shape
    if h==height and w==width: return img
    out_list = []
    for i in range(n):
        chw = arr[i]
        cur_h, cur_w = chw.shape[1], chw.shape[2]
        ratio = max(cur_w / width, cur_h / height)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)

        hwc = np.transpose(chw, (1, 2, 0))
        resized = cv2.resize(hwc, (resized_w, resized_h), interpolation=interpolation)

        canvas = np.full((height, width, c), pad_value, dtype=resized.dtype)
        y0 = height - resized_h
        x0 = width - resized_w
        canvas[y0:height, x0:width, :] = resized

        chw_out = np.transpose(canvas, (2, 0, 1))
        out_list.append(chw_out)

    out = np.stack(out_list, axis=0)
    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out.astype(arr.dtype, copy=False)

def _save_example_batch(obs, act, save_dir):
    """
    Save example observation (MetaObs) and action (MetaAct) for debugging.
    Saves only the FIRST environment's data (index 0).
    Saves images as PNG, states/actions as CSV, and metadata as TXT.
    
    Args:
        obs: MetaObs object containing observations (batch)
        act: MetaAct object or numpy array containing actions (batch)
        save_dir: Directory to save the examples
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if example already exists
        check_file = os.path.join(save_dir, 'info.txt')
        if os.path.exists(check_file):
            logger.info(f"Example already exists in {save_dir}, skipping save.")
            return
        
        # 1. Save images
        if hasattr(obs, 'image') and obs.image is not None:
            image_data = obs.image
            # Convert to numpy if tensor
            if isinstance(image_data, torch.Tensor):
                image_data = image_data.cpu().numpy()
            
            # Handle different image shapes - ONLY SAVE FIRST ENVIRONMENT (index 0)
            # Expected formats:
            # - (batch, cameras, H, W, C) - 5D with channels last
            # - (batch, cameras, C, H, W) - 5D with channels first
            # - (batch, H, W, C) - 4D with channels last
            # - (batch, C, H, W) - 4D with channels first
            if len(image_data.shape) == 5:
                # Check if channels first (C, H, W) or channels last (H, W, C)
                if image_data.shape[-1] in [1, 3, 4]:  # (batch, cameras, H, W, C)
                    batch_size, num_cameras, H, W, C = image_data.shape
                    # Only save first environment (b=0)
                    for c in range(num_cameras):
                        img = image_data[0, c]  # (H, W, C)
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        img_pil = Image.fromarray(img)
                        img_path = os.path.join(save_dir, f'image_cam{c}.png')
                        img_pil.save(img_path)
                    logger.info(f"Saved {num_cameras} images from first environment to {save_dir}")
                elif image_data.shape[2] in [1, 3, 4]:  # (batch, cameras, C, H, W)
                    batch_size, num_cameras, C, H, W = image_data.shape
                    # Only save first environment (b=0)
                    for c in range(num_cameras):
                        img = image_data[0, c].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        img_pil = Image.fromarray(img)
                        img_path = os.path.join(save_dir, f'image_cam{c}.png')
                        img_pil.save(img_path)
                    logger.info(f"Saved {num_cameras} images from first environment to {save_dir}")
                else:
                    logger.warning(f"Unsupported 5D image shape: {image_data.shape}, skipping image save")
            elif len(image_data.shape) == 4:
                # Check if channels first or channels last
                # Priority: try to detect based on which dimension looks like channels
                # Common channel counts: 1 (grayscale), 3 (RGB), 4 (RGBA)
                dim_sizes = image_data.shape
                
                # Heuristic: channels are usually the smallest dimension and in [1,3,4]
                possible_formats = []
                if dim_sizes[-1] in [1, 3, 4]:
                    possible_formats.append(('channels_last', -1))
                if dim_sizes[1] in [1, 3, 4]:
                    possible_formats.append(('channels_first', 1))
                
                if not possible_formats:
                    logger.warning(f"Cannot determine image format for shape {image_data.shape}, skipping")
                else:
                    # Use the first valid format
                    format_type, channel_dim = possible_formats[0]
                    
                    if format_type == 'channels_last':  # (batch, H, W, C)
                        batch_size, H, W, C = image_data.shape
                        # Only save first environment (b=0)
                        img = image_data[0]  # (H, W, C)
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        if C == 1:
                            img = img.squeeze(-1)  # Remove channel dimension for grayscale
                        img_pil = Image.fromarray(img)
                        img_path = os.path.join(save_dir, f'image.png')
                        img_pil.save(img_path)
                        logger.info(f"Saved 1 image from first environment to {save_dir}")
                    else:  # channels_first: (batch, C, H, W)
                        batch_size, C, H, W = image_data.shape
                        # Only save first environment (b=0)
                        img = image_data[0].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        if C == 1:
                            img = img.squeeze(-1)  # Remove channel dimension for grayscale
                        img_pil = Image.fromarray(img)
                        img_path = os.path.join(save_dir, f'image.png')
                        img_pil.save(img_path)
                        logger.info(f"Saved 1 image from first environment to {save_dir}")
            else:
                logger.warning(f"Unsupported image shape: {image_data.shape} (expected 4D or 5D), skipping image save")
        
        # 2. Save state (raw data from environment) - ONLY FIRST ENVIRONMENT
        if hasattr(obs, 'state') and obs.state is not None:
            state_data = obs.state
            if isinstance(state_data, torch.Tensor):
                state_data = state_data.cpu().numpy()
            
            # Only save first environment (index 0)
            if state_data.ndim > 1:
                state_data = state_data[0:1]  # Keep 2D shape (1, state_dim)
            else:
                state_data = state_data.reshape(1, -1)
            
            # Save raw state (unnormalized data from environment)
            state_file = os.path.join(save_dir, 'state_raw.csv')
            np.savetxt(state_file, state_data, delimiter=',', 
                      header=','.join([f'state_{i}' for i in range(state_data.shape[1])]),
                      comments='')
            logger.info(f"Saved raw state from first environment to {state_file}")
        
        # 3. Save action (raw data from policy)
        action_data = None
        if hasattr(act, 'action'):
            action_data = act.action
        elif isinstance(act, (np.ndarray, torch.Tensor)):
            action_data = act
            # If it's a numpy array of dicts (dtype='object'), extract the 'action' field
            if isinstance(action_data, np.ndarray) and action_data.dtype == np.object_:
                # This is likely an array of MetaAction dicts from policy.inference
                try:
                    # Extract 'action' field from each dict
                    action_list = []
                    for item in action_data.flat:
                        if isinstance(item, dict) and 'action' in item:
                            action_list.append(item['action'])
                        elif isinstance(item, np.ndarray):
                            action_list.append(item)
                    if action_list:
                        action_data = np.array(action_list)
                    else:
                        logger.warning(f"Cannot extract action data from object array, skipping action save")
                        action_data = None
                except Exception as e:
                    logger.warning(f"Failed to extract action from object array: {e}, skipping action save")
                    action_data = None
        
        if action_data is not None:
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.cpu().numpy()
            
            # Ensure action_data is numeric and at least 2D for CSV saving
            if action_data.dtype == np.object_:
                logger.warning(f"Action data still has object dtype after extraction, skipping save")
            else:
                # Only save first environment (index 0)
                if action_data.ndim > 1:
                    action_data = action_data[0:1]  # Keep 2D shape (1, action_dim)
                else:
                    action_data = action_data.reshape(1, -1)
                
                # Save raw action (denormalized data from policy)
                action_file = os.path.join(save_dir, 'action_raw.csv')
                np.savetxt(action_file, action_data, delimiter=',',
                          header=','.join([f'action_{i}' for i in range(action_data.shape[1])]),
                          comments='')
                logger.info(f"Saved raw action from first environment to {action_file}")
        
        # 4. Save metadata and info
        info_file = os.path.join(save_dir, 'info.txt')
        with open(info_file, 'w') as f:
            f.write("=== Example Batch Info ===\n\n")
            
            # Observation info
            f.write("Observation (MetaObs):\n")
            if hasattr(obs, '__dict__'):
                for key, value in obs.__dict__.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        shape = value.shape if hasattr(value, 'shape') else 'N/A'
                        dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
                        f.write(f"  {key}: shape={shape}, dtype={dtype}\n")
                    elif value is not None:
                        f.write(f"  {key}: {type(value).__name__} = {value}\n")
            f.write("\n")
            
            # Action info
            f.write("Action (MetaAct):\n")
            if hasattr(act, '__dict__'):
                for key, value in act.__dict__.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        shape = value.shape if hasattr(value, 'shape') else 'N/A'
                        dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
                        f.write(f"  {key}: shape={shape}, dtype={dtype}\n")
                    elif value is not None:
                        f.write(f"  {key}: {type(value).__name__} = {value}\n")
            elif isinstance(act, (np.ndarray, torch.Tensor)):
                f.write(f"  shape={act.shape}, dtype={act.dtype}\n")
            
            f.write("\n")
            f.write("Files saved (ONLY FIRST ENVIRONMENT, index 0):\n")
            f.write("  - image_cam{j}.png or image.png: observation images from first environment\n")
            f.write("  - state_raw.csv: raw state values from first environment (1 row)\n")
            f.write("  - action_raw.csv: raw action values for first environment (1 row)\n")
            f.write("  - info.txt: this file\n")
            f.write("\n")
            f.write("Note: Only the first environment (index 0) from the batch is saved.\n")
            f.write("      State comes from environment (raw/unnormalized).\n")
            f.write("      Action comes from policy (denormalized if using normalizers).\n")
        
        logger.info(f"Saved example batch info to {info_file}")
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to save example batch: {e}")
        logger.error(traceback.format_exc())

def organize_obs(obs: np.ndarray):
    """Organize obs returned by SubprocVectorEnv into a dict"""
    # Lazy import to avoid circular dependency at module import time
    from .base import dict2meta
    if isinstance(obs, dict): return obs
    if isinstance(obs[0], dict):
        all_obs_dict = list(obs)
    else:
        all_obs_dict = list(asdict(obsi) for obsi in obs)
    assert len(all_obs_dict)>0, "emypt observation"
    all_keys = list(all_obs_dict[0].keys())
    res = {k:[vi[k] for vi in all_obs_dict] for k in all_keys}
    for k in res:
        if isinstance(res[k][0], np.ndarray):
            res[k] = np.stack(res[k])
        elif res[k][0] is None:
            res[k] = None
    res['state'] = res['state']
    # Note: camera selection is now handled in each environment's obs2meta method
    return dict2meta(res)

def _render_video_frames(obs, video_frames, success, env):
    """Extract video frames from observation for recording."""
    frames = obs['image']
    # Handle multiple camera views: concatenate horizontally
    if len(frames.shape) == 5:
        # Shape: (batch, num_cameras, channels, height, width)
        batch_size, num_cameras, channels, height, width = frames.shape
        frames = frames.transpose(0, 1, 3, 4, 2)  # (batch, num_cameras, height, width, channels)
        concatenated_frames = []
        for b in range(batch_size):
            batch_frames = frames[b].transpose(1, 0, 2, 3)  # (height, num_cameras, width, channels)
            batch_frames = batch_frames.reshape(height, num_cameras * width, channels)
            concatenated_frames.append(batch_frames)
        frames = np.stack(concatenated_frames, axis=0)
    else:
        # Shape: (batch, channels, height, width)
        frames = frames.transpose(0, 2, 3, 1)
    for env_i in range(len(env)):
        if not success[env_i]:
            video_frames[env_i].append(frames[env_i])


def evaluate(
    args,
    env,
    action_manager,
    inference_ctx=None,
    video_writer=None,
    save_example_dir=None,
):
    """
    Run evaluation rollouts.
    
    Uses action_manager for action selection. The action_manager communicates
    with an inference_worker subprocess via InferenceContext (SHM) to obtain
    action chunks.
    
    Obs/trigger are decoupled:
    - Obs is written to obs_shm by this function (sim mode, via inference_ctx.update_obs)
    - action_manager only sends trigger signals (no obs handling)
    
    Args:
        args: Evaluation arguments (must have max_timesteps).
        env: Vectorized environment.
        action_manager: ActionManager with InferenceContext set.
        inference_ctx: InferenceContext for writing obs to SHM (sim mode).
                       In real mode this is not used (obs comes from device SHMs).
        video_writer: Optional video writer for recording.
        save_example_dir: Optional directory to save example batch.
    """
    video_frames = [[] for _ in range(len(env))]
    horizons = np.ones(len(env)) * args.max_timesteps
    
    # Start evaluation
    with torch.inference_mode():
        time_start_eval = time.time()
        success = np.zeros(len(env), dtype=np.bool_)
        obs = env.reset()
        obs = organize_obs(obs)
        
        # Save first observation and action for debugging
        first_obs_saved = False
        
        for t in range(args.max_timesteps):
            if video_writer is not None:
                _render_video_frames(obs, video_frames, success, env)
            
            # Write obs to SHM for inference worker (sim mode)
            if inference_ctx is not None:
                inference_ctx.update_obs(obs, t)
            
            # Get action from action_manager (sends trigger, polls action_shm)
            # Step counter is managed internally by action_manager
            act = action_manager.select_action()
            
            # Save first obs and action
            if not first_obs_saved and save_example_dir is not None:
                _save_example_batch(obs, act, save_example_dir)
                first_obs_saved = True
            
            obs, reward, done, info = env.step(act)
            obs = organize_obs(obs)
            # Decide if success
            success = success | done
            if success.all(): 
                for sidx in range(success.shape[0]):
                    if horizons[sidx] > t:
                        horizons[sidx] = t
                break
            elif success.any():
                success_idx = np.where(success == True)[0]
                for sidx in success_idx: 
                    if horizons[sidx] > t:
                        horizons[sidx] = t

    # Cleanup
    env.close()
    
    # Compute metrics
    total_successes = int(success.sum().item())
    total = len(env)
    success_rate = 1.0 * total_successes / len(env)
    eval_time = time.time() - time_start_eval
    
    # Save video
    if video_writer is not None:
        for env_i in range(len(env)):
            for frame in video_frames[env_i]:
                video_writer.append_data(frame)
    
    return {
        'success': success.tolist(),
        'total_success': total_successes,
        'total': total,
        'success_rate': success_rate,
        'horizon': horizons.tolist(),
        'horizon_success': (success * horizons).sum() / max(total_successes, 1),
        'eval_time': eval_time,
    }
