"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

import os
import concurrent.futures
import pathlib
import shutil
import time
import urllib.parse
import torch
import filelock
import fsspec
from tqdm import tqdm
from typing import Union, List
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds

# Suppress TensorFlow logging after import
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import collections
import numpy as np
import json
from pathlib import Path
import logging
import warnings
from vlastudio.data_utils.utils import ensure_uint8_image
from loguru import logger

class DroidDataset:
    def __init__(
        self,
        dataset_dir: str,
        name: str = "droid_100",
        version: str = "1.0.0",
        shuffle: bool = True,
        chunk_size: int = 16,
        ctrl_space: str = 'joint',
        ctrl_type: str = 'abs', 
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        filter_dict_path=None,  # Path to json file with indices to sample during training
        data_processor: callable = None,  # Optional function to apply to each data sample
        data_collator: callable = None,  # Optional function to collate a batch of data samples
        gpu_for_dataset: List[int] = [],
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        self.dataset_dir = dataset_dir
        self.name = name
        self.version = version
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_calls = num_parallel_calls
        self.filter_dict_path = filter_dict_path
        self.data_processor = data_processor
        self.data_collator = data_collator
        gpus = tf.config.list_physical_devices('GPU')   
        tf.config.set_visible_devices([gpus[i] for i in gpu_for_dataset] , "GPU")
        # Load dataset
        dataset = self.load_data()
        # Shuffle, batch
        # dataset = dataset.shuffle(shuffle_buffer_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)
        self.dataset = dataset

    def extract_all(self, features: Union[List[str], str]):
        if isinstance(features, str): features = [features]
        all_funcs = {}
        for key in features:
            if 'action' in key or 'actions' in key:
                all_funcs['action'] = lambda x: x['actions']
            elif 'state' in key:
                all_funcs['state'] = lambda traj: tf.concat([tf.cast(traj["observation"]["joint_position"], tf.float32), tf.cast(traj["observation"]["gripper_position"], tf.float32)], axis=-1) 
            elif 'image' in key:
                all_funcs['image'] = lambda traj: tf.stack([traj["observation"]["image"], traj["observation"]["wrist_image"]], axis=0)  # (2, H, W, C)
        dataset = self.traj_dataset.map(lambda traj: {feat: all_funcs[feat](traj) for feat in features}, num_parallel_calls=self.num_parallel_calls)
        # batch_size = 1024
        # dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        numpy_iterator = tfds.as_numpy(dataset)
        all_data_batches = collections.defaultdict(list)
        for batch in tqdm(numpy_iterator):
            for feat in features:
                if feat in batch:
                    all_data_batches[feat].append(batch[feat])
        final_data = {}
        for feat in features:
            if all_data_batches[feat]:
                final_data[feat] = np.concatenate(all_data_batches[feat], axis=0)
            else:
                final_data[feat] = np.array([]) # 处理空数据集或特征不存在的情况
        return final_data

    def load_data(self, *args, **kwargs):
        data_dir=self.dataset_dir
        shuffle=self.shuffle
        action_chunk_size=self.chunk_size
        action_space=self.ctrl_space
        num_parallel_reads=self.num_parallel_reads
        num_parallel_calls=self.num_parallel_calls
        filter_dict_path=self.filter_dict_path

        # tf.config.set_visible_devices([] , "GPU")

        builder = tfds.builder(self.name, data_dir=data_dir, version=self.version)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # # Repeat dataset so we never run out of data.
        

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        
        if filter_dict_path is not None:
            cached_filter_dict_path = self.download(filter_dict_path, cache_dir=data_dir)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            if action_space=='joint': action_key = "joint_position"
            elif action_space=='ee': action_key = "cartesian"
            elif action_space=='joint_vel': action_key = "joint_velocity"
            elif action_space=='ee_vel': action_key = "cartesian_velocity"
            else: raise ValueError(f"Invalid action space {action_space}")
            actions = tf.concat(
                (
                    (
                        traj["action_dict"][action_key]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)
        self.traj_dataset = dataset
        
        def chunk_actions(traj):
            """
            Splits episode into action chunks and adds a corresponding padding mask.
            
            Args:
                traj (dict): A dictionary containing at least the key "actions".
                action_chunk_size (int): The size of the action chunks.

            Returns:
                dict: The modified trajectory dictionary with chunked "actions" and
                    a new "action_is_pad" boolean mask.
            """
            traj_len = tf.shape(traj["actions"])[0]
            original_indices = tf.broadcast_to(
                tf.range(action_chunk_size, dtype=tf.int32)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len, dtype=tf.int32)[:, None],
                [traj_len, action_chunk_size],
            )
            action_is_pad = original_indices >= traj_len
            traj["action_is_pad"] = action_is_pad
            action_chunk_indices = tf.minimum(original_indices, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        def format_convert(frame):
            # Convert int64 to int32 for PyTorch compatibility
            data_dict = {}
            obs = frame["observation"]
            jp = tf.cast(obs["joint_position"], tf.float32)
            gp = tf.cast(obs["gripper_position"], tf.float32)
            data_dict["state"] = tf.concat([jp, gp], axis=-1) 
            data_dict["action"] = tf.cast(frame["actions"], tf.float32)
            data_dict["raw_lang"] = frame["prompt"]
            data_dict["is_pad"] = frame["action_is_pad"]
            data_dict["image"] = tf.stack([obs["image"], obs["wrist_image"]], axis=0)  # (2, H, W, C)
            step_id =  tf.strings.split(frame['step_id'], sep="--")
            data_dict["episode_id"] = step_id[0] + "--" + step_id[1]
            data_dict["timestamp"] = tf.strings.to_number(step_id[2], out_type=tf.int32)
            # create empty dict as reasoning 
            data_dict["reasoning"] = {}
            return data_dict
        
        dataset = dataset.map(format_convert, num_parallel_calls)
        dataset = dataset.repeat()
        return dataset
        
    def __iter__(self):
        for data in self.dataset.as_numpy_iterator():
            data['raw_lang'] = data['raw_lang'].decode('utf-8')
            data['episode_id'] = data['episode_id'].decode('utf-8')
            
            # Convert image to torch tensor and ensure uint8 format
            image_np = data['image']
            image_np = ensure_uint8_image(image_np)
            data['image'] = torch.einsum('k h w c -> k c h w', torch.from_numpy(image_np))
            data['state'] = torch.from_numpy(data['state']).float()
            data['action'] = torch.from_numpy(data['action']).float()
            data['is_pad'] = torch.from_numpy(data['is_pad']).bool()
            if self.data_processor is not None:
                data = self.data_processor(data)
            yield data

    def download(self, url: str, cache_dir: str, force_download: bool = False, **kwargs) -> pathlib.Path:
        # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
        parsed = urllib.parse.urlparse(url)
        # Short circuit if this is a local path.
        if parsed.scheme == "":
            path = pathlib.Path(url)
            if not path.exists():
                raise FileNotFoundError(f"File not found at {url}")
            return path.resolve()
        cache_dir = pathlib.Path(cache_dir)
        local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
        local_path = local_path.resolve()
        # Check if the cache should be invalidated.
        invalidate_cache = False
        if local_path.exists():
            return local_path
        try:
            lock_path = local_path.with_suffix(".lock")
            with filelock.FileLock(lock_path):
                # Download the data to a local cache.
                logger.info(f"Downloading {url} to {local_path}")
                scratch_path = local_path.with_suffix(".partial")
                def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs):
                    """Download a file from a remote filesystem to the local cache, and return the local path."""
                    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
                    info = fs.info(url)
                    # Folders are represented by 0-byte objects with a trailing forward slash.
                    if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
                        total_size = fs.du(url)
                    else:
                        total_size = info["size"]
                    with tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
                        while not future.done():
                            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
                            pbar.update(current_size - pbar.n)
                            time.sleep(1)
                        pbar.update(total_size - pbar.n)
                _download_fsspec(url, scratch_path, **kwargs)
                shutil.move(scratch_path, local_path)
        except PermissionError as e:
            msg = (
                f"Local file permission error was encountered while downloading {url}. "
                f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
            )
            raise PermissionError(msg) from e
        return local_path
    
if __name__=='__main__':
    ds = DroidDataset(dataset_dir="/inspire/hdd/project/robot-action/public/data/droid", name='droid_100')
    data = next(iter(ds))

    ds_full = DroidDataset(dataset_dir="/inspire/hdd/project/robot-action/public/data/droid", name='droid', version="1.0.1",filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json")
    data2 = next(iter(ds))