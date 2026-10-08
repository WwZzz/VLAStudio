"""Reuse the HDF5 reader with explicit paths and the example's image convention."""
import os
from pathlib import Path
import numpy as np
from vlastudio.data_utils.datasets.libero_h5 import LiberoHDF5
from vlastudio.paths import cache_root


class ObjectDataset(LiberoHDF5):
    def __init__(self, root="", **kwargs):
        root = root or os.environ.get("LIBERO_DATASET_ROOT") or str(cache_root() / "data/libero")
        super().__init__(root=str(Path(root).expanduser()), **kwargs)

    def _download_dataset_if_needed(self, dataset_dir="", split=None):
        root = Path(dataset_dir)
        if len(list((root / "libero_object").glob("*.hdf5"))) < 10:
            from huggingface_hub import snapshot_download
            snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                              revision="f13aa24a3da8c43c7225569f28c562979fa0e35a",
                              allow_patterns=["libero_object/**"], local_dir=str(root))
        return str(root)

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        sample = super().load_onestep_from_episode(dataset_path, start_ts)
        sample["image"] = {key: np.ascontiguousarray(image[::-1, ::-1])
                           for key, image in sample["image"].items()}
        return sample

    def load_feat_from_episode(self, dataset_path, feats=None):
        sample = super().load_feat_from_episode(dataset_path, feats or [])
        for key in ("image_primary", "image_wrist"):
            if key in sample:
                sample[key] = np.ascontiguousarray(sample[key][:, ::-1, ::-1])
        return sample
