"""Shared FastAPI observation codecs (JPEG images, legacy-server prep)."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from benchmark.base import MetaObs
from benchmark.utils import resize_with_pad

NDARRAY_TYPE_TAG = "__type__"
NDARRAY_TYPE_NAME = "ndarray"
JPEG_TYPE_NAME = "jpeg_nchw"


def encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr)}")
    return {
        NDARRAY_TYPE_TAG: NDARRAY_TYPE_NAME,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(int(x) for x in obj["shape"])
        raw = base64.b64decode(obj["data"])
    except Exception as e:
        raise ValueError(f"Invalid ndarray payload: {e}") from e
    arr = np.frombuffer(raw, dtype=dtype)
    try:
        return arr.reshape(shape)
    except Exception as e:
        raise ValueError(f"Invalid ndarray shape {shape} for buffer: {e}") from e


def encode_images_jpeg(
    image: np.ndarray,
    *,
    quality: int = 85,
) -> Dict[str, Any]:
    """Encode (K,C,H,W) uint8 RGB as per-camera JPEG (much smaller on the wire)."""
    img = np.asarray(image)
    if img.ndim != 4:
        raise ValueError(f"jpeg encode expects (K,C,H,W), got {img.shape}")
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    frames: List[str] = []
    for i in range(img.shape[0]):
        # NCHW RGB → HWC BGR for cv2
        bgr = np.transpose(img[i], (1, 2, 0))[:, :, ::-1]
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for camera {i}")
        frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    return {
        NDARRAY_TYPE_TAG: JPEG_TYPE_NAME,
        "shape": list(img.shape),
        "quality": int(quality),
        "frames": frames,
    }


def decode_images_jpeg(obj: Dict[str, Any]) -> np.ndarray:
    shape = tuple(int(x) for x in obj["shape"])
    if len(shape) != 4:
        raise ValueError(f"jpeg_nchw shape must be (K,C,H,W), got {shape}")
    frames = obj.get("frames")
    if not isinstance(frames, list) or len(frames) != shape[0]:
        raise ValueError(
            f"jpeg_nchw frames length {0 if frames is None else len(frames)} "
            f"!= K={shape[0]}"
        )

    cams = []
    for i, b64 in enumerate(frames):
        raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cv2.imdecode failed for camera {i}")
        rgb = bgr[:, :, ::-1]
        if rgb.shape[0] != shape[2] or rgb.shape[1] != shape[3]:
            rgb = cv2.resize(rgb, (shape[3], shape[2]), interpolation=cv2.INTER_LINEAR)
        cams.append(np.transpose(rgb, (2, 0, 1)))  # CHW
    out = np.stack(cams, axis=0).astype(np.uint8)
    if tuple(out.shape) != shape:
        raise ValueError(f"decoded jpeg shape {out.shape} != expected {shape}")
    return out


def letterbox_images(
    image: np.ndarray,
    image_size: Union[int, Sequence[int]] = (256, 256),
    pad_value: int = 0,
) -> np.ndarray:
    """Letterbox (K,C,H,W) to (K,C,H',W') with the same pad convention as ACT training."""
    if isinstance(image_size, int):
        width = height = int(image_size)
    else:
        width, height = int(image_size[0]), int(image_size[1])

    img = np.asarray(image)
    if img.ndim == 3:
        img = img[None, ...]
    if img.ndim != 4:
        raise ValueError(f"letterbox expects (K,C,H,W) or (C,H,W), got {img.shape}")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and float(img.max()) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    _, _, h, w = img.shape
    if h == height and w == width:
        return np.ascontiguousarray(img)

    out = resize_with_pad(
        img,
        width=width,
        height=height,
        pad_value=pad_value,
    )
    return np.ascontiguousarray(out.astype(np.uint8, copy=False))


def prepare_meta_obs_for_remote(
    meta_obs: MetaObs,
    *,
    prefer_state_ee: bool = True,
    image_size: Optional[Union[int, Sequence[int]]] = (256, 256),
    pad_value: int = 0,
) -> MetaObs:
    """
    Adapt a local MetaObs for older policy servers:
    - EE policies on Alicia publish state=qpos + state_ee=FK; legacy MetaPolicy
      only reads ``state``, so copy state_ee → state.
    - Optionally letterbox-resize images to the training resolution.
    """
    state = meta_obs.state
    state_ee = getattr(meta_obs, "state_ee", None)
    state_joint = getattr(meta_obs, "state_joint", None)

    if prefer_state_ee and state_ee is not None:
        state = np.asarray(state_ee, dtype=np.float32).copy()
    elif state is not None:
        state = np.asarray(state, dtype=np.float32).copy()

    image = meta_obs.image
    if image is not None and image_size is not None:
        image = letterbox_images(image, image_size=image_size, pad_value=pad_value)

    return MetaObs(
        state=state,
        state_ee=None if state_ee is None else np.asarray(state_ee, dtype=np.float32).copy(),
        state_joint=None if state_joint is None else np.asarray(state_joint, dtype=np.float32).copy(),
        state_obj=meta_obs.state_obj,
        image=image,
        depth=meta_obs.depth,
        pc=meta_obs.pc,
        raw_lang=meta_obs.raw_lang,
        timestep=meta_obs.timestep,
    )


def payload_nbytes_estimate(meta_obs_jsonable: Dict[str, Any]) -> Tuple[int, str]:
    """Rough on-wire size helper for logging."""
    import json

    raw = json.dumps(meta_obs_jsonable, separators=(",", ":")).encode("utf-8")
    return len(raw), f"{len(raw) / 1e6:.2f}MB"
