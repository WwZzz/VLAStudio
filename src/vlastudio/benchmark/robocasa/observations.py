"""Shared, simulator-independent RoboCasa observation contract."""
import json
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
STATE_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')


def metadata(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return json.loads(value) if isinstance(value, str) else dict(value)


def state_vector(obs, keys=STATE_KEYS):
    values = []
    for key in keys:
        value = np.asarray(obs[key])
        if key.endswith('_quat'):
            value = Rotation.from_quat(value).as_rotvec()
        values.append(value)
    result = np.concatenate(values, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('RoboCasa state contains non-finite values')
    return result


def rgb_image(value, flip=False):
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('RoboCasa RGB observations must be uint8 HWC arrays')
    return np.ascontiguousarray(image[::-1] if flip else image)


def instruction(ep_meta):
    text = metadata(ep_meta).get('lang')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('RoboCasa episode metadata must contain a nonempty lang instruction')
    return text
