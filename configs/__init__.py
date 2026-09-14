import os
import warnings
import logging

# Suppress TensorFlow warnings - must be set before TensorFlow is imported anywhere
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Only show ERROR logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN messages

# Suppress all warnings from third-party libraries to keep console clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress robosuite logging warnings
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
logging.getLogger("robosuite").setLevel(logging.ERROR)

# Suppress other common simulation library warnings
logging.getLogger("mujoco").setLevel(logging.ERROR)
logging.getLogger("dm_control").setLevel(logging.ERROR)

_default_cache = os.path.join(os.environ['VLASTUDIO_CACHE_DIR'], 'data') if os.environ.get('VLASTUDIO_CACHE_DIR') else os.path.join(os.path.expanduser('~'), '.cache/ilstd')
ILSTD_CACHE = os.environ.get('ILSTD_CACHE', _default_cache)
os.makedirs(ILSTD_CACHE, exist_ok=True)

import torch
import numpy as np

safe_types = [
    np.ndarray,
    np.dtype,
    np.core.multiarray._reconstruct,
]
for name in dir(np.dtypes):
    obj = getattr(np.dtypes, name)
    if isinstance(obj, type):
        safe_types.append(obj)
torch.serialization.add_safe_globals(safe_types)