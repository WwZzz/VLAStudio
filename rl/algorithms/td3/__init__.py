from .td3 import TD3Algorithm, TD3Config
from .. import register_algorithm

# Register algorithm with its config class
register_algorithm("td3", TD3Algorithm, TD3Config)

__all__ = ["TD3Algorithm", "TD3Config"]

