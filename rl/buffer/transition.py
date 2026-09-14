"""
RL transition data structures.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from benchmark.base import MetaObs, MetaAction


@dataclass
class RLTransition:
    """
    One-step transition for RL replay (raw environment meta data).
    """

    obs: MetaObs
    action: MetaAction
    next_obs: MetaObs
    reward: Any
    done: Any
    truncated: Optional[Any] = None
    info: Optional[Any] = None

    def to_batch(self) -> "RLTransition":
        """
        Ensure obs/action/next_obs are batched along env dimension.
        """
        if hasattr(self.obs, "to_batch"):
            self.obs.to_batch()
        if hasattr(self.action, "to_batch"):
            self.action.to_batch()
        if hasattr(self.next_obs, "to_batch"):
            self.next_obs.to_batch()
        return self

