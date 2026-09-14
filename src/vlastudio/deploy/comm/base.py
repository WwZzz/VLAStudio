#!/usr/bin/env python3
"""
Base classes for communication layer (Server / Client).

All concrete implementations (TCP pickle, FastAPI HTTP/JSON, etc.) should inherit from these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class BaseServer(ABC):
    """Abstract base class for policy servers."""

    @abstractmethod
    def start(self) -> None:
        """Start the server (blocking)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the server gracefully."""
        ...


class BaseClient(ABC):
    """
    Abstract base class for policy clients.

    Clients must implement `send_meta_obs` which sends a MetaObs to the remote
    server and returns the full action chunk list (same format as MetaPolicy.inference).
    
    The `inference` method is the primary interface used by `inference_worker`,
    equivalent to MetaPolicy.inference(mobs).
    """

    @abstractmethod
    def send_meta_obs(self, meta_obs: Any, **kwargs) -> List[np.ndarray]:
        """
        Send a MetaObs to the server and return the full list of action arrays.

        Returns:
            List[np.ndarray]: Each element is an object-dtype array where each
            item is a dict compatible with MetaAction.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        ...

    def close(self) -> None:
        """Close any network connections. Default is no-op."""
        pass

    # -------------------------------------------------------------------------
    # Optional properties for compatibility with existing code
    # -------------------------------------------------------------------------
    @property
    def ctrl_space(self) -> str:
        return getattr(self, "_ctrl_space", "ee")

    @ctrl_space.setter
    def ctrl_space(self, value: str) -> None:
        self._ctrl_space = value

    @property
    def ctrl_type(self) -> str:
        return getattr(self, "_ctrl_type", "delta")

    @ctrl_type.setter
    def ctrl_type(self, value: str) -> None:
        self._ctrl_type = value

    def inference(self, mobs) -> List[np.ndarray]:
        """
        Run inference on the remote server and return the full action chunk list.
        
        This is the primary interface used by inference_worker, equivalent to
        MetaPolicy.inference(mobs). Returns all action steps without truncation;
        chunk management is delegated to client-side action_manager.
        """
        return self.send_meta_obs(mobs)
