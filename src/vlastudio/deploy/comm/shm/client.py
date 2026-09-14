#!/usr/bin/env python3
"""
Shared Memory Policy Client for Ultra-Low-Latency Inference.

Architecture:
1. Client creates {client_id}_com and {client_id}_obs SHM channels.
2. Client monitors server's {shm_name}_com for connection notification.
3. Server creates {client_id}_action and notifies client via {shm_name}_com.
4. Client writes observations to {client_id}_obs, reads actions from {client_id}_action.

This avoids network overhead entirely - suitable for same-machine deployment.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from vlastudio.deploy.shm_utils import SharedMemoryChannel
from vlastudio.benchmark.base import MetaObs

from ..base import BaseClient


def _generate_client_id() -> str:
    """Generate a unique client ID."""
    pid = os.getpid()
    short_uuid = uuid.uuid4().hex[:8]
    return f"c{pid}_{short_uuid}"


class SHMPolicyClient(BaseClient):
    """
    Shared Memory Policy Client.

    Creates {client_id}_com and {client_id}_obs, registers with server,
    waits for server to create {client_id}_action, then communicates.
    """

    def __init__(
        self,
        shm_name: str = "policy",
        client_id: Optional[str] = None,
        com_shm_size_mb: int = 4,
        obs_shm_size_mb: int = 64,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
        timeout_s: float = 30.0,
    ):
        """
        Args:
            shm_name: Base name for server's com SHM channel.
            client_id: Unique client identifier. If None, auto-generated.
            com_shm_size_mb: Size of communication SHM in MB.
            obs_shm_size_mb: Size of observation SHM in MB.
            ctrl_space: Control space ("ee" or "joint").
            ctrl_type: Control type ("delta" or "abs").
            timeout_s: Timeout for waiting on server response.
        """
        self.shm_name = shm_name
        self.client_id = client_id or _generate_client_id()
        self.com_shm_size_mb = com_shm_size_mb
        self.obs_shm_size_mb = obs_shm_size_mb
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.timeout_s = timeout_s

        # SHM channel names
        self.server_com_name = f"{shm_name}_com"
        self.client_com_name = f"{self.client_id}_com"
        self.obs_shm_name = f"{self.client_id}_obs"
        self.action_shm_name = f"{self.client_id}_action"

        # SHM channels
        self._client_com_channel: Optional[SharedMemoryChannel] = None
        self._server_com_channel: Optional[SharedMemoryChannel] = None
        self._obs_channel: Optional[SharedMemoryChannel] = None
        self._action_channel: Optional[SharedMemoryChannel] = None

        self._connected = False
        self._last_action_timestamp: float = 0.0

        self._connect()

    def _connect(self) -> None:
        """Initialize SHM channels and register with server."""
        # Create client's com channel (server reads, we write for heartbeat)
        self._client_com_channel = SharedMemoryChannel(
            name=self.client_com_name,
            max_size_mb=self.com_shm_size_mb,
            is_writer=True,
        )
        logger.info(f"✓ Created client com SHM: {self.client_com_name}")

        # Create observation SHM (client writes, server reads)
        self._obs_channel = SharedMemoryChannel(
            name=self.obs_shm_name,
            max_size_mb=self.obs_shm_size_mb,
            is_writer=True,
        )
        logger.info(f"✓ Created obs SHM: {self.obs_shm_name}")

        # Write initial registration message to client's com channel
        self._send_registration()

        # Connect to server's com channel to receive notifications
        logger.info(f"⏳ Connecting to server com SHM: {self.server_com_name}...")
        try:
            self._server_com_channel = SharedMemoryChannel(
                name=self.server_com_name,
                is_writer=False,
                timeout=self.timeout_s,
            )
            logger.info(f"✓ Connected to server com SHM: {self.server_com_name}")
        except TimeoutError:
            logger.warning(f"⚠ Server com SHM not available yet, will retry on first request")

        # Wait for server to create our action SHM
        logger.info(f"⏳ Waiting for server to create action SHM: {self.action_shm_name}...")
        try:
            self._action_channel = SharedMemoryChannel(
                name=self.action_shm_name,
                is_writer=False,
                timeout=self.timeout_s,
            )
            self._connected = True
            logger.info(f"✓ Connected to action SHM: {self.action_shm_name}")
        except TimeoutError:
            logger.warning(f"⚠ Action SHM not available yet, will retry on first request")

        logger.info(f"✓ SHM Policy Client ready (client_id: {self.client_id})")
        logger.info(f"   Client com: {self.client_com_name} (write)")
        logger.info(f"   Client obs: {self.obs_shm_name} (write)")
        logger.info(f"   Action: {self.action_shm_name} (read)")

    def _send_registration(self) -> None:
        """Send registration info to our com channel for server to discover."""
        registration = {
            "type": "register",
            "client_id": self.client_id,
            "com_shm": self.client_com_name,
            "obs_shm": self.obs_shm_name,
            "timestamp": time.time(),
        }
        self._client_com_channel.write(registration)

    def _ensure_action_channel(self) -> bool:
        """Ensure action channel is connected."""
        if self._action_channel is not None and self._action_channel.shm is not None:
            return True

        # Re-send registration
        self._send_registration()

        try:
            self._action_channel = SharedMemoryChannel(
                name=self.action_shm_name,
                is_writer=False,
                timeout=self.timeout_s,
            )
            self._connected = True
            logger.info(f"✓ Connected to action SHM: {self.action_shm_name}")
            return True
        except TimeoutError:
            logger.error(f"✗ Timeout waiting for action SHM: {self.action_shm_name}")
            return False

    def send_meta_obs(self, meta_obs: MetaObs, **kwargs) -> List[np.ndarray]:
        """
        Send MetaObs to server via SHM and wait for action response.

        Args:
            meta_obs: Observation to send.
            **kwargs: Ignored (for interface compatibility).

        Returns:
            List of action arrays (same format as other clients).
        """
        if not self._ensure_action_channel():
            raise RuntimeError(f"Action SHM not available: {self.action_shm_name}")

        # Convert MetaObs to dict and write to obs SHM
        obs_dict = self._metaobs_to_dict(meta_obs)
        self._obs_channel.write(obs_dict)

        # Update heartbeat
        self._send_heartbeat()

        # Wait for new action from server
        t_start = time.time()
        while True:
            action_data = self._action_channel.read(
                blocking=False,
                skip_unchanged=True,
            )

            if action_data is not None:
                return self._dict_to_mact_list(action_data)

            if time.time() - t_start > self.timeout_s:
                raise RuntimeError(f"Timeout waiting for server response ({self.timeout_s}s)")

            time.sleep(0.0001)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to keep connection alive."""
        heartbeat = {
            "type": "heartbeat",
            "client_id": self.client_id,
            "timestamp": time.time(),
        }
        self._client_com_channel.write(heartbeat)

    def _metaobs_to_dict(self, mobs: MetaObs) -> dict:
        """Convert MetaObs to dict for SHM transmission."""
        return asdict(mobs)

    def _dict_to_mact_list(self, data: dict) -> List[np.ndarray]:
        """Convert action dict from SHM to mact_list format."""
        mact_list_raw = data.get("mact_list", [])

        # Convert back to List[np.ndarray(dtype=object)]
        result = []
        for step in mact_list_raw:
            if isinstance(step, list):
                result.append(np.array(step, dtype=object))
            elif isinstance(step, np.ndarray):
                result.append(step.astype(object))
            else:
                result.append(np.array([step], dtype=object))

        return result

    def reset(self) -> None:
        """Reset internal state."""
        self._last_action_timestamp = 0.0
        logger.debug("  🔄 SHM policy client reset")

    def close(self, wait_timeout_s: float = 2.0) -> None:
        """
        Close SHM channels and notify server of disconnection.

        Args:
            wait_timeout_s: Seconds to wait for server acknowledgment before force cleanup.
        """
        if not self._connected and self._client_com_channel is None:
            return  # Already closed

        # Step 1: Notify server via our com channel (server monitors this)
        self._notify_server_disconnect()

        # Step 2: Wait for server to acknowledge (or timeout)
        server_acked = self._wait_for_server_ack(wait_timeout_s)
        if server_acked:
            logger.debug(f"  Server acknowledged disconnect")
        else:
            logger.debug(f"  Server ack timeout, force cleanup")

        # Step 4: Close server com channel handle (don't unlink - server owns it)
        if self._server_com_channel:
            try:
                if self._server_com_channel.shm:
                    self._server_com_channel.shm.close()
                    self._server_com_channel.shm = None
            except Exception:
                pass
            self._server_com_channel = None

        # Step 5: Close action channel handle (don't unlink - server owns it)
        if self._action_channel:
            try:
                if self._action_channel.shm:
                    self._action_channel.shm.close()
                    self._action_channel.shm = None
            except Exception:
                pass
            self._action_channel = None

        # Step 6: Force cleanup our own SHM (we own these)
        if self._client_com_channel:
            self._client_com_channel.destroy()
            self._client_com_channel = None

        if self._obs_channel:
            self._obs_channel.destroy()
            self._obs_channel = None

        self._connected = False
        logger.info(f"✓ SHM Policy Client closed (client_id: {self.client_id})")

    def _wait_for_server_ack(self, timeout_s: float) -> bool:
        """Wait for server to acknowledge disconnect via our com channel."""
        if not self._client_com_channel:
            return False

        t_start = time.time()
        while time.time() - t_start < timeout_s:
            try:
                # Check if server wrote ack to our com channel
                # We need to read from our own com channel (which we wrote to)
                # Server will overwrite with ack message
                data = self._client_com_channel.read(blocking=False, skip_unchanged=True)
                if data is not None:
                    msg_type = data.get("type", "")
                    if msg_type == "disconnect_ack":
                        return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def _notify_server_disconnect(self) -> None:
        """Notify server that we are disconnecting by writing to our own com channel."""
        # Server monitors our com channel in its cleanup loop
        # Just write disconnect message to our own com channel
        if self._client_com_channel:
            try:
                disconnect_msg = {
                    "type": "disconnect",
                    "client_id": self.client_id,
                    "timestamp": time.time(),
                }
                self._client_com_channel.write(disconnect_msg)
                logger.debug(f"  Sent disconnect notification")
            except Exception as e:
                logger.debug(f"  Failed to send disconnect: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except Exception:
            pass
