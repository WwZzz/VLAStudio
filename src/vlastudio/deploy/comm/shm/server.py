#!/usr/bin/env python3
"""
Shared Memory Policy Server for Ultra-Low-Latency Batched Inference.

Architecture:
1. Server creates main com_shm ({shm_name}_com) to receive client registration requests.
2. Client sends registration to com_shm with its {client_id}_com and {client_id}_obs addresses.
3. Server connects to client's com_shm and obs_shm, creates {client_id}_action for response.
4. Server notifies client via {client_id}_com that action_shm is ready.
5. Server batches observations from multiple clients, runs inference, writes actions back.
6. Clients inactive for >timeout are disconnected and their SHM is cleaned up.
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from vlastudio.deploy.shm_utils import SharedMemoryChannel
from vlastudio.benchmark.base import MetaObs, dict2meta

from ..base import BaseServer


@dataclass
class ClientInfo:
    """Information about a connected client."""
    client_id: str
    com_channel: SharedMemoryChannel  # Server reads from client's com
    obs_channel: SharedMemoryChannel  # Server reads observations
    action_channel: SharedMemoryChannel  # Server writes actions
    last_active_time: float
    connected_at: float


class SHMPolicyServer(BaseServer):
    """
    Shared Memory Policy Server with batched inference.

    - Accepts client registrations via {shm_name}_com
    - Batches observations from multiple clients
    - Writes actions to per-client {client_id}_action SHM
    - Cleans up inactive clients after timeout
    """

    def __init__(
        self,
        policy,
        shm_name: str = "policy",
        com_shm_size_mb: int = 4,
        action_shm_size_mb: int = 16,
        poll_interval_s: float = 0.0001,
        client_timeout_s: float = 60.0,
        batch_wait_ms: float = 1.0,
    ):
        """
        Args:
            policy: The MetaPolicy to serve.
            shm_name: Base name for server's com SHM channel.
            com_shm_size_mb: Size of communication SHM channels.
            action_shm_size_mb: Size of each action SHM in MB.
            poll_interval_s: Polling interval when waiting for observations.
            client_timeout_s: Disconnect clients inactive for this duration.
            batch_wait_ms: Wait time to collect observations before batching (ms).
        """
        self.policy = policy
        self.shm_name = shm_name
        self.com_shm_size_mb = com_shm_size_mb
        self.action_shm_size_mb = action_shm_size_mb
        self.poll_interval_s = poll_interval_s
        self.client_timeout_s = client_timeout_s
        self.batch_wait_ms = batch_wait_ms

        # Main communication channel for registration
        self._com_channel: Optional[SharedMemoryChannel] = None
        self.com_shm_name = f"{shm_name}_com"

        # Connected clients
        self._clients: Dict[str, ClientInfo] = {}
        self._clients_lock = threading.Lock()

        # Disconnected clients pending cleanup (client_id -> disconnect_time)
        self._pending_cleanup: Dict[str, float] = {}

        # Pending observations queue: (client_id, obs_data, timestamp)
        self._obs_queue: deque = deque(maxlen=1000)

        self._running = False
        self._request_count = 0

    def start(self) -> None:
        """Start the server (blocking inference loop)."""
        self._running = True

        # Clean up any stale SHM from previous runs
        self._cleanup_stale_shm()

        # Create main com SHM for client registration
        self._com_channel = SharedMemoryChannel(
            name=self.com_shm_name,
            max_size_mb=self.com_shm_size_mb,
            is_writer=True,  # Server writes responses/status
        )
        logger.info(f"✓ Created server com SHM: {self.com_shm_name}")

        logger.info("🚀 SHM Policy Server started")
        logger.info(f"   Server com SHM: {self.com_shm_name}")
        logger.info(f"   Client timeout: {self.client_timeout_s}s")
        logger.info(f"   Batch wait: {self.batch_wait_ms}ms")

        # Start client discovery thread
        self._discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self._discovery_thread.start()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        # Main inference loop
        self._inference_loop()

    def _cleanup_stale_shm(self) -> None:
        """Clean up stale SHM segments from previous server runs."""
        shm_dir = "/dev/shm"
        if not os.path.exists(shm_dir):
            return

        stale_count = 0
        stale_clients = set()

        try:
            for name in os.listdir(shm_dir):
                # Clean up server's own com channel from previous run
                if name == self.com_shm_name:
                    self._unlink_shm(name)
                    stale_count += 1
                    continue

                # Clean up action channels created by this server (pattern: *_action)
                # These are created by server, so server should clean them
                if name.endswith("_action"):
                    self._unlink_shm(name)
                    stale_count += 1
                    continue

                # Detect stale client com/obs channels by checking timestamp
                if name.endswith("_com") or name.endswith("_obs"):
                    try:
                        from multiprocessing import shared_memory
                        shm = shared_memory.SharedMemory(name=name)

                        # Try to read and check if stale (older than timeout)
                        # For simplicity, we'll check file modification time
                        shm_path = os.path.join(shm_dir, name)
                        mtime = os.path.getmtime(shm_path)
                        age = time.time() - mtime

                        shm.close()

                        if age > self.client_timeout_s:
                            # Stale client SHM
                            if name.endswith("_com"):
                                client_id = name[:-4]
                            else:  # _obs
                                client_id = name[:-4]
                            stale_clients.add(client_id)

                    except Exception:
                        pass

            # Clean up stale clients
            for client_id in stale_clients:
                for suffix in ["_com", "_obs", "_action"]:
                    self._unlink_shm(f"{client_id}{suffix}")
                    stale_count += 1

            if stale_count > 0:
                logger.info(f"  Cleaned up {stale_count} stale SHM segments from previous run")

        except Exception as e:
            logger.warning(f"  Error cleaning stale SHM: {e}")

    def _unlink_shm(self, name: str) -> None:
        """Safely unlink a shared memory segment."""
        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"  Could not unlink SHM {name}: {e}")

    def _client_shm_exists(self, client_id: str) -> bool:
        """Check if any SHM for this client still exists."""
        shm_dir = "/dev/shm"
        com_name = f"{client_id}_com"
        obs_name = f"{client_id}_obs"
        action_name = f"{client_id}_action"

        for name in [com_name, obs_name, action_name]:
            if os.path.exists(os.path.join(shm_dir, name)):
                return True
        return False

    def _discovery_loop(self) -> None:
        """Background thread to discover and connect new clients."""
        shm_dir = "/dev/shm"

        while self._running:
            try:
                if not os.path.exists(shm_dir):
                    time.sleep(1.0)
                    continue

                # Scan for client com channels: {client_id}_com
                for name in os.listdir(shm_dir):
                    if name.endswith("_com") and not name.startswith(self.shm_name):
                        # Potential client com channel
                        client_id = name[:-4]  # Remove "_com" suffix

                        # Skip if already connected or it's server's own channel
                        if client_id in self._clients or name == self.com_shm_name:
                            continue

                        # Check if obs channel exists
                        obs_name = f"{client_id}_obs"
                        if os.path.exists(os.path.join(shm_dir, obs_name)):
                            self._try_connect_client(client_id)

                time.sleep(0.5)  # Scan every 500ms

            except Exception as e:
                logger.error(f"Discovery error: {e}")
                time.sleep(1.0)

    def _try_connect_client(self, client_id: str) -> bool:
        """Attempt to connect to a new client."""
        com_name = f"{client_id}_com"
        obs_name = f"{client_id}_obs"
        action_name = f"{client_id}_action"

        try:
            with self._clients_lock:
                if client_id in self._clients:
                    return True  # Already connected

                # Check if pending cleanup - only reconnect if SHM is fully cleaned
                if client_id in self._pending_cleanup:
                    if self._client_shm_exists(client_id):
                        return False  # SHM still exists, don't reconnect
                    else:
                        # SHM cleaned, remove from pending and allow reconnect
                        del self._pending_cleanup[client_id]

                # Connect to client's com channel (server reads registration)
                com_channel = SharedMemoryChannel(
                    name=com_name,
                    is_writer=False,
                    timeout=2.0,
                )

                # Verify client is alive by checking its com channel data
                com_data = com_channel.read(blocking=False)
                if com_data is not None:
                    # Check if this is a stale client (disconnect message or old timestamp)
                    msg_type = com_data.get("type", "")
                    if msg_type == "disconnect":
                        logger.info(f"  Cleaning up disconnected client SHM: {client_id}")
                        com_channel.shm.close()
                        # Clean up all client SHM
                        self._unlink_shm(com_name)
                        self._unlink_shm(obs_name)
                        self._unlink_shm(action_name)
                        return False

                    # Check timestamp - if too old, consider it stale
                    timestamp = com_data.get("timestamp", 0)
                    if time.time() - timestamp > self.client_timeout_s:
                        logger.info(f"  Cleaning up stale client SHM: {client_id} (inactive {time.time() - timestamp:.1f}s)")
                        com_channel.shm.close()
                        # Clean up all stale client SHM
                        self._unlink_shm(com_name)
                        self._unlink_shm(obs_name)
                        self._unlink_shm(action_name)
                        return False

                # Connect to client's obs channel
                obs_channel = SharedMemoryChannel(
                    name=obs_name,
                    is_writer=False,
                    timeout=2.0,
                )

                # Create action channel for this client (server writes)
                action_channel = SharedMemoryChannel(
                    name=action_name,
                    max_size_mb=self.action_shm_size_mb,
                    is_writer=True,
                )

                now = time.time()
                client_info = ClientInfo(
                    client_id=client_id,
                    com_channel=com_channel,
                    obs_channel=obs_channel,
                    action_channel=action_channel,
                    last_active_time=now,
                    connected_at=now,
                )
                self._clients[client_id] = client_info

            # Notify client via server's com channel
            self._notify_client_connected(client_id, action_name)

            logger.info(f"✓ Client connected: {client_id}")
            logger.info(f"   Client com: {com_name}")
            logger.info(f"   Client obs: {obs_name}")
            logger.info(f"   Action SHM: {action_name}")
            return True

        except Exception as e:
            logger.debug(f"Failed to connect client {client_id}: {e}")
            return False

    def _notify_client_connected(self, client_id: str, action_shm_name: str) -> None:
        """Notify client that connection is established."""
        # Write to server's com channel (clients read this)
        notification = {
            "type": "connected",
            "client_id": client_id,
            "action_shm": action_shm_name,
            "timestamp": time.time(),
        }
        self._com_channel.write(notification)

    def _cleanup_loop(self) -> None:
        """Background thread to cleanup inactive clients and check for disconnect messages."""
        while self._running:
            try:
                now = time.time()
                clients_to_remove = []

                with self._clients_lock:
                    for client_id, info in self._clients.items():
                        # Check for timeout
                        if now - info.last_active_time > self.client_timeout_s:
                            clients_to_remove.append((client_id, "timeout"))
                            continue

                        # Check for disconnect message in client's com channel
                        try:
                            com_data = info.com_channel.read(blocking=False)
                            if com_data is not None:
                                msg_type = com_data.get("type", "")
                                if msg_type == "disconnect":
                                    clients_to_remove.append((client_id, "client_disconnect"))
                        except Exception:
                            pass

                for client_id, reason in clients_to_remove:
                    self._disconnect_client(client_id, reason=reason)

                time.sleep(1.0)  # Check every 1 second for faster disconnect response

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(1.0)

    def _disconnect_client(self, client_id: str, reason: str = "unknown") -> None:
        """Disconnect and cleanup a client."""
        with self._clients_lock:
            if client_id not in self._clients:
                return

            info = self._clients.pop(client_id)

            # Add to pending cleanup - block reconnect until SHM is cleaned
            self._pending_cleanup[client_id] = time.time()

        # Step 1: Destroy action channel first (server owns this SHM)
        try:
            info.action_channel.destroy()
        except Exception:
            pass

        # Step 2: Send disconnect_ack to client's com channel
        self._send_disconnect_ack(client_id, info.com_channel)

        # Step 3: Close our handle to client's com channel
        try:
            if info.com_channel.shm:
                info.com_channel.shm.close()
                info.com_channel.shm = None
        except Exception:
            pass

        # Step 4: Close our handle to client's obs channel
        try:
            if info.obs_channel.shm:
                info.obs_channel.shm.close()
                info.obs_channel.shm = None
        except Exception:
            pass

        logger.info(f"  Client disconnected: {client_id} (reason: {reason})")

        # Step 5: Wait briefly then cleanup client's SHM if still exists
        # This handles the case where client failed to cleanup
        self._schedule_client_shm_cleanup(client_id)

    def _schedule_client_shm_cleanup(self, client_id: str) -> None:
        """Schedule cleanup of client's SHM after a delay."""
        def delayed_cleanup():
            time.sleep(3.0)  # Wait for client to cleanup
            com_name = f"{client_id}_com"
            obs_name = f"{client_id}_obs"
            action_name = f"{client_id}_action"

            cleaned = False
            for name in [com_name, obs_name, action_name]:
                try:
                    from multiprocessing import shared_memory
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                    cleaned = True
                except FileNotFoundError:
                    pass  # Already cleaned by client
                except Exception:
                    pass

            if cleaned:
                logger.debug(f"  Server cleaned up residual SHM for {client_id}")

        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()

    def _send_disconnect_ack(self, client_id: str, com_channel: SharedMemoryChannel) -> None:
        """Send disconnect acknowledgment to client's com channel."""
        try:
            # We need to write to client's com channel
            # Create a temporary writer since com_channel is reader mode
            com_name = f"{client_id}_com"
            com_writer = SharedMemoryChannel(
                name=com_name,
                max_size_mb=self.com_shm_size_mb,
                is_writer=True,
            )
            ack_msg = {
                "type": "disconnect_ack",
                "client_id": client_id,
                "timestamp": time.time(),
            }
            com_writer.write(ack_msg)
            # Don't destroy - just close handle (client owns this SHM)
            if com_writer.shm:
                com_writer.shm.close()
                com_writer.shm = None
            logger.debug(f"  Sent disconnect_ack to {client_id}")
        except Exception as e:
            logger.debug(f"  Failed to send disconnect_ack to {client_id}: {e}")

    def _inference_loop(self) -> None:
        """Main inference loop: collect observations, batch, infer, distribute."""
        with torch.no_grad():
            while self._running:
                try:
                    # Collect observations from all clients
                    batch_data = self._collect_observations()

                    if not batch_data:
                        time.sleep(self.poll_interval_s)
                        continue

                    # Run batched inference
                    results = self._batch_inference(batch_data)

                    # Distribute results to clients
                    self._distribute_actions(results)

                    self._request_count += len(batch_data)
                    if self._request_count % 100 == 0:
                        logger.debug(f"  Processed {self._request_count} requests, "
                                   f"{len(self._clients)} active clients")

                except KeyboardInterrupt:
                    logger.info("⏸ Server interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"✗ Inference loop error: {e}")
                    import traceback
                    traceback.print_exc()

    def _collect_observations(self) -> List[Tuple[str, dict]]:
        """Collect new observations from all clients."""
        collected = []
        now = time.time()

        with self._clients_lock:
            clients_snapshot = list(self._clients.items())

        for client_id, info in clients_snapshot:
            try:
                obs_data = info.obs_channel.read(
                    blocking=False,
                    skip_unchanged=True,
                )

                if obs_data is not None:
                    collected.append((client_id, obs_data))
                    # Update last active time
                    with self._clients_lock:
                        if client_id in self._clients:
                            self._clients[client_id].last_active_time = now

            except Exception as e:
                logger.debug(f"Error reading obs from {client_id}: {e}")
                # Mark for potential disconnection on repeated failures
                continue

        # Optional: wait a bit to collect more observations for better batching
        if collected and self.batch_wait_ms > 0:
            time.sleep(self.batch_wait_ms / 1000.0)

            # Collect any additional observations that arrived
            for client_id, info in clients_snapshot:
                if any(c[0] == client_id for c in collected):
                    continue  # Already collected from this client

                try:
                    obs_data = info.obs_channel.read(
                        blocking=False,
                        skip_unchanged=True,
                    )
                    if obs_data is not None:
                        collected.append((client_id, obs_data))
                        with self._clients_lock:
                            if client_id in self._clients:
                                self._clients[client_id].last_active_time = now
                except Exception:
                    pass

        return collected

    def _batch_inference(self, batch_data: List[Tuple[str, dict]]) -> Dict[str, list]:
        """
        Run batched inference on collected observations.

        Args:
            batch_data: List of (client_id, obs_dict)

        Returns:
            Dict mapping client_id -> mact_list
        """
        if not batch_data:
            return {}

        # Convert all obs dicts to MetaObs
        client_ids = []
        mobs_list = []

        for client_id, obs_dict in batch_data:
            try:
                filtered = {k: v for k, v in obs_dict.items() if not k.startswith("__")}
                mobs = dict2meta(filtered, mtype="obs")
                client_ids.append(client_id)
                mobs_list.append(mobs)
            except Exception as e:
                logger.error(f"Error converting obs from {client_id}: {e}")
                continue

        if not mobs_list:
            return {}

        # Batch the MetaObs
        batched_mobs = self._batch_metaobs(mobs_list)

        # Uses MetaPolicy.inference only (same as TCP PolicyServer); no separate denorm path like FastAPI JSON.
        try:
            mact_list = self.policy.inference(batched_mobs)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {}

        # Unbatch results
        results = self._unbatch_mact_list(mact_list, client_ids)
        return results

    def _batch_metaobs(self, mobs_list: List[MetaObs]) -> MetaObs:
        """Batch multiple MetaObs into one."""
        if len(mobs_list) == 1:
            return mobs_list[0]

        # Helper to safely concatenate arrays
        def safe_concat(attr_name):
            arrays = [getattr(m, attr_name) for m in mobs_list if getattr(m, attr_name, None) is not None]
            if not arrays:
                return None
            return np.concatenate(arrays, axis=0)

        # Stack all fields along batch dimension
        batched = MetaObs(
            state=safe_concat('state'),
            state_ee=safe_concat('state_ee'),
            state_joint=safe_concat('state_joint'),
            state_obj=safe_concat('state_obj'),
            image=safe_concat('image'),
            depth=safe_concat('depth'),
            pc=safe_concat('pc'),
            timestep=safe_concat('timestep'),
            raw_lang=mobs_list[0].raw_lang if mobs_list[0].raw_lang else '',
        )
        return batched

    def _unbatch_mact_list(self, mact_list: list, client_ids: List[str]) -> Dict[str, list]:
        """
        Unbatch mact_list back to per-client results.

        Args:
            mact_list: Batched action list from inference
            client_ids: List of client IDs in batch order

        Returns:
            Dict mapping client_id -> per-client mact_list
        """
        n_clients = len(client_ids)

        if n_clients == 1:
            return {client_ids[0]: mact_list}

        # mact_list is List[np.ndarray(dtype=object)] where each array has batch_size elements
        results = {cid: [] for cid in client_ids}

        for step_arr in mact_list:
            if isinstance(step_arr, np.ndarray) and len(step_arr) == n_clients:
                # Split by client
                for i, cid in enumerate(client_ids):
                    results[cid].append(np.array([step_arr[i]], dtype=object))
            else:
                # Can't unbatch properly, give same result to all
                for cid in client_ids:
                    results[cid].append(step_arr)

        return results

    def _distribute_actions(self, results: Dict[str, list]) -> None:
        """Write action results to each client's action SHM."""
        for client_id, mact_list in results.items():
            with self._clients_lock:
                if client_id not in self._clients:
                    continue
                info = self._clients[client_id]

            try:
                action_data = self._mact_list_to_dict(mact_list)
                info.action_channel.write(action_data)
            except Exception as e:
                logger.error(f"Error writing action to {client_id}: {e}")

    def _mact_list_to_dict(self, mact_list: list) -> dict:
        """Convert mact_list to dict for SHM transmission."""
        serialized = []
        for step_arr in mact_list:
            if isinstance(step_arr, np.ndarray):
                step_list = [dict(item) if isinstance(item, dict) else item for item in step_arr]
                for item in step_list:
                    if isinstance(item, dict) and "action" in item:
                        action = item["action"]
                        if isinstance(action, np.ndarray):
                            item["action"] = action
                serialized.append(step_list)
            else:
                serialized.append(step_arr)

        return {
            "mact_list": serialized,
            "count": len(serialized),
            "timestamp": time.time(),
        }

    def stop(self) -> None:
        """Stop the server and cleanup all resources."""
        self._running = False

        # Disconnect all clients
        with self._clients_lock:
            client_ids = list(self._clients.keys())

        for client_id in client_ids:
            self._disconnect_client(client_id, reason="server_shutdown")

        # Cleanup main com channel
        if self._com_channel:
            self._com_channel.destroy()
            self._com_channel = None

        logger.info(f"✓ SHM Policy Server stopped (processed {self._request_count} requests)")
