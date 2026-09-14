"""
Shared Memory Communication Module for Policy Inference.

This module provides ultra-low-latency communication using shared memory:
- SHMPolicyServer: Reads MetaObs from client's SHM, runs inference, writes actions to its own SHM.
- SHMPolicyClient: Writes MetaObs to its own SHM, reads actions from server's SHM.

Usage:
    # Server side
    server = SHMPolicyServer(policy, obs_shm_name="policy_obs", action_shm_name="policy_action")
    server.start()  # Blocking loop

    # Client side
    client = SHMPolicyClient(obs_shm_name="policy_obs", action_shm_name="policy_action")
    mact_list = client.inference(mobs)  # Returns full action chunk list
"""

from .server import SHMPolicyServer
from .client import SHMPolicyClient

__all__ = ["SHMPolicyServer", "SHMPolicyClient"]

