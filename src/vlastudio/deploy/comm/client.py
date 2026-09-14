"""
Remote Policy Client for Evaluation (TCP + pickle)

This module provides the PolicyClient class that sends observations to a
remote policy server over TCP and receives action chunks back.
Action selection / chunk management is handled by action_manager.
"""

import socket
import pickle
import struct
import re
import os
from typing import Optional, List
import numpy as np
from loguru import logger
from vlastudio.benchmark.base import MetaObs, MetaAction

from .base import BaseClient


class PolicyClient(BaseClient):
    """
    Remote Policy Client that communicates with a policy server over TCP.
    
    Provides `send_meta_obs` / `inference` interface for sending observations
    and receiving the full action chunk list from the server.
    Action queue management is delegated to the action_manager.
    """
    
    def __init__(self, host: str, port: int, ctrl_space: str = 'ee', ctrl_type: str = 'delta'):
        self.host = host
        self.port = port
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.socket: Optional[socket.socket] = None
        
        # Connect to server
        self._connect()
        
        logger.info(f"✓ Connected to policy server at {host}:{port}")
    
    def _connect(self):
        """Connect to the policy server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to policy server at {self.host}:{self.port}: {e}")
    
    def _disconnect(self):
        """Disconnect from the policy server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def send_meta_obs(self, meta_obs: MetaObs, **kwargs) -> List:
        """
        Send MetaObs to server and receive MetaAction list.
        
        Args:
            meta_obs: MetaObs object to send
            **kwargs: Ignored (for interface compatibility with FastAPI client)
            
        Returns:
            List of MetaAction-like objects (numpy object arrays)
        """
        if not self.socket:
            raise RuntimeError("Not connected to server")
        
        try:
            # Serialize MetaObs
            data_bytes = pickle.dumps(meta_obs)
            data_length = len(data_bytes)
            
            # Send length (4 bytes, big-endian)
            length_bytes = struct.pack('>I', data_length)
            self.socket.sendall(length_bytes)
            
            # Send data
            self.socket.sendall(data_bytes)
            
            # Receive response
            return self._receive_mact_list()
            
        except Exception as e:
            logger.error(f"✗ Error communicating with server: {e}")
            # Try to reconnect once
            try:
                self._disconnect()
                self._connect()
                logger.info("✓ Reconnected to server, retrying...")
                return self.send_meta_obs(meta_obs, **kwargs)
            except:
                raise RuntimeError(f"Failed to communicate with server and reconnection failed: {e}")

    def _receive_mact_list(self) -> List[MetaAction]:
        """
        Receive list of MetaAction from server.
        
        Returns:
            List of MetaAction objects
        """
        try:
            # Read 4 bytes for data length
            length_bytes = self._recv_exactly(4)
            if not length_bytes:
                return []
            
            data_length = struct.unpack('>I', length_bytes)[0]
            
            # Read the actual data
            data_bytes = self._recv_exactly(data_length)
            if not data_bytes:
                return []
            
            # Deserialize
            mact_list = pickle.loads(data_bytes)
            return mact_list
            
        except Exception as e:
            logger.error(f"✗ Error receiving MetaAction list: {e}")
            return []
    
    def _recv_exactly(self, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket.
        
        Returns:
            bytes or None if connection closed
        """
        data = b''
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def reset(self):
        """Reset internal state (no-op for TCP client)."""
        logger.debug("  🔄 Remote policy reset")

    def close(self):
        """Close the connection to the server."""
        self._disconnect()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.close()


def parse_server_address(model_path: str) -> tuple:
    """
    Parse server address from model path.
    
    Expected format: "host:port" or "ip:port"
    
    Args:
        model_path: String in format "host:port"
        
    Returns:
        tuple: (host, port)
        
    Raises:
        ValueError: If format is invalid
    """
    # Check if it looks like a server address (contains colon and port number)
    if ':' in model_path and not os.path.exists(model_path):
        # Try to parse as host:port
        match = re.match(r'^(.+):(\d+)$', model_path)
        if match:
            host = match.group(1)
            port = int(match.group(2))
            return host, port
    
    raise ValueError(f"Invalid server address format: {model_path}. Expected format: 'host:port'")


def is_server_address(model_path: str) -> bool:
    """
    Check if model_path looks like a server address.
    
    Args:
        model_path: Path or server address string
        
    Returns:
        bool: True if it looks like a server address
    """
    try:
        parse_server_address(model_path)
        return True
    except ValueError:
        return False
