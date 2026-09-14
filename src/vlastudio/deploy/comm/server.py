"""
Policy Server for Remote Inference (TCP + pickle)

This module provides the PolicyServer class that listens for observation data 
and returns predicted actions over a network connection.
"""

import socket
import pickle
import struct
import threading
import torch
import traceback
from loguru import logger
from typing import Optional
from vlastudio.benchmark.base import MetaObs, MetaAction, MetaPolicy

from .base import BaseServer


class PolicyServer(BaseServer):
    """
    Policy server that listens for observation data and returns predicted actions.
    
    Communication protocol:
    1. Client sends: [4 bytes length] + [pickled MetaObs data]
    2. Server responds: [4 bytes length] + [pickled list of MetaAction]
    """
    
    def __init__(self, policy: MetaPolicy, host: str = '0.0.0.0', port: int = 5000):
        self.policy = policy
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_count = 0
        
    def start(self):
        """Start the policy server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"🚀 Policy Server started on {self.host}:{self.port}")
            logger.info("   Ctrl+C to stop the server")
            logger.info("   Waiting for connections...")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.client_count += 1
                    client_id = self.client_count
                    
                    logger.info(f"✓ Client #{client_id} connected from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address, client_id),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"✗ Error accepting connection: {e}")
                    
        finally:
            self.stop()
    
    def handle_client(self, client_socket: socket.socket, client_address, client_id: int):
        """Handle a single client connection"""
        request_count = 0
        
        try:
            while self.running:
                # Receive MetaObs from client
                meta_obs = self.receive_meta_obs(client_socket)
                if meta_obs is None:
                    logger.info(f"  Client #{client_id} disconnected")
                    break
                
                request_count += 1
                
                # Perform inference
                try:
                    # Use torch.no_grad() to avoid gradient computation during inference
                    with torch.no_grad():
                        mact_list = self.policy.inference(meta_obs)
                    
                    # Send action list back to client
                    self.send_mact_list(client_socket, mact_list)
                    
                    if request_count % 10 == 0:
                        logger.info(f"  Client #{client_id}: {request_count} requests processed")
                    
                except Exception as e:
                    logger.error(f"✗ Inference error for client #{client_id}: {e}")
                    # Send empty list on error
                    self.send_mact_list(client_socket, [])
                    
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"✗ Inference error for client #{client_id}:\n{error_trace}")
            # logger.error(f"✗ Error handling client #{client_id}: {e}")
        finally:
            client_socket.close()
            logger.info(f"  Client #{client_id} connection closed (processed {request_count} requests)")
    
    def receive_meta_obs(self, client_socket: socket.socket):
        """
        Receive MetaObs from client.
        
        Protocol: [4 bytes length (big-endian)] + [pickled data]
        
        Returns:
            MetaObs object or None if connection closed
        """
        try:
            # Read 4 bytes for data length
            length_bytes = self._recv_exactly(client_socket, 4)
            if not length_bytes:
                return None
            
            data_length = struct.unpack('>I', length_bytes)[0]
            
            # Read the actual data
            data_bytes = self._recv_exactly(client_socket, data_length)
            if not data_bytes:
                return None
            
            # Deserialize
            meta_obs = pickle.loads(data_bytes)
            return meta_obs
            
        except Exception as e:
            logger.error(f"✗ Error receiving MetaObs: {e}")
            return None
    
    def send_mact_list(self, client_socket: socket.socket, mact_list):
        """
        Send list of MetaAction to client.
        
        Protocol: [4 bytes length (big-endian)] + [pickled data]
        """
        try:
            # Serialize the action list
            data_bytes = pickle.dumps(mact_list)
            data_length = len(data_bytes)
            
            # Send length (4 bytes, big-endian)
            length_bytes = struct.pack('>I', data_length)
            client_socket.sendall(length_bytes)
            
            # Send data
            client_socket.sendall(data_bytes)
            
        except Exception as e:
            logger.error(f"✗ Error sending action list: {e}")
            raise
    
    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket.
        
        Returns:
            bytes or None if connection closed
        """
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        logger.info("✓ Policy Server stopped")
