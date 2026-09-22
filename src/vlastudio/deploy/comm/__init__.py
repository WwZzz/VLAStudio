"""
Communication Module for Remote Policy Inference (deploy.comm)

This module provides a unified interface for server/client communication:
- TCP + pickle (default): PolicyServer, PolicyClient
- HTTP/JSON (FastAPI):    FastAPIPolicyServer, FastAPIPolicyClient
- Shared Memory (SHM):    SHMPolicyServer, SHMPolicyClient

Use the factory functions `create_server` / `create_client` for transparent
transport selection, or use the address detection helpers `is_server_address`,
`is_http_address`, and `is_shm_address`.

Address formats:
- TCP:  host:port  (e.g., "localhost:5000")
- HTTP: http(s)://host[:port]  (e.g., "http://0.0.0.0:8000")
- SHM:  shm://obs_name/action_name  (e.g., "shm://policy_obs/policy_action")
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
# Base classes and transports are loaded on demand; address helpers stay lightweight.

# ---------------------------------------------------------------------------
# TCP + pickle implementation (default)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# FastAPI HTTP/JSON implementation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared Memory implementation
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Address parsing & detection utilities
# ---------------------------------------------------------------------------

def is_http_address(address: str) -> bool:
    """Return True if address starts with http:// or https://."""
    return address.startswith("http://") or address.startswith("https://")


def is_shm_address(address: str) -> bool:
    """Return True if address starts with shm://."""
    return address.startswith("shm://")


def parse_shm_address(address: str) -> str:
    """
    Parse SHM address into base shm_name.

    Expected format: shm://shm_name

    Returns:
        shm_name (base name for shared memory channels)
    """
    if not is_shm_address(address):
        raise ValueError(f"Not a SHM address: {address}")

    # Remove shm:// prefix
    shm_name = address[6:]  # len("shm://") = 6

    # Default name if empty
    return shm_name if shm_name else "policy"


def is_server_address(model_path: str) -> bool:
    """
    Return True if model_path looks like a remote/IPC server address.

    Supported formats:
    - TCP:  host:port  (e.g. 192.168.1.10:5000, localhost:5000)
    - HTTP: http(s)://host[:port][/path]
    - SHM:  shm://obs_name/action_name
    """
    if is_http_address(model_path):
        return True
    if is_shm_address(model_path):
        return True
    # TCP format: host:port (not an existing file path)
    if ":" in model_path and not os.path.exists(model_path):
        match = re.match(r"^(.+):(\d+)$", model_path)
        if match:
            return True
    return False


def parse_server_address(model_path: str) -> Tuple[str, int]:
    """
    Parse a TCP server address from model_path.

    Expected format: host:port

    Returns:
        (host, port)

    Raises:
        ValueError: If format is invalid or address is HTTP (use URL directly).
    """
    if is_http_address(model_path):
        raise ValueError(
            f"HTTP address detected: {model_path}. Use the URL directly with FastAPIPolicyClient."
        )
    if ":" in model_path and not os.path.exists(model_path):
        match = re.match(r"^(.+):(\d+)$", model_path)
        if match:
            host = match.group(1)
            port = int(match.group(2))
            return host, port
    raise ValueError(f"Invalid TCP server address format: {model_path}. Expected 'host:port'.")


# ---------------------------------------------------------------------------
# Factory functions (unified interface)
# ---------------------------------------------------------------------------

def create_client(
    address: str,
    ctrl_space: str = "ee",
    ctrl_type: str = "delta",
    timeout_s: float = 3600.0,
) -> BaseClient:
    """
    Create a policy client based on address format.

    - http(s)://...    -> FastAPIPolicyClient
    - shm://shm_name   -> SHMPolicyClient (client_id auto-generated)
    - host:port        -> PolicyClient (TCP)
    """
    if is_http_address(address):
        return __getattr__("FastAPIPolicyClient")(
            base_url=address,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            timeout_s=timeout_s,
        )
    elif is_shm_address(address):
        shm_name = parse_shm_address(address)
        return __getattr__("SHMPolicyClient")(
            shm_name=shm_name,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            timeout_s=timeout_s,
        )
    else:
        host, port = parse_server_address(address)
        return __getattr__("PolicyClient")(
            host=host,
            port=port,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
        )


def create_server(
    policy,
    address: str,
    port: Optional[int] = None,
    batch_wait_ms: float = 1.0,
) -> BaseServer:
    """
    Create a policy server based on address format.

    Transport selection:
    - http(s)://host[:port]  -> FastAPIPolicyServer
    - shm://shm_name         -> SHMPolicyServer
    - host (plain string)    -> PolicyServer (TCP)

    For HTTPS (https://), SSL certificates are read from environment variables:
        ILSTD_SSL_KEYFILE: Path to SSL private key file
        ILSTD_SSL_CERTFILE: Path to SSL certificate file

    For explicit control, instantiate the class directly.

    Args:
        policy: The MetaPolicy (or compatible) to serve.
        address: Bind address. Examples:
            - "0.0.0.0" (TCP on all interfaces)
            - "http://0.0.0.0:8000" (HTTP)
            - "shm://policy" (Shared Memory)
        port: Port number. If None, defaults to 5000 (TCP) or 8000 (HTTP).
              Ignored for SHM mode.
        batch_wait_ms: Time to wait for collecting requests before batching (ms).
                       Used by FastAPI and SHM servers. Default: 1.0ms.
    """
    if is_http_address(address):
        # Extract host and scheme from URL for binding
        from urllib.parse import urlparse

        parsed = urlparse(address)
        host = parsed.hostname or "0.0.0.0"
        p = parsed.port or port or 8000
        is_https = parsed.scheme == "https"

        ssl_keyfile = None
        ssl_certfile = None

        if is_https:
            # Read SSL config from environment variables
            ssl_keyfile = os.environ.get("ILSTD_SSL_KEYFILE")
            ssl_certfile = os.environ.get("ILSTD_SSL_CERTFILE")

            if not ssl_keyfile or not ssl_certfile:
                raise ValueError(
                    "HTTPS requires SSL certificates. Please set the following environment variables:\n"
                    "\n"
                    "    export ILSTD_SSL_KEYFILE=/path/to/your/key.pem\n"
                    "    export ILSTD_SSL_CERTFILE=/path/to/your/cert.pem\n"
                    "\n"
                    "To generate a self-signed certificate for testing:\n"
                    "\n"
                    "    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"\n"
                    "    export ILSTD_SSL_KEYFILE=key.pem\n"
                    "    export ILSTD_SSL_CERTFILE=cert.pem\n"
                )

            # Check that files actually exist
            if not os.path.isfile(ssl_keyfile):
                raise ValueError(
                    f"SSL key file not found: {ssl_keyfile}\n"
                    f"  (from ILSTD_SSL_KEYFILE environment variable)\n"
                    "\n"
                    "Please check the path or generate a new certificate:\n"
                    "\n"
                    "    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"\n"
                    "    export ILSTD_SSL_KEYFILE=$(pwd)/key.pem\n"
                    "    export ILSTD_SSL_CERTFILE=$(pwd)/cert.pem\n"
                )
            if not os.path.isfile(ssl_certfile):
                raise ValueError(
                    f"SSL certificate file not found: {ssl_certfile}\n"
                    f"  (from ILSTD_SSL_CERTFILE environment variable)\n"
                    "\n"
                    "Please check the path or generate a new certificate:\n"
                    "\n"
                    "    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"\n"
                    "    export ILSTD_SSL_KEYFILE=$(pwd)/key.pem\n"
                    "    export ILSTD_SSL_CERTFILE=$(pwd)/cert.pem\n"
                )

        return __getattr__("FastAPIPolicyServer")(
            policy,
            host=host,
            port=p,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            batch_wait_ms=batch_wait_ms,
        )
    elif is_shm_address(address):
        # Shared memory mode
        shm_name = parse_shm_address(address)
        return __getattr__("SHMPolicyServer")(
            policy,
            shm_name=shm_name,
            batch_wait_ms=batch_wait_ms,
        )
    else:
        # Plain host string -> TCP
        host = address
        p = port if port is not None else 5000
        return __getattr__("PolicyServer")(policy, host=host, port=p)


__all__ = [
    # Base
    "BaseServer",
    "BaseClient",
    # TCP implementations
    "PolicyServer",
    "PolicyClient",
    # FastAPI implementations
    "FastAPIPolicyServer",
    "FastAPIPolicyClient",
    # SHM implementations
    "SHMPolicyServer",
    "SHMPolicyClient",
    # Utilities
    "is_server_address",
    "is_http_address",
    "is_shm_address",
    "parse_server_address",
    "parse_shm_address",
    # Factories
    "create_server",
    "create_client",
]


_TRANSPORTS = {'BaseServer': '.base', 'BaseClient': '.base', 'PolicyServer': '.server', 'PolicyClient': '.client', 'FastAPIPolicyServer': '.fastapi.server', 'FastAPIPolicyClient': '.fastapi.client', 'SHMPolicyServer': '.shm.server', 'SHMPolicyClient': '.shm.client'}


def __getattr__(name):
    """Load only the selected transport and its optional dependencies."""
    if name not in _TRANSPORTS:
        raise AttributeError(name)
    import importlib
    cls = getattr(importlib.import_module(_TRANSPORTS[name], __name__), name)
    globals()[name] = cls
    return cls
