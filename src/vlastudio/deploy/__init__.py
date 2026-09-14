"""
Deployment Module

This module provides deployment utilities for IL-Studio policies.

Note: The subpackage `deploy.remote` has been renamed to `deploy.comm`.
      Imports here are kept for backward compatibility.
"""

__all__ = [
    # TCP
    "PolicyServer",
    "PolicyClient",
    # FastAPI
    "FastAPIPolicyServer",
    "FastAPIPolicyClient",
    # Utilities
    "parse_server_address",
    "is_server_address",
    "is_http_address",
    # Factories
    "create_server",
    "create_client",
    # Base classes
    "BaseServer",
    "BaseClient",
]



def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import comm
    value = getattr(comm, name)
    globals()[name] = value
    return value
