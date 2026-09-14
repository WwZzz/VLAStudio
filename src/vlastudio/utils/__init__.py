"""
Utils package initialization.
Automatically configures logging when utils is imported.
"""

# Configure logging on package import
from .logger import logger

__all__ = ['logger']

