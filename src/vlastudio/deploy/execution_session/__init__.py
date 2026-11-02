"""Persistent robot/simulation execution with native scheduling and SHM."""
from .core import ExecutionSession, IntentSupersededError

__all__ = ['ExecutionSession', 'IntentSupersededError']
