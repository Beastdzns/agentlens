"""
AgentLens: Adaptive Causal Observability for AI Agents

This package provides a Python SDK for capturing, analyzing, and visualizing
structured execution events from autonomous AI agents.

Core modules:
- events:   Event data models and schemas (Shridhar)
- collector: Event collection and normalization (Niraj — sdk_collector/)
- storage:  Storage abstraction and in-memory backends (Vyankatesh)
"""

__version__ = "0.1.0"
__author__ = "AgentLens Contributors"
__all__ = [
    "events",
    "collector",
    "storage",
]
