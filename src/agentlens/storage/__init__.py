"""
AgentLens Storage Layer
=======================

This package provides the storage abstraction and in-memory implementations
for persisting ``AgentEvent`` and ``Trace`` objects.

The interfaces are **protocol-style** abstract base classes designed to be
replaced with real backends (PostgreSQL, Redis, etc.) in later phases without
touching the SDK or collector code.

Public API::

    from agentlens.storage import (
        EventStore,
        TraceStore,
        InMemoryEventStore,
        InMemoryTraceStore,
    )

Quick-start::

    from agentlens.storage import InMemoryEventStore, InMemoryTraceStore

    event_store = InMemoryEventStore()
    trace_store = InMemoryTraceStore()

    # Use with EventCollector
    from agentlens.sdk_collector import EventCollector
    collector = EventCollector(storage_backend=event_store)
"""

from agentlens.storage.base import EventStore, TraceStore
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore

__all__ = [
    "EventStore",
    "TraceStore",
    "InMemoryEventStore",
    "InMemoryTraceStore",
]
