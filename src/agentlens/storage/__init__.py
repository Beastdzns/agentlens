"""
AgentLens Storage Layer
=======================

This package provides the storage abstraction, in-memory implementations,
and an async SQLite backend for persisting ``AgentEvent`` and ``Trace`` objects.

The interfaces are **protocol-style** abstract base classes designed to be
replaced with real backends in later phases without touching the SDK or
collector code.

Default backend
---------------
``AGENTLENS_STORAGE`` defaults to ``memory``, so existing tests continue to
use the in-memory stores.  Only :mod:`run_demo` (and code that explicitly
calls :func:`init_db`) opts into the SQLite backend.

Public API::

    from agentlens.storage import (
        EventStore,
        TraceStore,
        InMemoryEventStore,
        InMemoryTraceStore,
        SqliteEventStore,
        SqliteTraceStore,
        init_db,
        get_database,
    )
"""

from agentlens.storage.base import EventStore, TraceStore
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore
from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore
from agentlens.storage.db import init_db, get_database

__all__ = [
    # Abstractions
    "EventStore",
    "TraceStore",
    # In-memory (default)
    "InMemoryEventStore",
    "InMemoryTraceStore",
    # SQLite backend
    "SqliteEventStore",
    "SqliteTraceStore",
    # Database lifecycle
    "init_db",
    "get_database",
]

