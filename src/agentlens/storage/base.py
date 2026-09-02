"""
Abstract base classes for AgentLens storage backends.

Design goals
------------
- **Protocol-style**: Concrete backends only need to implement the abstract
  methods; no shared state is inherited.
- **Async-first**: Every public method is a coroutine so implementations can
  use non-blocking I/O (DB drivers, HTTP clients) without interface changes.
- **Future-proof**: Designed for Phase 1 (in-memory) but compatible with
  Phase 1+ backends (PostgreSQL, Redis, Neo4j) — just subclass and implement.
- **Fail-open friendly**: Implementations are encouraged to swallow storage
  errors and emit warnings rather than crashing the agent under observation.

Interfaces
----------
- :class:`EventStore` — Persist and query individual ``AgentEvent`` records.
- :class:`TraceStore` — Persist and query ``Trace`` records (trace metadata).

Integration
-----------
``EventStore`` exposes a ``save()`` alias for ``save_event()`` so it is
directly compatible with Niraj's ``sdk_collector.EventCollector`` which calls::

    await self._storage.save(event)

Usage example::

    class MyPostgresEventStore(EventStore):
        async def save_event(self, event: AgentEvent) -> None:
            await db.execute("INSERT INTO events ...", event.to_dict())

        async def get_event(self, event_id: UUID) -> AgentEvent:
            row = await db.fetchone("SELECT * FROM events WHERE event_id=$1", event_id)
            return AgentEvent.from_dict(row)

        # ... implement remaining abstract methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from agentlens.events import AgentEvent, EventStatus, Trace


class EventStore(ABC):
    """
    Abstract interface for persisting and querying :class:`~agentlens.events.AgentEvent` records.

    Implementations must be **async-safe** — concurrent coroutines may call
    any method simultaneously.

    .. note::
        This class also exposes a :meth:`save` alias for :meth:`save_event`
        to stay compatible with ``EventCollector``'s storage backend protocol::

            await storage_backend.save(event)  # called by EventCollector

    Example — implementing for PostgreSQL (skeleton)::

        class PostgresEventStore(EventStore):
            async def save_event(self, event: AgentEvent) -> None:
                await self._pool.execute(INSERT_SQL, *event.to_dict().values())

            async def get_event(self, event_id: UUID) -> AgentEvent:
                row = await self._pool.fetchrow(SELECT_SQL, event_id)
                if row is None:
                    raise KeyError(f"Event {event_id} not found")
                return AgentEvent.from_dict(dict(row))

            # ... remaining abstract methods
    """

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def save_event(self, event: AgentEvent) -> None:
        """
        Persist a single agent event.

        Args:
            event: The :class:`~agentlens.events.AgentEvent` to store.
                   Implementations should be idempotent on ``event_id``
                   (upsert semantics) when possible.

        Raises:
            Exception: Implementations may raise on unrecoverable storage
                       failures; callers should handle or wrap in fail-open
                       logic.
        """
        ...

    async def save(self, event: AgentEvent) -> None:
        """
        Alias for :meth:`save_event` — provided for compatibility with
        :class:`~agentlens.sdk_collector.EventCollector` which calls
        ``await self._storage.save(event)``.

        Args:
            event: The :class:`~agentlens.events.AgentEvent` to persist.
        """
        await self.save_event(event)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_event(self, event_id: UUID) -> AgentEvent:
        """
        Retrieve a single event by its unique ID.

        Args:
            event_id: UUID of the event to retrieve.

        Returns:
            The matching :class:`~agentlens.events.AgentEvent`.

        Raises:
            KeyError: If no event with ``event_id`` exists in the store.
        """
        ...

    @abstractmethod
    async def get_trace_events(self, trace_id: UUID) -> list[AgentEvent]:
        """
        Retrieve all events belonging to a trace, ordered by timestamp (ascending).

        Args:
            trace_id: UUID of the trace whose events to fetch.

        Returns:
            List of :class:`~agentlens.events.AgentEvent` objects ordered by
            ``timestamp`` ascending.  Returns an empty list if the trace has
            no events.
        """
        ...

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def delete_trace(self, trace_id: UUID) -> None:
        """
        Delete **all events** associated with a trace.

        This is a bulk delete keyed by ``trace_id``, not a cascade from
        :class:`TraceStore` — both stores must be cleaned independently.

        Args:
            trace_id: UUID of the trace whose events should be deleted.
        """
        ...

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    @abstractmethod
    async def list_traces(self, limit: int = 100) -> list[UUID]:
        """
        Return the UUIDs of traces that have at least one stored event.

        Args:
            limit: Maximum number of trace UUIDs to return.  Implementations
                   should return the most recently active traces first when
                   possible.

        Returns:
            List of trace UUID objects (up to ``limit`` entries).
        """
        ...


class TraceStore(ABC):
    """
    Abstract interface for persisting and querying :class:`~agentlens.events.Trace` records.

    Stores high-level trace metadata (agent_id, start/end time, status,
    event_count, total_latency_ms).  Individual events are managed by
    :class:`EventStore`.

    Example — implementing for PostgreSQL (skeleton)::

        class PostgresTraceStore(TraceStore):
            async def save_trace(self, trace: Trace) -> None:
                await self._pool.execute(INSERT_TRACE_SQL, *...)

            async def get_trace(self, trace_id: UUID) -> Trace:
                row = await self._pool.fetchrow(SELECT_TRACE_SQL, trace_id)
                if row is None:
                    raise KeyError(f"Trace {trace_id} not found")
                return Trace.from_dict(dict(row))

            async def update_trace(self, trace_id: UUID, updates: dict) -> None:
                ...

            async def list_traces(self, limit: int = 100, **filters) -> list[Trace]:
                ...
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def save_trace(self, trace: Trace) -> None:
        """
        Persist a :class:`~agentlens.events.Trace` object.

        Implementations are encouraged to use upsert semantics so that
        :meth:`update_trace` can also be expressed as a full save.

        Args:
            trace: The :class:`~agentlens.events.Trace` to store.
        """
        ...

    @abstractmethod
    async def update_trace(self, trace_id: UUID, updates: dict[str, Any]) -> None:
        """
        Apply a partial update to an existing trace record.

        Only the keys present in ``updates`` are modified; all other fields
        remain unchanged.

        Args:
            trace_id: UUID of the trace to update.
            updates:  Dictionary of field names to new values.
                      Recognised keys mirror the :class:`~agentlens.events.Trace`
                      field names (e.g. ``{"status": EventStatus.SUCCESS,
                      "end_time": datetime.now(timezone.utc)}``).

        Raises:
            KeyError: If no trace with ``trace_id`` is found.
        """
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_trace(self, trace_id: UUID) -> Trace:
        """
        Retrieve a single trace by ID.

        Args:
            trace_id: UUID of the trace to retrieve.

        Returns:
            The matching :class:`~agentlens.events.Trace`.

        Raises:
            KeyError: If no trace with ``trace_id`` exists.
        """
        ...

    @abstractmethod
    async def list_traces(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
    ) -> list[Trace]:
        """
        Return a list of stored traces, optionally filtered.

        Args:
            limit:    Maximum number of traces to return.
            agent_id: If provided, only return traces for this agent.
            status:   If provided, only return traces with this status.

        Returns:
            List of :class:`~agentlens.events.Trace` objects ordered by
            ``start_time`` descending (most recent first).
        """
        ...
