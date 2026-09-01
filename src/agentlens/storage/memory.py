"""
In-memory storage implementations for AgentLens.

Both :class:`InMemoryEventStore` and :class:`InMemoryTraceStore` are
**Phase 1** implementations: data is held in plain Python dicts and protected
by ``asyncio.Lock`` for coroutine-safe concurrent access.

They satisfy all abstract methods defined in :mod:`agentlens.storage.base` and
are fully compatible with:

- ``EventCollector(storage_backend=InMemoryEventStore())``  (Niraj's SDK)
- All integration tests in ``tests/integration/``

Usage::

    import asyncio
    from agentlens.storage import InMemoryEventStore, InMemoryTraceStore

    event_store  = InMemoryEventStore()
    trace_store  = InMemoryTraceStore()

    # With async EventCollector (Niraj's SDK):
    from agentlens.sdk_collector import EventCollector
    collector = EventCollector(storage_backend=event_store)

    async def main():
        async with collector.traced(agent_id="demo-agent"):
            await collector.record_event(event_type="llm_call", ...)

    asyncio.run(main())

Thread-safety notes
-------------------
These implementations use ``asyncio.Lock`` which protects against *concurrent
coroutines* within the same event loop.  They are **not** safe for use across
multiple OS threads; use thread-safe data structures or a real DB backend for
multi-threaded scenarios.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from agentlens.events import AgentEvent, EventStatus, Trace
from agentlens.storage.base import EventStore, TraceStore


class InMemoryEventStore(EventStore):
    """
    Async-safe, in-memory implementation of :class:`~agentlens.storage.base.EventStore`.

    Internally uses:

    - ``_events: Dict[UUID, AgentEvent]`` — primary store keyed by ``event_id``
    - ``_trace_index: Dict[UUID, list[UUID]]`` — secondary index: trace_id → [event_ids]
    - ``asyncio.Lock`` — protects all mutations

    All methods are coroutines so they integrate cleanly with async collectors
    and test suites using ``pytest-asyncio``.

    Example::

        store = InMemoryEventStore()

        event = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )

        await store.save_event(event)
        retrieved = await store.get_event(event.event_id)
        assert retrieved == event

        trace_events = await store.get_trace_events(event.trace_id)
        assert len(trace_events) == 1
    """

    def __init__(self) -> None:
        """Initialise empty storage with an asyncio lock."""
        # Primary store: event_id → AgentEvent
        self._events: dict[UUID, AgentEvent] = {}
        # Secondary index: trace_id → [event_id, ...] (insertion order)
        self._trace_index: dict[UUID, list[UUID]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save_event(self, event: AgentEvent) -> None:
        """
        Persist a single :class:`~agentlens.events.AgentEvent`.

        Upsert semantics: if an event with the same ``event_id`` already
        exists, it is overwritten (last-writer-wins).

        Args:
            event: The event to persist.

        Example::

            await store.save_event(my_event)
        """
        async with self._lock:
            self._events[event.event_id] = event
            # Update trace index
            if event.trace_id not in self._trace_index:
                self._trace_index[event.trace_id] = []
            if event.event_id not in self._trace_index[event.trace_id]:
                self._trace_index[event.trace_id].append(event.event_id)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_event(self, event_id: UUID) -> AgentEvent:
        """
        Retrieve a single event by its ``event_id``.

        Args:
            event_id: UUID of the event to retrieve.

        Returns:
            The stored :class:`~agentlens.events.AgentEvent`.

        Raises:
            KeyError: If ``event_id`` is not found in the store.

        Example::

            event = await store.get_event(some_uuid)
        """
        async with self._lock:
            if event_id not in self._events:
                raise KeyError(f"Event {event_id!r} not found in store.")
            return self._events[event_id]

    async def get_trace_events(self, trace_id: UUID) -> list[AgentEvent]:
        """
        Return all events for a trace, sorted ascending by timestamp.

        Args:
            trace_id: UUID of the trace.

        Returns:
            List of events for this trace ordered by ``timestamp``.
            Empty list if the trace has no recorded events.

        Example::

            events = await store.get_trace_events(trace_id)
            assert events[0].timestamp <= events[-1].timestamp
        """
        async with self._lock:
            event_ids = self._trace_index.get(trace_id, [])
            events = [self._events[eid] for eid in event_ids if eid in self._events]

        # Sort outside the lock to keep critical section short
        return sorted(events, key=lambda e: e.timestamp)

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_trace(self, trace_id: UUID) -> None:
        """
        Remove all events belonging to ``trace_id`` from the store.

        This is a **bulk delete** — all events in the trace's index are
        removed atomically under the lock.  If the trace has no events, this
        is a no-op.

        Args:
            trace_id: UUID of the trace whose events should be deleted.

        Example::

            await store.delete_trace(trace_id)
            events = await store.get_trace_events(trace_id)
            assert events == []
        """
        async with self._lock:
            event_ids = self._trace_index.pop(trace_id, [])
            for eid in event_ids:
                self._events.pop(eid, None)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    async def list_traces(self, limit: int = 100) -> list[UUID]:
        """
        Return the UUIDs of traces that have at least one stored event.

        Returns traces in the order they first appeared (FIFO).

        Args:
            limit: Maximum number of trace UUIDs to return.

        Returns:
            List of trace UUIDs (up to ``limit``).

        Example::

            trace_ids = await store.list_traces(limit=10)
        """
        async with self._lock:
            keys = list(self._trace_index.keys())
        return keys[:limit]

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    async def count_events(self) -> int:
        """Return total number of events currently in the store."""
        async with self._lock:
            return len(self._events)

    async def count_traces(self) -> int:
        """Return total number of distinct traces in the store."""
        async with self._lock:
            return len(self._trace_index)

    def __repr__(self) -> str:
        return (
            f"InMemoryEventStore("
            f"events={len(self._events)}, "
            f"traces={len(self._trace_index)})"
        )


class InMemoryTraceStore(TraceStore):
    """
    Async-safe, in-memory implementation of :class:`~agentlens.storage.base.TraceStore`.

    Stores :class:`~agentlens.events.Trace` metadata objects (without
    embedded events — events are managed separately by :class:`InMemoryEventStore`).

    Internally uses:

    - ``_traces: Dict[UUID, Trace]`` — keyed by ``trace_id``
    - ``asyncio.Lock`` — protects all mutations

    Example::

        store = InMemoryTraceStore()

        trace = Trace(agent_id="research-bot")
        await store.save_trace(trace)

        loaded = await store.get_trace(trace.trace_id)
        assert loaded.agent_id == "research-bot"

        await store.update_trace(trace.trace_id, {"status": EventStatus.SUCCESS})
        updated = await store.get_trace(trace.trace_id)
        assert updated.status == EventStatus.SUCCESS
    """

    def __init__(self) -> None:
        """Initialise empty storage with an asyncio lock."""
        self._traces: dict[UUID, Trace] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save_trace(self, trace: Trace) -> None:
        """
        Persist a :class:`~agentlens.events.Trace` object.

        Upsert semantics: if a trace with the same ``trace_id`` already exists,
        it is replaced entirely.

        Args:
            trace: The trace to store.

        Example::

            trace = create_trace(agent_id="my-agent")
            await store.save_trace(trace)
        """
        async with self._lock:
            self._traces[trace.trace_id] = trace

    async def update_trace(self, trace_id: UUID, updates: dict[str, Any]) -> None:
        """
        Apply a partial update to an existing trace.

        The ``updates`` dict is applied as attribute overrides on the stored
        :class:`~agentlens.events.Trace`.  Only recognised Trace field names
        are applied; unrecognised keys are silently ignored to avoid crashing
        on benign schema mismatches.

        Args:
            trace_id: UUID of the trace to update.
            updates:  Mapping of field names → new values.
                      E.g. ``{"status": EventStatus.SUCCESS, "end_time": datetime.now(utc)}``.

        Raises:
            KeyError: If ``trace_id`` is not in the store.

        Example::

            await store.update_trace(
                trace_id,
                {"status": EventStatus.SUCCESS, "end_time": datetime.now(timezone.utc)},
            )
        """
        async with self._lock:
            if trace_id not in self._traces:
                raise KeyError(f"Trace {trace_id!r} not found in store.")

            existing = self._traces[trace_id]
            # Build a new dict from the existing trace, apply updates
            data = existing.model_dump()
            for key, value in updates.items():
                if key in data:
                    data[key] = value
            # Re-construct Trace to maintain Pydantic validation
            self._traces[trace_id] = Trace(**data)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_trace(self, trace_id: UUID) -> Trace:
        """
        Retrieve a trace by its UUID.

        Args:
            trace_id: UUID of the trace to retrieve.

        Returns:
            The stored :class:`~agentlens.events.Trace`.

        Raises:
            KeyError: If ``trace_id`` is not found.

        Example::

            trace = await store.get_trace(my_trace_id)
        """
        async with self._lock:
            if trace_id not in self._traces:
                raise KeyError(f"Trace {trace_id!r} not found in store.")
            return self._traces[trace_id]

    async def list_traces(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
    ) -> list[Trace]:
        """
        Return stored traces, optionally filtered by ``agent_id`` or ``status``.

        Results are sorted by ``start_time`` descending (most recent first).

        Args:
            limit:    Maximum number of traces to return.
            agent_id: If provided, only traces for this agent are returned.
            status:   If provided, only traces with this status are returned.

        Returns:
            List of :class:`~agentlens.events.Trace` objects.

        Example::

            active = await store.list_traces(limit=50, status=EventStatus.PENDING)
        """
        async with self._lock:
            traces = list(self._traces.values())

        # Apply optional filters
        if agent_id is not None:
            traces = [t for t in traces if t.agent_id == agent_id]
        if status is not None:
            traces = [t for t in traces if t.status == status]

        # Sort most-recent first
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return total number of traces currently in the store."""
        async with self._lock:
            return len(self._traces)

    def __repr__(self) -> str:
        return f"InMemoryTraceStore(traces={len(self._traces)})"
