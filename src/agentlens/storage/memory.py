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

    async def list_trace_page(
        self,
        *,
        limit: int = 20,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
    ) -> tuple[list[Trace], Optional[str]]:
        """
        Return a page of traces using in-memory keyset-style cursor pagination.

        The cursor encodes ``(start_time ISO string, trace_id string)`` in the
        same format as :class:`~agentlens.storage.sqlite.SqliteTraceStore` so
        that tests can exercise both backends with identical logic.

        Args:
            limit:      Page size (default 20).
            agent_id:   Optional agent filter.
            status:     Optional status filter.
            start_date: Include only traces whose ``start_time >= start_date``.
            end_date:   Include only traces whose ``start_time <= end_date``.
            cursor:     Opaque continuation token from a previous call.

        Returns:
            ``(traces, next_cursor)`` where ``next_cursor`` is ``None`` when
            there are no more results.
        """
        import base64  # local import to avoid circular at module level

        async with self._lock:
            all_traces = list(self._traces.values())

        # --- filter ---
        if agent_id is not None:
            all_traces = [t for t in all_traces if t.agent_id == agent_id]
        if status is not None:
            all_traces = [t for t in all_traces if t.status == status]
        if start_date is not None:
            # Make start_date timezone-aware for comparison
            sd = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
            all_traces = [t for t in all_traces if t.start_time >= sd]
        if end_date is not None:
            ed = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
            all_traces = [t for t in all_traces if t.start_time <= ed]

        # --- keyset cursor ---
        if cursor is not None:
            raw = base64.urlsafe_b64decode(cursor.encode()).decode()
            cursor_st_str, cursor_tid = raw.rsplit("|", 1)
            cursor_st = datetime.fromisoformat(cursor_st_str)
            if cursor_st.tzinfo is None:
                cursor_st = cursor_st.replace(tzinfo=timezone.utc)
            # keep only rows that come *after* the cursor in DESC order
            # i.e. (start_time, trace_id) < (cursor_st, cursor_tid)
            all_traces = [
                t for t in all_traces
                if (t.start_time, str(t.trace_id)) < (cursor_st, cursor_tid)
            ]

        # --- sort descending ---
        all_traces.sort(key=lambda t: (t.start_time, str(t.trace_id)), reverse=True)

        # --- paginate ---
        has_next = len(all_traces) > limit
        page = all_traces[:limit]

        next_cursor: Optional[str] = None
        if has_next and page:
            last = page[-1]
            st_str = last.start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            raw_cursor = f"{st_str}|{last.trace_id}"
            next_cursor = base64.urlsafe_b64encode(raw_cursor.encode()).decode()

        return page, next_cursor

    async def count_traces(
        self,
        *,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Return the total number of traces matching the given filters.

        Args:
            agent_id:   Optional agent filter.
            status:     Optional status filter.
            start_date: Minimum ``start_time`` (inclusive).
            end_date:   Maximum ``start_time`` (inclusive).

        Returns:
            Integer count.
        """
        async with self._lock:
            traces = list(self._traces.values())

        if agent_id is not None:
            traces = [t for t in traces if t.agent_id == agent_id]
        if status is not None:
            traces = [t for t in traces if t.status == status]
        if start_date is not None:
            sd = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
            traces = [t for t in traces if t.start_time >= sd]
        if end_date is not None:
            ed = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
            traces = [t for t in traces if t.start_time <= ed]

        return len(traces)

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return total number of traces currently in the store."""
        async with self._lock:
            return len(self._traces)

    def __repr__(self) -> str:
        return f"InMemoryTraceStore(traces={len(self._traces)})"
