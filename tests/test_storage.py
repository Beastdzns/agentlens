"""
Unit tests for AgentLens storage backends.

Tests cover:
- InMemoryEventStore: save, get, trace queries, delete, listing, concurrency
- InMemoryTraceStore: save, get, update, list with filters, error cases
- Base interface: save() alias compatibility with EventCollector

All tests are async (pytest-asyncio, asyncio_mode = "auto" in pyproject.toml).

Markers used:
  @pytest.mark.unit  — fast, isolated unit tests
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

import pytest

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    Trace,
    create_event,
    create_trace,
)
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore


# ============================================================================
# InMemoryEventStore — basic CRUD
# ============================================================================


@pytest.mark.unit
class TestInMemoryEventStoreBasic:
    """Test save, get, and basic error paths."""

    async def test_save_and_get_event(self, memory_event_store, sample_event):
        """Saved event can be retrieved by event_id."""
        event = sample_event()
        await memory_event_store.save_event(event)

        retrieved = await memory_event_store.get_event(event.event_id)
        assert retrieved == event

    async def test_save_alias_works(self, memory_event_store, sample_event):
        """
        store.save(event) is an alias for save_event(event).
        This is what EventCollector calls: await self._storage.save(event).
        """
        event = sample_event()
        await memory_event_store.save(event)  # alias

        retrieved = await memory_event_store.get_event(event.event_id)
        assert retrieved == event

    async def test_get_nonexistent_event_raises_key_error(self, memory_event_store):
        """get_event on an unknown UUID raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await memory_event_store.get_event(uuid4())

    async def test_upsert_semantics(self, memory_event_store, sample_event):
        """
        Saving two events with different event_ids stores two separate records.
        (Events are immutable, so upsert is validated by ensuring both exist.)
        """
        trace_id = uuid4()
        event1 = sample_event(trace_id=trace_id)
        event2 = sample_event(trace_id=trace_id, event_type=EventType.TOOL_CALL)

        await memory_event_store.save_event(event1)
        await memory_event_store.save_event(event2)

        assert await memory_event_store.count_events() == 2

    async def test_save_multiple_events_different_traces(self, memory_event_store, sample_event):
        """Events for different traces coexist without interference."""
        trace_a = uuid4()
        trace_b = uuid4()

        event_a = sample_event(trace_id=trace_a)
        event_b = sample_event(trace_id=trace_b)

        await memory_event_store.save_event(event_a)
        await memory_event_store.save_event(event_b)

        events_a = await memory_event_store.get_trace_events(trace_a)
        events_b = await memory_event_store.get_trace_events(trace_b)

        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].event_id != events_b[0].event_id


# ============================================================================
# InMemoryEventStore — trace queries
# ============================================================================


@pytest.mark.unit
class TestInMemoryEventStoreTraceQueries:
    """Test get_trace_events, ordering, and delete_trace."""

    async def test_get_trace_events_returns_empty_for_unknown_trace(
        self, memory_event_store
    ):
        """Unknown trace returns empty list (not an error)."""
        result = await memory_event_store.get_trace_events(uuid4())
        assert result == []

    async def test_get_trace_events_ordered_by_timestamp(
        self, memory_event_store
    ):
        """Events are returned sorted ascending by timestamp."""
        trace_id = uuid4()

        # Create events with explicit, spread-out timestamps
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        events_to_save = []
        for i in range(5):
            ts = t0 + timedelta(seconds=i * 10)
            e = AgentEvent(
                trace_id=trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                timestamp=ts,
                latency_ms=float(i * 10),
            )
            events_to_save.append(e)

        # Save in reverse order to prove sorting works
        for event in reversed(events_to_save):
            await memory_event_store.save_event(event)

        result = await memory_event_store.get_trace_events(trace_id)

        assert len(result) == 5
        timestamps = [e.timestamp for e in result]
        assert timestamps == sorted(timestamps), "Events must be sorted ascending by timestamp"

    async def test_get_trace_events_returns_only_matching_trace(
        self, memory_event_store, sample_event
    ):
        """Events from other traces do not appear in query results."""
        target_trace = uuid4()
        other_trace = uuid4()

        for _ in range(3):
            await memory_event_store.save_event(sample_event(trace_id=target_trace))
        for _ in range(5):
            await memory_event_store.save_event(sample_event(trace_id=other_trace))

        result = await memory_event_store.get_trace_events(target_trace)
        assert len(result) == 3
        assert all(e.trace_id == target_trace for e in result)

    async def test_delete_trace_removes_all_events(
        self, memory_event_store, sample_event
    ):
        """delete_trace removes all events for that trace."""
        trace_id = uuid4()
        event_ids = []

        for _ in range(4):
            e = sample_event(trace_id=trace_id)
            event_ids.append(e.event_id)
            await memory_event_store.save_event(e)

        await memory_event_store.delete_trace(trace_id)

        # Trace events should be empty
        assert await memory_event_store.get_trace_events(trace_id) == []

        # Individual events should also be gone
        for eid in event_ids:
            with pytest.raises(KeyError):
                await memory_event_store.get_event(eid)

    async def test_delete_trace_is_noop_for_unknown_trace(
        self, memory_event_store
    ):
        """Deleting a trace that doesn't exist should not raise."""
        await memory_event_store.delete_trace(uuid4())  # must not raise

    async def test_delete_one_trace_does_not_affect_others(
        self, memory_event_store, sample_event
    ):
        """Deleting trace A must not touch events from trace B."""
        trace_a = uuid4()
        trace_b = uuid4()

        event_b = sample_event(trace_id=trace_b)
        await memory_event_store.save_event(sample_event(trace_id=trace_a))
        await memory_event_store.save_event(event_b)

        await memory_event_store.delete_trace(trace_a)

        surviving = await memory_event_store.get_trace_events(trace_b)
        assert len(surviving) == 1
        assert surviving[0].event_id == event_b.event_id


# ============================================================================
# InMemoryEventStore — list_traces
# ============================================================================


@pytest.mark.unit
class TestInMemoryEventStoreListTraces:
    """Test list_traces with limit."""

    async def test_list_traces_empty(self, memory_event_store):
        result = await memory_event_store.list_traces()
        assert result == []

    async def test_list_traces_returns_all_trace_ids(
        self, memory_event_store, sample_event
    ):
        trace_ids = {uuid4() for _ in range(5)}
        for tid in trace_ids:
            await memory_event_store.save_event(sample_event(trace_id=tid))

        result = await memory_event_store.list_traces()
        assert set(result) == trace_ids

    async def test_list_traces_respects_limit(
        self, memory_event_store, sample_event
    ):
        for _ in range(10):
            await memory_event_store.save_event(sample_event())

        result = await memory_event_store.list_traces(limit=3)
        assert len(result) == 3

    async def test_list_traces_after_delete(
        self, memory_event_store, sample_event
    ):
        """Deleted trace should not appear in list."""
        trace_a = uuid4()
        trace_b = uuid4()

        await memory_event_store.save_event(sample_event(trace_id=trace_a))
        await memory_event_store.save_event(sample_event(trace_id=trace_b))

        await memory_event_store.delete_trace(trace_a)

        result = await memory_event_store.list_traces()
        assert trace_a not in result
        assert trace_b in result


# ============================================================================
# InMemoryEventStore — concurrency
# ============================================================================


@pytest.mark.unit
class TestInMemoryEventStoreConcurrency:
    """Verify asyncio.Lock prevents data corruption under concurrent saves."""

    async def test_concurrent_saves_to_same_trace(
        self, memory_event_store, sample_event
    ):
        """50 concurrent coroutines saving to the same trace = 50 events stored."""
        trace_id = uuid4()
        events = [sample_event(trace_id=trace_id) for _ in range(50)]

        await asyncio.gather(*(memory_event_store.save_event(e) for e in events))

        result = await memory_event_store.get_trace_events(trace_id)
        assert len(result) == 50

    async def test_concurrent_saves_to_different_traces_no_bleed(
        self, memory_event_store, sample_event
    ):
        """Concurrent writes to N traces must not mix events between traces."""
        n_traces = 5
        events_per_trace = 10
        trace_ids = [uuid4() for _ in range(n_traces)]

        all_saves = []
        expected: dict[UUID, set[UUID]] = {tid: set() for tid in trace_ids}

        for tid in trace_ids:
            for _ in range(events_per_trace):
                e = sample_event(trace_id=tid)
                expected[tid].add(e.event_id)
                all_saves.append(memory_event_store.save_event(e))

        await asyncio.gather(*all_saves)

        for tid in trace_ids:
            result = await memory_event_store.get_trace_events(tid)
            result_ids = {e.event_id for e in result}
            assert result_ids == expected[tid], (
                f"Trace {tid}: expected {len(expected[tid])} events, "
                f"got {len(result_ids)}"
            )


# ============================================================================
# InMemoryTraceStore — basic CRUD
# ============================================================================


@pytest.mark.unit
class TestInMemoryTraceStoreBasic:
    """Test save, get, and update operations."""

    async def test_save_and_get_trace(self, memory_trace_store, sample_trace):
        trace = sample_trace()
        await memory_trace_store.save_trace(trace)

        loaded = await memory_trace_store.get_trace(trace.trace_id)
        assert loaded.trace_id == trace.trace_id
        assert loaded.agent_id == trace.agent_id

    async def test_get_nonexistent_trace_raises_key_error(
        self, memory_trace_store
    ):
        with pytest.raises(KeyError, match="not found"):
            await memory_trace_store.get_trace(uuid4())

    async def test_update_trace_status(self, memory_trace_store, sample_trace):
        trace = sample_trace(status=EventStatus.PENDING)
        await memory_trace_store.save_trace(trace)

        await memory_trace_store.update_trace(
            trace.trace_id, {"status": EventStatus.SUCCESS}
        )

        updated = await memory_trace_store.get_trace(trace.trace_id)
        assert updated.status == EventStatus.SUCCESS

    async def test_update_trace_end_time(self, memory_trace_store, sample_trace):
        trace = sample_trace()
        await memory_trace_store.save_trace(trace)

        end = datetime.now(timezone.utc)
        await memory_trace_store.update_trace(trace.trace_id, {"end_time": end})

        updated = await memory_trace_store.get_trace(trace.trace_id)
        assert updated.end_time is not None

    async def test_update_nonexistent_trace_raises_key_error(
        self, memory_trace_store
    ):
        with pytest.raises(KeyError):
            await memory_trace_store.update_trace(uuid4(), {"status": EventStatus.SUCCESS})

    async def test_update_ignores_unknown_keys(
        self, memory_trace_store, sample_trace
    ):
        """update_trace with unrecognised keys must not crash."""
        trace = sample_trace()
        await memory_trace_store.save_trace(trace)

        # Should not raise
        await memory_trace_store.update_trace(
            trace.trace_id,
            {"status": EventStatus.SUCCESS, "nonexistent_field": "boom"},
        )
        loaded = await memory_trace_store.get_trace(trace.trace_id)
        assert loaded.status == EventStatus.SUCCESS


# ============================================================================
# InMemoryTraceStore — list with filters
# ============================================================================


@pytest.mark.unit
class TestInMemoryTraceStoreList:
    """Test list_traces with agent_id and status filters."""

    async def test_list_traces_empty(self, memory_trace_store):
        result = await memory_trace_store.list_traces()
        assert result == []

    async def test_list_traces_returns_all(self, memory_trace_store, sample_trace):
        traces = [sample_trace() for _ in range(5)]
        for t in traces:
            await memory_trace_store.save_trace(t)

        result = await memory_trace_store.list_traces()
        assert len(result) == 5

    async def test_list_traces_filter_by_agent_id(
        self, memory_trace_store, sample_trace
    ):
        for _ in range(3):
            await memory_trace_store.save_trace(sample_trace(agent_id="bot-A"))
        for _ in range(2):
            await memory_trace_store.save_trace(sample_trace(agent_id="bot-B"))

        result = await memory_trace_store.list_traces(agent_id="bot-A")
        assert len(result) == 3
        assert all(t.agent_id == "bot-A" for t in result)

    async def test_list_traces_filter_by_status(
        self, memory_trace_store, sample_trace
    ):
        for _ in range(4):
            t = sample_trace(status=EventStatus.PENDING)
            await memory_trace_store.save_trace(t)
        for _ in range(2):
            t = sample_trace(status=EventStatus.SUCCESS)
            await memory_trace_store.save_trace(t)

        pending = await memory_trace_store.list_traces(status=EventStatus.PENDING)
        assert len(pending) == 4

        success = await memory_trace_store.list_traces(status=EventStatus.SUCCESS)
        assert len(success) == 2

    async def test_list_traces_respects_limit(
        self, memory_trace_store, sample_trace
    ):
        for _ in range(10):
            await memory_trace_store.save_trace(sample_trace())

        result = await memory_trace_store.list_traces(limit=4)
        assert len(result) == 4

    async def test_list_traces_sorted_most_recent_first(
        self, memory_trace_store
    ):
        """Traces are returned with the most recently started trace first."""
        traces = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(5):
            t = Trace(
                agent_id="agent",
                start_time=base + timedelta(hours=i),
            )
            traces.append(t)
            await memory_trace_store.save_trace(t)

        result = await memory_trace_store.list_traces()
        start_times = [t.start_time for t in result]
        assert start_times == sorted(start_times, reverse=True)
