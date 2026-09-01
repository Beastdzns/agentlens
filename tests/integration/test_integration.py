"""
End-to-end integration tests for AgentLens storage layer.

These tests simulate real-world usage of the full stack:
    AgentEvent models  →  EventCollector (async wrapper)  →  InMemoryEventStore

Two groups of tests:

1. **Full flow**: create a trace, record a hierarchy of 8 events (parent/child),
   end the trace, retrieve and verify every event and relationship.

2. **Concurrency**: simulate 5 concurrent agent traces running in parallel and
   verify complete isolation — no event from trace A appears in trace B.

Notes on compatibility
----------------------
Niraj's async ``EventCollector`` (``sdk_collector/``) lives on
``feature/collector`` and is not yet merged.  To avoid blocking these tests,
we use a minimal ``AsyncCollectorShim`` that mirrors the exact same interface::

    await shim.start_trace(agent_id)
    await shim.record_event(event_type, input_data, output_data, ...)
    await shim.end_trace()

When ``feature/collector`` is merged, you can replace ``AsyncCollectorShim``
with the real ``EventCollector`` and these tests will continue to pass.

Markers used:
    @pytest.mark.integration — can be slow, requires no external services
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    Trace,
    create_event,
    create_trace,
)
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore


# ---------------------------------------------------------------------------
# AsyncCollectorShim
# ---------------------------------------------------------------------------
# A lightweight, self-contained async collector that wraps create_event and
# InMemoryEventStore.  Mirrors the interface of Niraj's EventCollector so
# integration tests can be swapped to the real SDK with minimal changes.
# ---------------------------------------------------------------------------


class AsyncCollectorShim:
    """
    Minimal async collector shim that mirrors ``sdk_collector.EventCollector``.

    Manages trace context (trace_id, agent_id, parent_id stack) and persists
    events via an injected ``storage_backend`` using ``await backend.save(event)``.

    Replace with the real ``EventCollector`` after ``feature/collector`` is merged.

    Example::

        store = InMemoryEventStore()
        collector = AsyncCollectorShim(storage_backend=store)

        trace_id = await collector.start_trace(agent_id="demo-agent")
        event = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "hello"},
            output_data={"response": "world"},
            latency_ms=120.0,
        )
        await collector.end_trace()
    """

    def __init__(self, storage_backend: Any = None) -> None:
        self._storage = storage_backend
        self._trace_id: Optional[UUID] = None
        self._agent_id: Optional[str] = None
        self._parent_stack: list[UUID] = []

    async def start_trace(
        self, agent_id: str, trace_id: Optional[UUID] = None
    ) -> UUID:
        """Begin a new trace and return the active trace_id."""
        self._trace_id = trace_id if trace_id is not None else uuid4()
        self._agent_id = agent_id
        self._parent_stack = []
        return self._trace_id

    async def end_trace(self) -> None:
        """Close the current trace and clear context."""
        self._trace_id = None
        self._agent_id = None
        self._parent_stack = []

    async def record_event(
        self,
        event_type: EventType | str = EventType.LLM_CALL,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        status: EventStatus | str = EventStatus.SUCCESS,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        """Record a single event under the current trace context."""
        if self._trace_id is None:
            raise RuntimeError("No active trace. Call start_trace() first.")

        if isinstance(event_type, str):
            event_type = EventType(event_type)
        if isinstance(status, str):
            status = EventStatus(status)

        parent_id = self._parent_stack[-1] if self._parent_stack else None

        event = create_event(
            trace_id=self._trace_id,
            agent_id=self._agent_id or "unknown",
            event_type=event_type,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            parent_id=parent_id,
            metadata=metadata,
        )

        if self._storage is not None:
            try:
                await self._storage.save(event)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AgentLens shim: storage failed ({event.event_id}): {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return event

    def push_parent(self, event_id: UUID) -> None:
        """Set event_id as the current parent for nested events."""
        self._parent_stack.append(event_id)

    def pop_parent(self) -> Optional[UUID]:
        """Return to previous parent context."""
        return self._parent_stack.pop() if self._parent_stack else None

    def get_current_trace_id(self) -> Optional[UUID]:
        return self._trace_id

    def get_current_parent_id(self) -> Optional[UUID]:
        return self._parent_stack[-1] if self._parent_stack else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_store() -> InMemoryEventStore:
    return InMemoryEventStore()


@pytest.fixture
def trace_store() -> InMemoryTraceStore:
    return InMemoryTraceStore()


@pytest.fixture
def collector(event_store):
    """AsyncCollectorShim wired to a fresh InMemoryEventStore."""
    return AsyncCollectorShim(storage_backend=event_store)


# ============================================================================
# 1. Full end-to-end flow
# ============================================================================


@pytest.mark.integration
class TestFullFlowIntegration:
    """
    Simulate a real agent execution with a hierarchy of events, then verify
    every aspect of the stored data.
    """

    async def test_full_trace_lifecycle(self, collector, event_store):
        """
        Record a realistic agent trace with 8 events in a parent-child hierarchy.

        Hierarchy::

            [root] DECISION
              ├── [A] LLM_CALL
              ├── [B] TOOL_CALL  (parent = root)
              │     ├── [C] RETRIEVAL  (parent = B)
              │     └── [D] RETRY      (parent = B)
              ├── [E] MEMORY_WRITE
              ├── [F] LLM_CALL
              └── [G] FINAL_RESPONSE
        """
        trace_id = await collector.start_trace(agent_id="research-bot")

        # ---- root event ------------------------------------------------
        root = await collector.record_event(
            event_type=EventType.DECISION,
            input_data={"question": "Research AI agents for 2026"},
            output_data={"plan": "search → retrieve → synthesize"},
            latency_ms=30.0,
            metadata={"step": "planning"},
        )
        assert root.parent_id is None
        assert root.trace_id == trace_id

        # ---- level-1 children -----------------------------------------
        event_a = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "Generate search queries"},
            output_data={"queries": ["ai agent 2026", "autonomous systems"]},
            latency_ms=180.0,
        )
        assert event_a.parent_id is None  # no parent pushed yet

        # Push root as parent for B, C, D
        collector.push_parent(root.event_id)

        event_b = await collector.record_event(
            event_type=EventType.TOOL_CALL,
            input_data={"tool": "web_search", "query": "ai agent 2026"},
            output_data={"hits": 10},
            latency_ms=220.0,
            metadata={"tool": "web_search"},
        )
        assert event_b.parent_id == root.event_id

        # ---- level-2 children under B ----------------------------------
        collector.push_parent(event_b.event_id)

        event_c = await collector.record_event(
            event_type=EventType.RETRIEVAL,
            input_data={"query": "ai agent 2026"},
            output_data={"docs": ["doc1", "doc2"]},
            latency_ms=90.0,
        )
        assert event_c.parent_id == event_b.event_id

        event_d = await collector.record_event(
            event_type=EventType.RETRY,
            input_data={"reason": "rate limit"},
            output_data={"status": "retried"},
            latency_ms=15.0,
            status=EventStatus.SUCCESS,
        )
        assert event_d.parent_id == event_b.event_id

        # Pop back to root level
        collector.pop_parent()

        event_e = await collector.record_event(
            event_type=EventType.MEMORY_WRITE,
            input_data={"key": "search_results"},
            output_data={"written": True},
            latency_ms=5.0,
        )
        assert event_e.parent_id == root.event_id

        # Pop root parent
        collector.pop_parent()

        event_f = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "Synthesize findings"},
            output_data={"summary": "AI agents are advancing rapidly"},
            latency_ms=340.0,
        )

        event_g = await collector.record_event(
            event_type=EventType.FINAL_RESPONSE,
            output_data={"answer": "AI agents are advancing rapidly in 2026"},
            latency_ms=2.0,
        )

        await collector.end_trace()

        # ----------------------------------------------------------------
        # Verify storage
        # ----------------------------------------------------------------
        stored = await event_store.get_trace_events(trace_id)
        assert len(stored) == 8, f"Expected 8 events, got {len(stored)}"

        stored_ids = {e.event_id for e in stored}
        for event in [root, event_a, event_b, event_c, event_d, event_e, event_f, event_g]:
            assert event.event_id in stored_ids, f"Event {event.event_id} missing from store"

        # Verify all events share the same trace_id
        assert all(e.trace_id == trace_id for e in stored)

        # Verify parent-child relationships
        by_id = {e.event_id: e for e in stored}
        assert by_id[event_b.event_id].parent_id == root.event_id
        assert by_id[event_c.event_id].parent_id == event_b.event_id
        assert by_id[event_d.event_id].parent_id == event_b.event_id
        assert by_id[event_e.event_id].parent_id == root.event_id
        assert by_id[event_a.event_id].parent_id is None
        assert by_id[event_f.event_id].parent_id is None
        assert by_id[event_g.event_id].parent_id is None

    async def test_events_ordered_by_timestamp(self, collector, event_store):
        """get_trace_events always returns events in chronological order."""
        trace_id = await collector.start_trace(agent_id="time-test-agent")

        events = []
        for et in [
            EventType.LLM_CALL,
            EventType.TOOL_CALL,
            EventType.RETRIEVAL,
            EventType.MEMORY_WRITE,
            EventType.FINAL_RESPONSE,
        ]:
            e = await collector.record_event(event_type=et)
            events.append(e)

        await collector.end_trace()

        stored = await event_store.get_trace_events(trace_id)
        assert len(stored) == 5

        timestamps = [e.timestamp for e in stored]
        assert timestamps == sorted(timestamps)

    async def test_retrieve_individual_events_by_id(self, collector, event_store):
        """Each recorded event can be retrieved by its event_id."""
        trace_id = await collector.start_trace(agent_id="retrieval-test")

        events = []
        for _ in range(5):
            e = await collector.record_event(
                event_type=EventType.LLM_CALL,
                input_data={"prompt": "test"},
                latency_ms=50.0,
            )
            events.append(e)

        await collector.end_trace()

        for event in events:
            retrieved = await event_store.get_event(event.event_id)
            assert retrieved == event

    async def test_failure_recovery_scenario(self, collector, event_store):
        """
        Simulate a realistic failure-and-recovery trace:
        tool call fails → retry → succeeds → final response.
        """
        trace_id = await collector.start_trace(agent_id="recovery-agent")

        failing_event = await collector.record_event(
            event_type=EventType.TOOL_CALL,
            input_data={"tool": "search", "query": "climate 2026"},
            output_data={"error": "timeout"},
            status=EventStatus.FAILURE,
            latency_ms=5000.0,
            metadata={"tool": "search"},
        )

        collector.push_parent(failing_event.event_id)
        retry_event = await collector.record_event(
            event_type=EventType.RETRY,
            input_data={"tool": "search", "query": "climate 2026 retry"},
            output_data={"hits": 5},
            status=EventStatus.SUCCESS,
            latency_ms=800.0,
        )
        assert retry_event.parent_id == failing_event.event_id
        collector.pop_parent()

        llm_event = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"docs": ["doc1", "doc2"]},
            output_data={"summary": "Climate change summary"},
            latency_ms=200.0,
        )

        final = await collector.record_event(
            event_type=EventType.FINAL_RESPONSE,
            output_data={"answer": "Climate summary 2026"},
            latency_ms=1.0,
        )

        await collector.end_trace()

        stored = await event_store.get_trace_events(trace_id)
        assert len(stored) == 4

        failures = [e for e in stored if e.status == EventStatus.FAILURE]
        assert len(failures) == 1
        assert failures[0].event_type == EventType.TOOL_CALL

        retries = [e for e in stored if e.event_type == EventType.RETRY]
        assert len(retries) == 1
        assert retries[0].parent_id == failures[0].event_id

    async def test_trace_store_integration_with_event_store(
        self, event_store, trace_store
    ):
        """TraceStore and EventStore work independently and consistently."""
        collector = AsyncCollectorShim(storage_backend=event_store)
        trace_id = await collector.start_trace(agent_id="dual-store-agent")

        # Save trace metadata
        trace = create_trace(agent_id="dual-store-agent")
        # Give the trace the same ID as what the collector is using
        trace_obj = Trace(
            trace_id=trace_id,
            agent_id="dual-store-agent",
        )
        await trace_store.save_trace(trace_obj)

        # Record 3 events
        for et in [EventType.LLM_CALL, EventType.TOOL_CALL, EventType.FINAL_RESPONSE]:
            await collector.record_event(event_type=et)

        await collector.end_trace()

        # Update trace metadata
        from datetime import timezone
        await trace_store.update_trace(
            trace_id,
            {"status": EventStatus.SUCCESS, "end_time": __import__("datetime").datetime.now(timezone.utc)},
        )

        # Verify both stores are consistent
        stored_trace = await trace_store.get_trace(trace_id)
        stored_events = await event_store.get_trace_events(trace_id)

        assert stored_trace.status == EventStatus.SUCCESS
        assert len(stored_events) == 3
        assert stored_trace.trace_id == trace_id


# ============================================================================
# 2. Concurrency integration tests
# ============================================================================


@pytest.mark.integration
class TestConcurrencyIntegration:
    """
    Simulate multiple concurrent agent traces to prove zero event bleed.

    Each "agent" runs as an independent coroutine.  All agents share the same
    InMemoryEventStore but use separate collectors so their trace contexts
    are completely isolated.
    """

    async def _run_agent(
        self,
        agent_id: str,
        n_events: int,
        store: InMemoryEventStore,
    ) -> tuple[UUID, list[UUID]]:
        """
        Run a single agent trace and return (trace_id, [event_ids]).

        This is the per-agent coroutine launched concurrently in tests below.
        """
        collector = AsyncCollectorShim(storage_backend=store)
        trace_id = await collector.start_trace(agent_id=agent_id)

        event_ids = []
        for i in range(n_events):
            event = await collector.record_event(
                event_type=EventType.LLM_CALL,
                input_data={"step": i, "agent": agent_id},
                output_data={"result": f"step_{i}_done"},
                latency_ms=float(10 * (i + 1)),
            )
            event_ids.append(event.event_id)

        await collector.end_trace()
        return trace_id, event_ids

    async def test_3_concurrent_traces_no_bleed(self, event_store):
        """
        3 agents run concurrently; each trace must contain exactly its own events.
        """
        agents = [("agent-alpha", 5), ("agent-beta", 7), ("agent-gamma", 3)]

        results = await asyncio.gather(
            *(self._run_agent(name, n, event_store) for name, n in agents)
        )

        for (agent_name, expected_count), (trace_id, event_ids) in zip(agents, results):
            stored = await event_store.get_trace_events(trace_id)
            stored_ids = {e.event_id for e in stored}
            expected_ids = set(event_ids)

            # Correct count
            assert len(stored) == expected_count, (
                f"{agent_name}: expected {expected_count} events, got {len(stored)}"
            )
            # No missing events
            assert stored_ids == expected_ids, (
                f"{agent_name}: event IDs do not match"
            )
            # No events from other traces
            assert all(e.trace_id == trace_id for e in stored), (
                f"{agent_name}: found events from a different trace!"
            )
            # Correct agent_id on every event
            assert all(e.agent_id == agent_name for e in stored), (
                f"{agent_name}: found events with wrong agent_id"
            )

    async def test_5_concurrent_traces_no_bleed(self, event_store):
        """
        5 agents, varying event counts — stress test for lock correctness.
        """
        agents = [
            ("agent-1", 10),
            ("agent-2", 15),
            ("agent-3", 8),
            ("agent-4", 12),
            ("agent-5", 6),
        ]

        results = await asyncio.gather(
            *(self._run_agent(name, n, event_store) for name, n in agents)
        )

        for (agent_name, expected_count), (trace_id, event_ids) in zip(agents, results):
            stored = await event_store.get_trace_events(trace_id)

            assert len(stored) == expected_count, (
                f"{agent_name}: expected {expected_count}, got {len(stored)}"
            )
            stored_ids = {e.event_id for e in stored}
            assert stored_ids == set(event_ids)
            assert all(e.agent_id == agent_name for e in stored)

    async def test_concurrent_traces_all_appear_in_list_traces(self, event_store):
        """
        After N concurrent traces, list_traces returns all N trace UUIDs.
        """
        n = 5
        agents = [(f"agent-{i}", 3) for i in range(n)]

        results = await asyncio.gather(
            *(self._run_agent(name, count, event_store) for name, count in agents)
        )

        trace_ids_created = {tid for tid, _ in results}
        listed = set(await event_store.list_traces(limit=100))

        assert trace_ids_created == listed

    async def test_concurrent_saves_all_events_counted(self, event_store):
        """
        100 coroutines each saving 1 event to the same trace = exactly 100 events.
        """
        trace_id = uuid4()
        events = [
            create_event(
                trace_id=trace_id,
                agent_id="stress-agent",
                event_type=EventType.LLM_CALL,
                latency_ms=10.0,
            )
            for _ in range(100)
        ]

        await asyncio.gather(*(event_store.save_event(e) for e in events))

        stored = await event_store.get_trace_events(trace_id)
        assert len(stored) == 100

    async def test_parent_child_hierarchy_survives_concurrency(self, event_store):
        """
        Parent-child relationships recorded under concurrent load are correct.
        """
        async def build_hierarchy(store: InMemoryEventStore) -> tuple[UUID, UUID, list[UUID]]:
            """Return (trace_id, parent_id, child_event_ids)."""
            collector = AsyncCollectorShim(storage_backend=store)
            tid = await collector.start_trace(agent_id="hierarchy-agent")

            parent = await collector.record_event(
                event_type=EventType.DECISION,
                input_data={"plan": "run children"},
                latency_ms=5.0,
            )
            collector.push_parent(parent.event_id)

            children = []
            for _ in range(3):
                child = await collector.record_event(
                    event_type=EventType.LLM_CALL,
                    latency_ms=50.0,
                )
                children.append(child.event_id)

            collector.pop_parent()
            await collector.end_trace()
            return tid, parent.event_id, children

        # Run 4 such hierarchies concurrently
        results = await asyncio.gather(
            *(build_hierarchy(event_store) for _ in range(4))
        )

        for trace_id, parent_id, child_ids in results:
            stored = await event_store.get_trace_events(trace_id)
            assert len(stored) == 4  # 1 parent + 3 children

            by_id = {e.event_id: e for e in stored}
            for cid in child_ids:
                assert cid in by_id, "Child event missing from store"
                assert by_id[cid].parent_id == parent_id, (
                    f"Child {cid} has wrong parent_id"
                )
