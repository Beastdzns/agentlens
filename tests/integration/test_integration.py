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
from fastapi.testclient import TestClient

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    Trace,
    create_event,
    create_trace,
)
from agentlens.storage.db import Database, _run_migrations
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore
from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore
from agentlens.web_api import app


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


# ============================================================================
# 3. SQLite persistence integration (Phase 9)
# ============================================================================
#
# The tests above exercise the in-memory backend.  The tests below run the
# same real collector workflow against a file-backed SQLite database and drive
# the merged trace-history API end to end.
# ============================================================================

SQLITE_MARKER = pytest.mark.integration


@pytest.fixture
async def sqlite_db(tmp_path):
    """A migrated SQLite database backed by a temporary file."""
    db = Database(tmp_path / "integration.db")
    await db._open()
    await _run_migrations(db)
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def sqlite_event_store(sqlite_db) -> SqliteEventStore:
    return SqliteEventStore(sqlite_db)


@pytest.fixture
async def sqlite_trace_store(sqlite_db) -> SqliteTraceStore:
    return SqliteTraceStore(sqlite_db)


def _use_sqlite_backend(tmp_path, monkeypatch) -> str:
    """Point the web app at a temp SQLite file and force offline runs."""
    db_file = tmp_path / "api-integration.db"
    monkeypatch.setenv("AGENTLENS_STORAGE", "sqlite")
    monkeypatch.setenv("AGENTLENS_DB_PATH", str(db_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return str(db_file)


@SQLITE_MARKER
class TestSqliteStoreIntegration:
    """The collector workflow produces the same structure on SQLite as in memory."""

    async def _run_workflow(self, store) -> list[tuple]:
        collector = AsyncCollectorShim(storage_backend=store)
        trace_id = await collector.start_trace(agent_id="parity-agent")

        root = await collector.record_event(
            event_type=EventType.DECISION, latency_ms=1.0
        )
        collector.push_parent(root.event_id)
        await collector.record_event(event_type=EventType.LLM_CALL, latency_ms=2.0)
        failing = await collector.record_event(
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            latency_ms=3.0,
        )
        collector.push_parent(failing.event_id)
        await collector.record_event(event_type=EventType.RETRY, latency_ms=4.0)
        collector.pop_parent()
        collector.pop_parent()
        await collector.record_event(event_type=EventType.FINAL_RESPONSE, latency_ms=5.0)
        await collector.end_trace()

        stored = await store.get_trace_events(trace_id)
        assert stored  # sanity
        # Normalise away random UUIDs/timestamps but keep ordering + structure.
        parents = {root.event_id: "root", failing.event_id: "tool"}
        return [
            (
                e.event_type.value,
                e.status.value,
                None if e.parent_id is None else parents.get(e.parent_id, "other"),
                e.latency_ms,
            )
            for e in stored
        ]

    async def test_sqlite_matches_in_memory_workflow(
        self, event_store, sqlite_event_store
    ):
        assert await self._run_workflow(sqlite_event_store) == await self._run_workflow(
            event_store
        )

    async def test_events_survive_reopen(self, tmp_path):
        path = tmp_path / "durable.db"

        db = Database(path)
        await db._open()
        await _run_migrations(db)
        store = SqliteEventStore(db)
        collector = AsyncCollectorShim(storage_backend=store)
        trace_id = await collector.start_trace(agent_id="durable-agent")
        for _ in range(3):
            await collector.record_event(event_type=EventType.LLM_CALL)
        await collector.end_trace()
        before = [e.event_id for e in await store.get_trace_events(trace_id)]
        await db.close()

        reopened = Database(path)
        await reopened._open()
        await _run_migrations(reopened)
        try:
            after = [
                e.event_id
                for e in await SqliteEventStore(reopened).get_trace_events(trace_id)
            ]
        finally:
            await reopened.close()

        assert len(before) == 3
        assert after == before


@SQLITE_MARKER
class TestTraceHistoryApiIntegration:
    """`POST /api/run` → `GET /api/traces` → `GET /api/traces/{id}` on SQLite."""

    def test_run_history_and_detail_round_trip(self, tmp_path, monkeypatch):
        _use_sqlite_backend(tmp_path, monkeypatch)

        with TestClient(app) as client:
            run = client.post("/api/run", json={"query": "integration query"}).json()
            trace_id = run["trace_id"]

            listing = client.get("/api/traces").json()
            assert listing["count"] == 1
            assert [item["trace_id"] for item in listing["items"]] == [trace_id]

            detail = client.get(f"/api/traces/{trace_id}").json()
            assert detail["trace_id"] == trace_id
            assert [e["event_id"] for e in detail["events"]] == [
                e["event_id"] for e in run["events"]
            ]

    def test_history_survives_restart(self, tmp_path, monkeypatch):
        _use_sqlite_backend(tmp_path, monkeypatch)

        with TestClient(app) as client:
            run = client.post("/api/run", json={"query": "persist me"}).json()
            trace_id = run["trace_id"]
            event_count = len(run["events"])
            assert event_count > 0

        # A second lifespan simulates a server restart.
        with TestClient(app) as client:
            assert client.get("/api/traces").json()["count"] == 1
            detail = client.get(f"/api/traces/{trace_id}")
            assert detail.status_code == 200
            assert len(detail.json()["events"]) == event_count

    def test_pagination_across_multiple_runs(self, tmp_path, monkeypatch):
        _use_sqlite_backend(tmp_path, monkeypatch)

        with TestClient(app) as client:
            for i in range(3):
                assert client.post(
                    "/api/run", json={"query": f"query-{i}"}
                ).status_code == 200

            first = client.get("/api/traces", params={"limit": 2}).json()
            assert len(first["items"]) == 2
            assert first["next_cursor"]

            second = client.get(
                "/api/traces",
                params={"limit": 2, "cursor": first["next_cursor"]},
            ).json()
            assert len(second["items"]) == 1
            assert second["next_cursor"] is None

            seen = {i["trace_id"] for i in first["items"]} | {
                i["trace_id"] for i in second["items"]
            }
            assert len(seen) == 3

    def test_filters_and_validation(self, tmp_path, monkeypatch):
        _use_sqlite_backend(tmp_path, monkeypatch)

        with TestClient(app) as client:
            client.post("/api/run", json={"query": "filter me"})

            success = client.get("/api/traces", params={"status": "success"}).json()
            assert success["count"] == 1

            failure = client.get("/api/traces", params={"status": "failure"}).json()
            assert failure["count"] == 0

            assert (
                client.get("/api/traces", params={"status": "bogus"}).status_code == 400
            )
            assert (
                client.get("/api/traces", params={"start_date": "not-a-date"}).status_code
                == 400
            )


# ===========================================================================
# Adaptive observation policy - end-to-end integration
# ===========================================================================


@pytest.mark.integration
class TestAdaptiveObservationIntegration:
    """End-to-end integration tests for AGENTLENS_OBSERVATION_POLICY=adaptive.

    These tests run the full collector + policy + storage stack and verify:
    - Events are saved with importance_score and observation_level populated.
    - D4 guarantee: FAILURE events always have observation_level=full.
    - API exposes both fields in the serialized event payload.
    - observation_summary is present in the RunResponse.
    """

    @pytest.mark.asyncio
    async def test_adaptive_events_have_observation_fields(self, monkeypatch):
        """Events in adaptive mode carry importance_score and observation_level."""
        monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "adaptive")

        from agentlens.events import EventType, EventStatus
        from agentlens.observation import build_policy_and_scorer
        from agentlens.sdk_collector.collector import EventCollector
        from agentlens.storage import InMemoryEventStore, InMemoryTraceStore

        store = InMemoryEventStore()
        scorer, policy = build_policy_and_scorer()
        collector = EventCollector(storage_backend=store, scorer=scorer, policy=policy)

        await collector.start_trace(agent_id="integration-agent")
        ev = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "hello"},
            output_data={"response": "world"},
            status=EventStatus.SUCCESS,
        )
        await collector.end_trace()

        # importance_score must be in [0, 1] and level must be an ObservationLevel.
        from agentlens.events import ObservationLevel
        assert 0.0 <= ev.importance_score <= 1.0
        assert isinstance(ev.observation_level, ObservationLevel)

    @pytest.mark.asyncio
    async def test_d4_failure_always_full(self, monkeypatch):
        """D4: FAILURE events are always FULL, regardless of score."""
        monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "adaptive")

        from agentlens.events import EventType, EventStatus, ObservationLevel
        from agentlens.observation import build_policy_and_scorer
        from agentlens.sdk_collector.collector import EventCollector
        from agentlens.storage import InMemoryEventStore

        store = InMemoryEventStore()
        scorer, policy = build_policy_and_scorer()
        collector = EventCollector(storage_backend=store, scorer=scorer, policy=policy)

        await collector.start_trace(agent_id="failure-agent")
        ev = await collector.record_event(
            event_type=EventType.LLM_CALL,
            status=EventStatus.FAILURE,
            metadata={"exception_type": "RuntimeError", "exception_message": "boom"},
        )
        await collector.end_trace()

        assert ev.observation_level == ObservationLevel.FULL

    @pytest.mark.asyncio
    async def test_api_exposes_observation_fields(self, monkeypatch):
        """API /api/run returns importance_score and observation_level per event."""
        monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "adaptive")
        # Force the deterministic offline runner (a real key in .env would
        # otherwise trigger a live Gemini call).
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from fastapi.testclient import TestClient
        from agentlens.web_api import app

        # NOTE: the module is deliberately *not* reloaded here.  Reloading
        # ``agentlens.web_api`` rebinds the module-level ``event_store`` /
        # ``trace_store`` singletons, which breaks the identity-sensitive
        # tests in ``tests/test_web_api.py``.  The observation policy is built
        # per request (``_AgentRunner.__init__``), so the env var above is
        # picked up without a reload.
        with TestClient(app) as client:
            resp = client.post("/api/run", json={"query": "test adaptive"})
            assert resp.status_code == 200
            data = resp.json()

            # observation_summary should be present
            assert "observation_summary" in data
            assert isinstance(data["observation_summary"], dict)

            # Each event must have both fields
            for ev in data["events"]:
                assert "importance_score" in ev, f"Missing importance_score in {ev}"
                assert "observation_level" in ev, f"Missing observation_level in {ev}"
                assert isinstance(ev["importance_score"], float)
                assert ev["observation_level"] in ("full", "compressed", "metadata_only", "drop")
