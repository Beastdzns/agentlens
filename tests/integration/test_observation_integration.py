"""
End-to-end integration tests for adaptive observability (Phase 10).

These tests drive the **whole application** — collector, observation policy,
storage backend and HTTP API — and assert the guarantees that matter for
production use:

- the API exposes ``importance_score`` and ``observation_level``;
- failures keep their full evidence (D4) all the way to the client;
- full mode stays a strict no-op, so nothing changes unless it is switched on;
- adaptive mode measurably reduces the retained payload;
- observations survive a SQLite round-trip with their fields intact;
- scoring is deterministic and the repeat counter is isolated per trace;
- every observation level preserves the event's identity and structure.

Markers used:
    @pytest.mark.integration — needs no external services
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentlens.events import EventStatus, EventType, ObservationLevel
from agentlens.observation import build_policy_and_scorer
from agentlens.sdk_collector.collector import EventCollector
from agentlens.storage import (
    InMemoryEventStore,
    SqliteEventStore,
    init_db,
)
from agentlens.web_api import app

pytestmark = pytest.mark.integration

VALID_LEVELS = {"full", "compressed", "metadata_only", "drop"}


# ===========================================================================
# Fixtures & helpers
# ===========================================================================


@pytest.fixture()
def offline(monkeypatch):
    """Force the deterministic offline runner (never call a live model)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _adaptive(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "adaptive")


def _full(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "full")


def _payload_bytes(events: list[Any]) -> int:
    """Total serialised size of the retained payloads."""
    total = 0
    for event in events:
        total += len(
            json.dumps(
                {
                    "input": event.input_data,
                    "output": event.output_data,
                    "metadata": event.metadata,
                },
                default=str,
            ).encode()
        )
    return total


async def _record_trace(store, events_spec):
    """Record a trace through the real collector.

    Returns ``(trace_id, stored_events)``.  The trace id is captured *before*
    ``end_trace()`` because ending a trace clears the collector's context.
    """
    scorer, policy = build_policy_and_scorer()
    collector = EventCollector(storage_backend=store, scorer=scorer, policy=policy)
    await collector.start_trace(agent_id="integration-agent")
    for spec in events_spec:
        await collector.record_event(**spec)
    trace_id = collector._ctx.get_trace_id()
    await collector.end_trace()
    return trace_id, await store.get_trace_events(trace_id)


# ===========================================================================
# API surface
# ===========================================================================


class TestAdaptiveApiSurface:
    """The HTTP API must expose the observation fields."""

    def test_run_response_exposes_observation_fields(self, monkeypatch, offline):
        _adaptive(monkeypatch)
        with TestClient(app) as client:
            resp = client.post("/api/run", json={"query": "adaptive api query"})
            assert resp.status_code == 200, resp.text
            data = resp.json()

        assert data["events"], "the run must produce events"
        for event in data["events"]:
            assert 0.0 <= event["importance_score"] <= 1.0
            assert event["observation_level"] in VALID_LEVELS

    def test_observation_summary_matches_events(self, monkeypatch, offline):
        _adaptive(monkeypatch)
        with TestClient(app) as client:
            data = client.post("/api/run", json={"query": "summary query"}).json()

        summary = data["observation_summary"]
        assert isinstance(summary, dict)
        assert sum(summary.values()) == len(data["events"])
        for event in data["events"]:
            assert summary.get(event["observation_level"], 0) >= 1

    def test_trace_detail_also_exposes_observation_fields(self, monkeypatch, offline):
        _adaptive(monkeypatch)
        with TestClient(app) as client:
            run = client.post("/api/run", json={"query": "detail query"}).json()
            detail = client.get(f"/api/traces/{run['trace_id']}")
            assert detail.status_code == 200, detail.text
            events = detail.json()["events"]

        assert events
        for event in events:
            assert "importance_score" in event
            assert event["observation_level"] in VALID_LEVELS


# ===========================================================================
# Full mode stays a no-op
# ===========================================================================


class TestFullModeIsNoOp:
    """With the default policy nothing about the stored data changes."""

    def test_every_event_is_full(self, monkeypatch, offline):
        _full(monkeypatch)
        with TestClient(app) as client:
            data = client.post("/api/run", json={"query": "full mode query"}).json()

        assert data["events"]
        assert all(event["observation_level"] == "full" for event in data["events"])

    def test_payloads_are_preserved(self, monkeypatch, offline):
        _full(monkeypatch)
        with TestClient(app) as client:
            data = client.post("/api/run", json={"query": "full payload query"}).json()

        for event in data["events"]:
            if event["event_type"] != "final_response":
                assert event["input_data"] is not None

    def test_default_policy_is_full_without_env_var(self, monkeypatch, offline):
        monkeypatch.delenv("AGENTLENS_OBSERVATION_POLICY", raising=False)
        with TestClient(app) as client:
            data = client.post("/api/run", json={"query": "default policy query"}).json()

        assert all(event["observation_level"] == "full" for event in data["events"])


# ===========================================================================
# D4 — evidence is never degraded
# ===========================================================================


class TestEvidenceGuarantee:
    """Failures must be observable in full, with their exception details."""

    @pytest.mark.asyncio
    async def test_failure_event_is_full_in_adaptive_mode(self, monkeypatch):
        _adaptive(monkeypatch)
        store = InMemoryEventStore()
        _, events = await _record_trace(
            store,
            [
                {
                    "event_type": EventType.LLM_CALL,
                    "status": EventStatus.FAILURE,
                    "input_data": {"prompt": "boom"},
                    "output_data": {"error": "kaboom"},
                    "metadata": {
                        "exception_type": "RuntimeError",
                        "exception_message": "kaboom",
                    },
                }
            ],
        )

        assert len(events) == 1
        failure = events[0]
        assert failure.observation_level == ObservationLevel.FULL
        assert failure.input_data == {"prompt": "boom"}
        assert failure.output_data == {"error": "kaboom"}
        assert failure.metadata["exception_type"] == "RuntimeError"
        assert failure.metadata["exception_message"] == "kaboom"

    @pytest.mark.asyncio
    async def test_error_retry_and_final_response_are_always_full(self, monkeypatch):
        _adaptive(monkeypatch)
        store = InMemoryEventStore()
        _, events = await _record_trace(
            store,
            [
                {"event_type": EventType.ERROR, "status": EventStatus.SUCCESS},
                {"event_type": EventType.RETRY, "status": EventStatus.SUCCESS},
                {
                    "event_type": EventType.FINAL_RESPONSE,
                    "status": EventStatus.SUCCESS,
                    "output_data": {"answer": "42"},
                },
            ],
        )

        assert [e.observation_level for e in events] == [
            ObservationLevel.FULL,
            ObservationLevel.FULL,
            ObservationLevel.FULL,
        ]


# ===========================================================================
# Adaptive actually reduces the retained payload
# ===========================================================================


class TestPayloadReduction:
    """Adaptive mode must store less than full mode for a repetitive trace."""

    @pytest.mark.asyncio
    async def test_adaptive_stores_less_payload_than_full(self, monkeypatch):
        specs = [
            {
                "event_type": EventType.MEMORY_READ,
                "status": EventStatus.SUCCESS,
                "input_data": {"query": "x" * 500},
                "output_data": {"rows": "y" * 500},
            }
            for _ in range(6)
        ]

        _full(monkeypatch)
        full_store = InMemoryEventStore()
        _, full_events = await _record_trace(full_store, specs)

        _adaptive(monkeypatch)
        adaptive_store = InMemoryEventStore()
        _, adaptive_events = await _record_trace(adaptive_store, specs)

        assert _payload_bytes(adaptive_events) < _payload_bytes(full_events)
        assert len(adaptive_events) == len(full_events)

    @pytest.mark.asyncio
    async def test_event_count_is_unchanged(self, monkeypatch):
        """Observability shrinks, it does not silently lose events."""
        _adaptive(monkeypatch)
        store = InMemoryEventStore()
        specs = [{"event_type": EventType.TOOL_CALL} for _ in range(5)]
        _, events = await _record_trace(store, specs)
        assert len(events) == 5


# ===========================================================================
# Structure preservation
# ===========================================================================


class TestStructurePreservation:
    """Whatever the observation level, the event skeleton is intact."""

    @pytest.mark.asyncio
    async def test_identity_survives_every_level(self, monkeypatch):
        _adaptive(monkeypatch)
        store = InMemoryEventStore()
        scorer, policy = build_policy_and_scorer()
        collector = EventCollector(storage_backend=store, scorer=scorer, policy=policy)

        await collector.start_trace(agent_id="structure-agent")
        recorded = []
        for index in range(4):
            if index == 0:
                recorded.append(
                    await collector.record_event(
                        event_type=EventType.DECISION,
                        status=EventStatus.SUCCESS,
                        input_data={"plan": "p" * 300},
                    )
                )
                collector.push_parent(recorded[0].event_id)
                continue
            recorded.append(
                await collector.record_event(
                    event_type=EventType.MEMORY_READ,
                    status=EventStatus.SUCCESS,
                    input_data={"blob": "z" * 400},
                    output_data={"blob": "z" * 400},
                )
            )
        trace_id = collector._ctx.get_trace_id()
        await collector.end_trace()

        events = await store.get_trace_events(trace_id)
        assert len(events) == len(recorded)

        for stored, original in zip(events, recorded, strict=True):
            assert stored.event_id == original.event_id
            assert stored.trace_id == original.trace_id
            assert stored.parent_id == original.parent_id
            assert stored.event_type == original.event_type
            assert stored.status == original.status
            assert stored.timestamp == original.timestamp
            assert stored.observation_level in set(ObservationLevel)

    @pytest.mark.asyncio
    async def test_dropped_events_keep_parent_links(self, monkeypatch):
        """A dropped event must still point at its parent (causal graph)."""
        _adaptive(monkeypatch)
        store = InMemoryEventStore()
        scorer, policy = build_policy_and_scorer()
        collector = EventCollector(storage_backend=store, scorer=scorer, policy=policy)

        await collector.start_trace(agent_id="drop-agent")
        parent = await collector.record_event(
            event_type=EventType.DECISION,
            status=EventStatus.SUCCESS,
            input_data={"plan": "p" * 300},
        )
        collector.push_parent(parent.event_id)
        for _ in range(4):
            await collector.record_event(
                event_type=EventType.MEMORY_READ,
                status=EventStatus.SUCCESS,
                input_data={"blob": "z" * 400},
            )
        trace_id = collector._ctx.get_trace_id()
        await collector.end_trace()

        events = await store.get_trace_events(trace_id)
        dropped = [e for e in events if e.observation_level == ObservationLevel.DROP]
        assert dropped, "expected at least one dropped event in this scenario"
        for event in dropped:
            assert event.parent_id is not None
            assert event.event_id is not None
            assert event.trace_id is not None


# ===========================================================================
# Determinism & per-trace isolation
# ===========================================================================


class TestDeterminismAndIsolation:
    """Scoring must be reproducible and must not leak between traces."""

    @pytest.mark.asyncio
    async def test_same_trace_scores_twice_are_identical(self, monkeypatch):
        _adaptive(monkeypatch)
        specs = [
            {"event_type": EventType.LLM_CALL, "input_data": {"n": 1}},
            {"event_type": EventType.MEMORY_READ, "input_data": {"n": 2}},
            {"event_type": EventType.TOOL_CALL, "input_data": {"n": 3}},
        ]

        _, first = await _record_trace(InMemoryEventStore(), specs)
        _, second = await _record_trace(InMemoryEventStore(), specs)

        assert [e.importance_score for e in first] == [e.importance_score for e in second]
        assert [e.observation_level for e in first] == [e.observation_level for e in second]

    @pytest.mark.asyncio
    async def test_repeat_counter_is_isolated_per_trace(self, monkeypatch):
        """The same step in a second trace must not be treated as a repeat."""
        _adaptive(monkeypatch)
        specs = [{"event_type": EventType.MEMORY_READ} for _ in range(3)]

        _, first = await _record_trace(InMemoryEventStore(), specs)
        _, second = await _record_trace(InMemoryEventStore(), specs)

        assert [e.importance_score for e in first] == [e.importance_score for e in second]

    @pytest.mark.asyncio
    async def test_repeated_step_scores_lower_than_first(self, monkeypatch):
        _adaptive(monkeypatch)
        specs = [{"event_type": EventType.MEMORY_READ} for _ in range(3)]
        _, events = await _record_trace(InMemoryEventStore(), specs)

        assert len(events) == 3
        assert events[1].importance_score < events[0].importance_score
        assert events[2].importance_score <= events[1].importance_score


# ===========================================================================
# SQLite round-trip
# ===========================================================================


class TestSqliteRoundTrip:
    """Adaptive observations must survive a persistent backend round-trip."""

    @pytest.mark.asyncio
    async def test_observation_fields_survive_sqlite(self, tmp_path, monkeypatch):
        _adaptive(monkeypatch)
        db_path = tmp_path / "observation.db"
        db = await init_db(db_path)
        try:
            store = SqliteEventStore(db)
            specs = [
                {
                    "event_type": EventType.LLM_CALL,
                    "status": EventStatus.SUCCESS,
                    "input_data": {"prompt": "hello"},
                },
                {
                    "event_type": EventType.ERROR,
                    "status": EventStatus.FAILURE,
                    "metadata": {"exception_type": "RuntimeError"},
                },
            ]
            trace_id, events = await _record_trace(store, specs)
            assert events
            for event in events:
                assert 0.0 <= event.importance_score <= 1.0
                assert event.observation_level in set(ObservationLevel)

            failure = [e for e in events if e.status == EventStatus.FAILURE]
            assert failure
            assert failure[0].observation_level == ObservationLevel.FULL
            assert failure[0].metadata["exception_type"] == "RuntimeError"
        finally:
            from agentlens.storage.db import close_database

            await close_database()

    @pytest.mark.asyncio
    async def test_sqlite_rows_match_in_memory_observations(self, tmp_path, monkeypatch):
        """The same trace must score identically on both backends."""
        _adaptive(monkeypatch)
        specs = [{"event_type": EventType.TOOL_CALL} for _ in range(3)]

        _, memory_events = await _record_trace(InMemoryEventStore(), specs)

        db = await init_db(tmp_path / "parity.db")
        try:
            _, sqlite_events = await _record_trace(SqliteEventStore(db), specs)
        finally:
            from agentlens.storage.db import close_database

            await close_database()

        assert [e.importance_score for e in memory_events] == [
            e.importance_score for e in sqlite_events
        ]
        assert [e.observation_level for e in memory_events] == [
            e.observation_level for e in sqlite_events
        ]
