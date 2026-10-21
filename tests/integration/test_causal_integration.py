"""
End-to-end integration tests for the causal graph (Phase 11, member V).

These tests drive the **real HTTP API** against a **real stored trace** and
assert the guarantees the causal graph has to keep:

- the graph endpoint serves the recorded structure;
- unknown / malformed trace ids behave like the rest of the API;
- the graph embedded in the trace response matches the dedicated endpoint;
- failures are flagged, and the structure survives adaptive observation;
- the graph is deterministic and identical on every backend.

Markers used:
    @pytest.mark.integration — needs no external services
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentlens import web_api
from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    Trace,
)
from agentlens.storage import SqliteEventStore, init_db
from agentlens.web_api import app

pytestmark = pytest.mark.integration


# ===========================================================================
# Helpers
# ===========================================================================


def _event(
    trace_id,
    *,
    event_type: EventType = EventType.LLM_CALL,
    status: EventStatus = EventStatus.SUCCESS,
    parent_id=None,
    metadata: Optional[dict[str, Any]] = None,
    input_data: Optional[dict[str, Any]] = None,
    output_data: Optional[dict[str, Any]] = None,
    importance_score: float = 0.5,
    observation_level: ObservationLevel = ObservationLevel.FULL,
) -> AgentEvent:
    return AgentEvent(
        trace_id=trace_id,
        agent_id="causal-agent",
        event_type=event_type,
        status=status,
        parent_id=parent_id,
        metadata=metadata,
        input_data=input_data,
        output_data=output_data,
        importance_score=importance_score,
        observation_level=observation_level,
    )


def _nested_trace() -> Trace:
    """A trace with a parent/child chain and one failed step."""
    trace = Trace(agent_id="causal-agent")
    root = _event(trace.trace_id, event_type=EventType.SUBAGENT_START)
    child = _event(
        trace.trace_id,
        event_type=EventType.TOOL_CALL,
        parent_id=root.event_id,
        input_data={"query": "search"},
        output_data={"tool": "search"},
    )
    failed = _event(
        trace.trace_id,
        event_type=EventType.LLM_CALL,
        status=EventStatus.FAILURE,
        parent_id=root.event_id,
        output_data={"error": "boom"},
        metadata={"exception_type": "RuntimeError", "exception_message": "boom"},
    )
    for event in (root, child, failed):
        trace.add_event(event)
    trace.finalize(EventStatus.FAILURE)
    return trace


async def _store_trace(trace: Trace) -> str:
    """Persist a trace into the API's shared in-memory trace store."""
    await web_api.trace_store.save_trace(trace)
    for event in trace.events:
        await web_api.event_store.save_event(event)
    return str(trace.trace_id)


# ===========================================================================
# API surface
# ===========================================================================


class TestCausalGraphEndpoint:
    """The dedicated graph endpoint must serve the recorded structure."""

    @pytest.mark.asyncio
    async def test_graph_endpoint_returns_nodes_and_edges(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            resp = client.get(f"/api/traces/{trace_id}/causal-graph")
            assert resp.status_code == 200, resp.text
            data = resp.json()

        assert data["trace_id"] == trace_id
        assert len(data["nodes"]) == len(trace.events)
        assert data["edges"], "a parent/child trace must produce edges"

    @pytest.mark.asyncio
    async def test_node_ids_match_the_recorded_event_ids(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            graph = client.get(f"/api/traces/{trace_id}/causal-graph").json()
            detail = client.get(f"/api/traces/{trace_id}").json()

        graph_ids = {node["event_id"] for node in graph["nodes"]}
        event_ids = {event["event_id"] for event in detail["events"]}
        assert graph_ids == event_ids

    @pytest.mark.asyncio
    async def test_parent_of_edges_match_recorded_parent_links(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            graph = client.get(f"/api/traces/{trace_id}/causal-graph").json()

        parent_of = {
            (edge["source"], edge["target"])
            for edge in graph["edges"]
            if edge["relation"] == "parent_of"
        }
        expected = {
            (str(e.parent_id), str(e.event_id))
            for e in trace.events
            if e.parent_id is not None
        }
        assert parent_of == expected

    @pytest.mark.asyncio
    async def test_failed_nodes_are_flagged(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            graph = client.get(f"/api/traces/{trace_id}/causal-graph").json()

        failures = [node for node in graph["nodes"] if node["is_failure"]]
        assert len(failures) == 1
        assert failures[0]["error_message"] == "boom"
        assert failures[0]["exception_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_unknown_trace_returns_404(self):
        with TestClient(app) as client:
            resp = client.get(f"/api/traces/{uuid4()}/causal-graph")
        assert resp.status_code == 404

    def test_malformed_trace_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.get("/api/traces/not-a-uuid/causal-graph")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_trace_yields_an_empty_graph(self):
        trace = Trace(agent_id="causal-agent")
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            resp = client.get(f"/api/traces/{trace_id}/causal-graph")
            assert resp.status_code == 200, resp.text
            data = resp.json()

        assert data["nodes"] == []
        assert data["edges"] == []


# ===========================================================================
# Trace response integration
# ===========================================================================


class TestTraceResponseEmbedding:
    """The trace response carries the graph as an additive field."""

    @pytest.mark.asyncio
    async def test_trace_response_embeds_the_causal_graph(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            data = client.get(f"/api/traces/{trace_id}").json()

        assert data["causal_graph"] is not None
        assert len(data["causal_graph"]["nodes"]) == len(trace.events)

    @pytest.mark.asyncio
    async def test_embedded_graph_matches_the_dedicated_endpoint(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            embedded = client.get(f"/api/traces/{trace_id}").json()["causal_graph"]
            dedicated = client.get(f"/api/traces/{trace_id}/causal-graph").json()

        assert embedded == dedicated

    @pytest.mark.asyncio
    async def test_graph_can_be_excluded(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            data = client.get(
                f"/api/traces/{trace_id}", params={"include_causal_graph": "false"}
            ).json()

        assert data["causal_graph"] is None
        assert data["events"], "the trace itself is unaffected"

    @pytest.mark.asyncio
    async def test_existing_response_contract_is_preserved(self):
        """The additive field must not remove or rename anything."""
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            data = client.get(f"/api/traces/{trace_id}").json()

        for field in (
            "trace_id",
            "agent_id",
            "status",
            "event_count",
            "total_latency_ms",
            "success_count",
            "failure_count",
            "partial_count",
            "pending_count",
            "failure_event_ids",
            "events",
        ):
            assert field in data, f"missing response field: {field}"


# ===========================================================================
# Robustness
# ===========================================================================


class TestCausalGraphRobustness:
    """The graph must never break the API."""

    @pytest.mark.asyncio
    async def test_graph_is_deterministic_across_requests(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            first = client.get(f"/api/traces/{trace_id}/causal-graph").json()
            second = client.get(f"/api/traces/{trace_id}/causal-graph").json()

        assert first == second

    @pytest.mark.asyncio
    async def test_dropped_events_keep_the_causal_structure(self):
        """D5: adaptive observation must not erase the causal structure."""
        trace = Trace(agent_id="causal-agent")
        root = _event(trace.trace_id, event_type=EventType.SUBAGENT_START)
        child = _event(
            trace.trace_id,
            event_type=EventType.TOOL_CALL,
            parent_id=root.event_id,
            observation_level=ObservationLevel.DROP,
            input_data=None,
            output_data=None,
        )
        for event in (root, child):
            trace.add_event(event)
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            graph = client.get(f"/api/traces/{trace_id}/causal-graph").json()

        assert len(graph["nodes"]) == 2
        assert any(
            edge["relation"] == "parent_of" for edge in graph["edges"]
        ), "the parent link must survive a dropped payload"

    @pytest.mark.asyncio
    async def test_graph_survives_a_sqlite_round_trip(self, tmp_path):
        db_path = tmp_path / "causal.db"
        db = await init_db(db_path)
        try:
            store = SqliteEventStore(db)
            trace = _nested_trace()
            for event in trace.events:
                await store.save_event(event)

            from agentlens.storage import SqliteTraceStore

            trace_store = SqliteTraceStore(db)
            await trace_store.save_trace(trace)

            from agentlens.causal import build_graph

            events = await store.get_trace_events(trace.trace_id)
            graph = build_graph(events)
            assert len(graph.nodes) == len(trace.events)
            assert any(edge.relation.value == "parent_of" for edge in graph.edges)
        finally:
            from agentlens.storage.db import close_database

            await close_database()

    @pytest.mark.asyncio
    async def test_graph_payload_is_json_serialisable(self):
        trace = _nested_trace()
        trace_id = await _store_trace(trace)

        with TestClient(app) as client:
            resp = client.get(f"/api/traces/{trace_id}/causal-graph")
            json.dumps(resp.json())  # must not raise

    @pytest.mark.asyncio
    async def test_graph_endpoint_is_listed_in_the_openapi_schema(self):
        with TestClient(app) as client:
            schema = client.get("/openapi.json").json()

        assert "/api/traces/{trace_id}/causal-graph" in schema["paths"]
