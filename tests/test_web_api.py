"""
Tests for the AgentLens Web API (``agentlens.web_api``).

Covers:
- Successful agent run (``POST /api/run``)
- Empty / blank query validation
- Gemini failure propagation (502)
- Trace retrieval (``GET /api/traces/{trace_id}``)
- Trace not found (404)
- Parent IDs are preserved
- Every child parent_id exists in the same trace
- Input and output data are returned
- Failure filtering returns only failed events
- Invalid filters return HTTP 400
- Success and failure counts are correct
- Empty failure results return an empty list
- Trace retrieval still works without filters
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentlens.events import AgentEvent, EventStatus, EventType, Trace
from agentlens.web_api import app, event_store, trace_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stores(monkeypatch):
    """Reset stores and force deterministic offline runs by default."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()
    yield
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()


@pytest.fixture
def client():
    """Synchronous test client for FastAPI."""
    return TestClient(app)


@pytest.fixture
def run_response(client: TestClient):
    """Run the agent once and return the parsed JSON response."""
    resp = client.post("/api/run", json={"query": "How can AI agents be debugged?"})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def trace_id(run_response):
    """Return the trace_id from a completed run."""
    return run_response["trace_id"]


# ---------------------------------------------------------------------------
# Helper: inject a synthetic trace with configurable event statuses
# ---------------------------------------------------------------------------


def _build_synthetic_trace(
    statuses: list[EventStatus],
    event_types: list[EventType] | None = None,
) -> tuple[str, list[str]]:
    """
    Create and save a synthetic Trace with events whose statuses are given.

    Returns ``(trace_id_str, list_of_event_id_strs)``.
    """
    import asyncio

    trace_uuid = uuid4()
    agent = "test-agent"

    if event_types is None:
        event_types = [EventType.TOOL_CALL] * len(statuses)

    events = []
    parent: AgentEvent | None = None
    for i, (st, et) in enumerate(zip(statuses, event_types)):
        ev = AgentEvent(
            trace_id=trace_uuid,
            agent_id=agent,
            event_type=et,
            status=st,
            input_data={"step": i},
            output_data={"result": f"step-{i}"} if st != EventStatus.FAILURE
            else {"error": "something went wrong"},
            latency_ms=float(10 * (i + 1)),
            parent_id=parent.event_id if parent else None,
            metadata={"index": i},
        )
        events.append(ev)
        parent = ev

    trace = Trace(
        trace_id=trace_uuid,
        agent_id=agent,
        status=EventStatus.FAILURE
        if EventStatus.FAILURE in statuses
        else EventStatus.SUCCESS,
        events=events,
        event_count=len(events),
        total_latency_ms=sum(e.latency_ms or 0.0 for e in events),
    )

    async def _save():
        for ev in events:
            await event_store.save_event(ev)
        await trace_store.save_trace(trace)

    asyncio.run(_save())
    return str(trace_uuid), [str(e.event_id) for e in events]


# ---------------------------------------------------------------------------
# POST /api/run — successful run (offline / no Gemini key)
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    """Tests for POST /api/run."""

    def test_successful_run(self, client: TestClient):
        """A valid query should return a complete trace with all event fields."""
        response = client.post(
            "/api/run",
            json={"query": "How can AI agents be debugged?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["trace_id"]  # non-empty UUID string
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["events"], list)
        assert len(data["events"]) > 0
        assert isinstance(data["total_latency_ms"], (int, float))
        assert data["total_latency_ms"] >= 0

        # Verify each event has the required serialized fields.
        for event in data["events"]:
            assert "event_id" in event
            assert "trace_id" in event
            assert "event_type" in event
            assert "status" in event
            assert "timestamp" in event
            assert "latency_ms" in event
            assert "parent_id" in event  # may be null
            assert "agent_id" in event
            assert "metadata" in event
            assert "children" in event

    def test_successful_run_stores_trace(self, client: TestClient):
        """After POST /api/run the trace should be retrievable via GET."""
        run_resp = client.post(
            "/api/run",
            json={"query": "What is observability?"},
        )
        assert run_resp.status_code == 200
        tid = run_resp.json()["trace_id"]

        get_resp = client.get(f"/api/traces/{tid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["trace_id"] == tid

    def test_empty_query_returns_422(self, client: TestClient):
        """An empty string query should be rejected by Pydantic validation."""
        response = client.post("/api/run", json={"query": ""})
        assert response.status_code == 422

    def test_blank_query_returns_422(self, client: TestClient):
        """A whitespace-only query should be rejected."""
        response = client.post("/api/run", json={"query": "   "})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient):
        """Omitting the query field entirely should be rejected."""
        response = client.post("/api/run", json={})
        assert response.status_code == 422

    def test_gemini_failure_returns_502(self, client: TestClient):
        """When Gemini raises an exception the API should return 502."""
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError(
            "Gemini quota exceeded"
        )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("agentlens.web_api.os.getenv", side_effect=lambda k, *a: {
                "GEMINI_API_KEY": "fake-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "CORS_ORIGINS": "*",
            }.get(k, a[0] if a else None)):
                with patch("google.genai.Client", return_value=fake_client):
                    response = client.post(
                        "/api/run",
                        json={"query": "trigger gemini failure"},
                    )

        assert response.status_code == 502
        assert "Agent execution failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/traces/{trace_id} — basic retrieval
# ---------------------------------------------------------------------------


class TestTraceEndpoint:
    """Tests for GET /api/traces/{trace_id}."""

    def test_trace_retrieval_after_run(self, client: TestClient):
        """Running the agent then fetching the trace should succeed."""
        run_resp = client.post(
            "/api/run",
            json={"query": "Explain causal tracing."},
        )
        assert run_resp.status_code == 200
        tid = run_resp.json()["trace_id"]

        trace_resp = client.get(f"/api/traces/{tid}")
        assert trace_resp.status_code == 200

        data = trace_resp.json()
        assert data["trace_id"] == tid
        assert data["agent_id"] == "multistep-research-agent"
        assert data["status"] == "success"
        assert data["event_count"] > 0
        assert data["total_latency_ms"] >= 0
        assert isinstance(data["events"], list)
        # event_count matches the number of events returned when no filter
        assert len(data["events"]) == data["event_count"]

    def test_trace_not_found_returns_404(self, client: TestClient):
        """Requesting a non-existent trace should return 404."""
        fake_id = str(uuid4())
        response = client.get(f"/api/traces/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_invalid_trace_id_returns_400(self, client: TestClient):
        """A malformed UUID should return 400."""
        response = client.get("/api/traces/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid trace_id" in response.json()["detail"]

    def test_trace_retrieval_without_filters(self, client: TestClient, trace_id: str):
        """Trace retrieval without any query params should return all events."""
        resp = client.get(f"/api/traces/{trace_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] > 0
        assert len(data["events"]) == data["event_count"]


# ---------------------------------------------------------------------------
# Parent / child relationship tests
# ---------------------------------------------------------------------------


class TestParentChildRelationships:
    """Tests for parent_id / children fields in event responses."""

    def test_parent_ids_are_preserved(self, client: TestClient, run_response):
        """Events that were recorded with a parent should carry parent_id."""
        events = run_response["events"]
        # The workflow sets parent relationships for search & retrieval steps.
        parent_ids = [e["parent_id"] for e in events if e["parent_id"] is not None]
        assert len(parent_ids) > 0, "Expected at least some events with a parent_id"

    def test_every_parent_id_exists_in_trace(self, client: TestClient, run_response):
        """Every non-null parent_id must reference an event_id in the same trace."""
        events = run_response["events"]
        all_ids = {e["event_id"] for e in events}
        for event in events:
            if event["parent_id"] is not None:
                assert event["parent_id"] in all_ids, (
                    f"parent_id {event['parent_id']!r} not found in trace events"
                )

    def test_children_field_is_consistent(self, client: TestClient, run_response):
        """children list should be the inverse of parent_id references."""
        events = run_response["events"]
        children_by_id: dict[str, list[str]] = {e["event_id"]: e["children"] for e in events}

        for event in events:
            if event["parent_id"] is not None:
                assert event["event_id"] in children_by_id.get(event["parent_id"], []), (
                    f"Event {event['event_id']} not found in parent's children list"
                )

    def test_trace_endpoint_preserves_parent_ids(self, client: TestClient, trace_id: str):
        """GET /api/traces/{trace_id} should also return parent_ids correctly."""
        resp = client.get(f"/api/traces/{trace_id}")
        assert resp.status_code == 200
        events = resp.json()["events"]
        all_ids = {e["event_id"] for e in events}
        for event in events:
            if event["parent_id"] is not None:
                assert event["parent_id"] in all_ids

    def test_synthetic_parent_child_chain(self, client: TestClient):
        """A synthetic chain where each event's parent is the previous one."""
        tid, eids = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.SUCCESS, EventStatus.SUCCESS],
            event_types=[EventType.DECISION, EventType.TOOL_CALL, EventType.RETRIEVAL],
        )
        resp = client.get(f"/api/traces/{tid}")
        assert resp.status_code == 200
        events = resp.json()["events"]

        # events[1].parent_id == events[0].event_id
        assert events[1]["parent_id"] == events[0]["event_id"]
        # events[2].parent_id == events[1].event_id
        assert events[2]["parent_id"] == events[1]["event_id"]

        # children of events[0] includes events[1]
        assert events[1]["event_id"] in events[0]["children"]


# ---------------------------------------------------------------------------
# Complete event data tests
# ---------------------------------------------------------------------------


class TestCompleteEventData:
    """Verify every event contains the full required field set."""

    _REQUIRED_FIELDS = {
        "event_id", "trace_id", "parent_id", "agent_id",
        "event_type", "status", "timestamp",
        "input_data", "output_data", "latency_ms", "metadata", "children",
    }

    def test_run_events_have_all_fields(self, client: TestClient, run_response):
        """POST /api/run events must include all required fields."""
        for event in run_response["events"]:
            missing = self._REQUIRED_FIELDS - set(event.keys())
            assert not missing, f"Event missing fields: {missing}"

    def test_trace_events_have_all_fields(self, client: TestClient, trace_id: str):
        """GET /api/traces events must include all required fields."""
        resp = client.get(f"/api/traces/{trace_id}")
        assert resp.status_code == 200
        for event in resp.json()["events"]:
            missing = self._REQUIRED_FIELDS - set(event.keys())
            assert not missing, f"Event missing fields: {missing}"

    def test_input_output_data_are_returned(self, client: TestClient):
        """Events should expose their input_data and output_data."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS],
            event_types=[EventType.TOOL_CALL],
        )
        resp = client.get(f"/api/traces/{tid}")
        event = resp.json()["events"][0]
        assert event["input_data"] == {"step": 0}
        assert event["output_data"] == {"result": "step-0"}

    def test_failure_event_output_data_contains_error(self, client: TestClient):
        """A failure event's output_data should carry an error key."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.FAILURE],
            event_types=[EventType.TOOL_CALL],
        )
        resp = client.get(f"/api/traces/{tid}")
        event = resp.json()["events"][0]
        assert event["status"] == "failure"
        assert "error" in event["output_data"]

    def test_llm_call_prompt_and_response_preserved(self, client: TestClient, run_response):
        """LLM_CALL events must keep prompt in input_data and response in output_data."""
        llm_events = [e for e in run_response["events"] if e["event_type"] == "llm_call"]
        assert llm_events, "Expected at least one llm_call event"
        for e in llm_events:
            assert e["input_data"] is not None
            assert "prompt" in e["input_data"]
            assert e["output_data"] is not None
            assert "response" in e["output_data"]


# ---------------------------------------------------------------------------
# Failure-event filtering
# ---------------------------------------------------------------------------


class TestStatusFilter:
    """Tests for ?status= query param on GET /api/traces/{trace_id}."""

    def test_filter_failure_returns_only_failed_events(self, client: TestClient):
        """?status=failure must return only events with status=failure."""
        tid, _ = _build_synthetic_trace(
            statuses=[
                EventStatus.SUCCESS,
                EventStatus.FAILURE,
                EventStatus.SUCCESS,
            ]
        )
        resp = client.get(f"/api/traces/{tid}?status=failure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 1
        assert all(e["status"] == "failure" for e in data["events"])

    def test_filter_success_returns_only_success_events(self, client: TestClient):
        """?status=success must return only events with status=success."""
        tid, _ = _build_synthetic_trace(
            statuses=[
                EventStatus.SUCCESS,
                EventStatus.FAILURE,
                EventStatus.SUCCESS,
            ]
        )
        resp = client.get(f"/api/traces/{tid}?status=success")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 2
        assert all(e["status"] == "success" for e in data["events"])

    def test_filter_all_returns_every_event(self, client: TestClient):
        """?status=all must return every event regardless of status."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE]
        )
        resp = client.get(f"/api/traces/{tid}?status=all")
        assert resp.status_code == 200
        assert resp.json()["event_count"] == 2

    def test_filter_partial_returns_partial_events(self, client: TestClient):
        """?status=partial must return only partial events."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.PARTIAL, EventStatus.SUCCESS]
        )
        resp = client.get(f"/api/traces/{tid}?status=partial")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 1
        assert data["events"][0]["status"] == "partial"

    def test_filter_pending_returns_pending_events(self, client: TestClient):
        """?status=pending must return only pending events."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.PENDING, EventStatus.SUCCESS]
        )
        resp = client.get(f"/api/traces/{tid}?status=pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 1
        assert data["events"][0]["status"] == "pending"

    def test_empty_failure_filter_returns_empty_list(self, client: TestClient):
        """?status=failure on an all-success trace returns an empty events list."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.SUCCESS]
        )
        resp = client.get(f"/api/traces/{tid}?status=failure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 0
        assert data["events"] == []

    def test_invalid_status_filter_returns_400(self, client: TestClient):
        """An unknown status value should return 400."""
        tid, _ = _build_synthetic_trace(statuses=[EventStatus.SUCCESS])
        resp = client.get(f"/api/traces/{tid}?status=bad_value")
        assert resp.status_code == 400
        assert "Invalid status filter" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Event-type filter
# ---------------------------------------------------------------------------


class TestEventTypeFilter:
    """Tests for ?event_type= query param."""

    def test_filter_by_event_type(self, client: TestClient):
        """?event_type=tool_call must return only tool_call events."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.SUCCESS],
            event_types=[EventType.TOOL_CALL, EventType.LLM_CALL],
        )
        resp = client.get(f"/api/traces/{tid}?event_type=tool_call")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 1
        assert data["events"][0]["event_type"] == "tool_call"

    def test_combined_status_and_event_type_filter(self, client: TestClient):
        """Combining status=failure&event_type=tool_call returns only matching events."""
        tid, _ = _build_synthetic_trace(
            statuses=[
                EventStatus.FAILURE,
                EventStatus.FAILURE,
                EventStatus.SUCCESS,
            ],
            event_types=[
                EventType.TOOL_CALL,
                EventType.LLM_CALL,
                EventType.TOOL_CALL,
            ],
        )
        resp = client.get(f"/api/traces/{tid}?status=failure&event_type=tool_call")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 1
        assert data["events"][0]["event_type"] == "tool_call"
        assert data["events"][0]["status"] == "failure"

    def test_invalid_event_type_filter_returns_400(self, client: TestClient):
        """An unknown event_type should return 400."""
        tid, _ = _build_synthetic_trace(statuses=[EventStatus.SUCCESS])
        resp = client.get(f"/api/traces/{tid}?event_type=nonexistent_type")
        assert resp.status_code == 400
        assert "Invalid event_type filter" in resp.json()["detail"]

    def test_no_matching_event_type_returns_empty(self, client: TestClient):
        """A valid event_type with no matches returns an empty events list."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS],
            event_types=[EventType.TOOL_CALL],
        )
        resp = client.get(f"/api/traces/{tid}?event_type=llm_call")
        assert resp.status_code == 200
        assert resp.json()["event_count"] == 0
        assert resp.json()["events"] == []


# ---------------------------------------------------------------------------
# Trace status summaries
# ---------------------------------------------------------------------------


class TestTraceSummary:
    """Tests for summary fields in TraceResponse."""

    def test_all_success_trace_summary(self, client: TestClient):
        """All-success trace should have correct counts and empty failure list."""
        tid, _ = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.SUCCESS, EventStatus.SUCCESS]
        )
        data = client.get(f"/api/traces/{tid}").json()
        assert data["success_count"] == 3
        assert data["failure_count"] == 0
        assert data["partial_count"] == 0
        assert data["failure_event_ids"] == []

    def test_mixed_trace_summary(self, client: TestClient):
        """Mixed trace should have accurate counts and failure_event_ids."""
        tid, eids = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE, EventStatus.SUCCESS]
        )
        data = client.get(f"/api/traces/{tid}").json()
        assert data["success_count"] == 2
        assert data["failure_count"] == 1
        assert data["failure_event_ids"] == [eids[1]]

    def test_all_failure_trace_summary(self, client: TestClient):
        """All-failure trace should have zero success count."""
        tid, eids = _build_synthetic_trace(
            statuses=[EventStatus.FAILURE, EventStatus.FAILURE]
        )
        data = client.get(f"/api/traces/{tid}").json()
        assert data["success_count"] == 0
        assert data["failure_count"] == 2
        assert set(data["failure_event_ids"]) == set(eids)

    def test_summary_not_affected_by_filter(self, client: TestClient):
        """Summary counts are always over the full event list, not the filtered subset."""
        tid, eids = _build_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE, EventStatus.SUCCESS]
        )
        # Filter to only success events — summary should still show 1 failure
        data = client.get(f"/api/traces/{tid}?status=success").json()
        assert data["event_count"] == 2       # only successes returned
        assert data["success_count"] == 2     # global count
        assert data["failure_count"] == 1     # global count
        assert len(data["failure_event_ids"]) == 1

    def test_run_response_includes_summary_on_retrieval(self, client: TestClient, trace_id: str):
        """Summary fields must appear in the GET /api/traces response after a real run."""
        data = client.get(f"/api/traces/{trace_id}").json()
        assert "success_count" in data
        assert "failure_count" in data
        assert "partial_count" in data
        assert "failure_event_ids" in data
        assert isinstance(data["failure_event_ids"], list)
