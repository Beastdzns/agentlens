"""
Phase 8 failure-tracking scenario tests (``feature/failure-testing``).

This module is the *failure testing* work-stream of Phase 8.  It complements
``tests/test_failure_capture.py`` (which unit-tests the capture wrappers) by
exercising four end-to-end **scenarios** described in the project roadmap:

1. **Timeout scenarios**     — a slow/aborted LLM call must still leave a
   persisted, identifiable failure event behind.
2. **Invalid-input scenarios** — bad requests must be rejected *without*
   polluting the trace store.
3. **Tool-failure scenarios** — a crashing tool records a TOOL_CALL failure and
   stops the pipeline instead of pretending the downstream steps succeeded.
4. **Retry behaviour**       — a retry is linked to the failed attempt through
   ``retry_of`` / ``attempt`` metadata, visible through the HTTP API.
5. **Exact identification**  — for any failed run, ``failure_event_ids`` names
   *exactly* the failed event(s) and the failed step is identifiable.

Conventions
-----------
* ``asyncio_mode = "auto"`` is configured in ``pyproject.toml``.
* The autouse fixture clears the in-memory stores and removes ``GEMINI_API_KEY``
  so runs are deterministic and offline unless a test opts into a fake client.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from agentlens.events import AgentEvent, EventStatus, EventType, Trace
from agentlens.sdk_collector import EventCollector
from agentlens.web_api import app, event_store, trace_store


# ===========================================================================
# Fixtures & helpers
# ===========================================================================


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
def client() -> TestClient:
    """Synchronous FastAPI test client."""
    return TestClient(app)


@contextlib.contextmanager
def gemini_raising(exc: Exception) -> Iterator[MagicMock]:
    """
    Patch the Gemini client so any ``generate_content`` call raises ``exc``.

    Mirrors the env-patching pattern used elsewhere in the suite so the
    ``_AgentRunner`` takes its live-Gemini branch but the call always fails.
    """
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = exc
    env = {
        "GEMINI_API_KEY": "fake-key",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "CORS_ORIGINS": "*",
    }
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        with patch(
            "agentlens.web_api.os.getenv",
            side_effect=lambda k, *a: env.get(k, a[0] if a else None),
        ):
            with patch("google.genai.Client", return_value=fake_client):
                yield fake_client


def _persisted_trace() -> Trace:
    """Return the single trace currently stored (asserting there is exactly one)."""
    assert len(trace_store._traces) == 1, (
        f"Expected exactly 1 stored trace, found {len(trace_store._traces)}"
    )
    return next(iter(trace_store._traces.values()))


def _failure_events(trace: Trace) -> list[AgentEvent]:
    return [e for e in trace.events if e.status == EventStatus.FAILURE]


def _save_synthetic_trace(
    statuses: list[EventStatus],
    event_types: list[EventType] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """
    Persist a synthetic trace and return ``(trace_id_str, event_id_strs)``.

    Each event is a child of the previous one, matching a linear agent run.
    """
    trace_uuid = uuid4()
    agent = "test-agent"
    if event_types is None:
        event_types = [EventType.TOOL_CALL] * len(statuses)
    if metadatas is None:
        metadatas = [{} for _ in statuses]

    events: list[AgentEvent] = []
    parent: AgentEvent | None = None
    for i, (st, et, meta) in enumerate(zip(statuses, event_types, metadatas)):
        ev = AgentEvent(
            trace_id=trace_uuid,
            agent_id=agent,
            event_type=et,
            status=st,
            input_data={"step": i},
            output_data={"error": "boom"} if st == EventStatus.FAILURE else {"result": f"step-{i}"},
            latency_ms=float(10 * (i + 1)),
            parent_id=parent.event_id if parent else None,
            metadata=meta,
        )
        events.append(ev)
        parent = ev

    trace = Trace(
        trace_id=trace_uuid,
        agent_id=agent,
        status=EventStatus.FAILURE if EventStatus.FAILURE in statuses else EventStatus.SUCCESS,
        events=events,
        event_count=len(events),
        total_latency_ms=sum(e.latency_ms or 0.0 for e in events),
    )

    async def _save() -> None:
        for ev in events:
            await event_store.save_event(ev)
        await trace_store.save_trace(trace)

    asyncio.run(_save())
    return str(trace_uuid), [str(e.event_id) for e in events]


# ===========================================================================
# 1. Timeout scenarios
# ===========================================================================


class TestTimeoutScenarios:
    """A timed-out LLM call must persist an identifiable failure event."""

    def test_gemini_timeout_returns_502_and_persists_trace(self, client: TestClient):
        """A TimeoutError from Gemini surfaces as HTTP 502 and a stored FAILURE trace."""
        with gemini_raising(TimeoutError("Gemini request timed out after 30s")):
            resp = client.post("/api/run", json={"query": "timeout scenario"})

        assert resp.status_code == 502
        assert "Agent execution failed" in resp.json()["detail"]

        trace = _persisted_trace()
        assert trace.status == EventStatus.FAILURE

    def test_timeout_records_exactly_one_failure_event(self, client: TestClient):
        """Exactly one FAILURE event is recorded for a single timed-out step."""
        with gemini_raising(TimeoutError("read timed out")):
            client.post("/api/run", json={"query": "timeout scenario"})

        trace = _persisted_trace()
        failures = _failure_events(trace)
        assert len(failures) == 1, (
            f"Expected exactly 1 FAILURE event, found {len(failures)}"
        )

    def test_timeout_failure_carries_exception_type_and_message(self, client: TestClient):
        """The failure event stores the exception class and message."""
        with gemini_raising(TimeoutError("Gemini request timed out after 30s")):
            client.post("/api/run", json={"query": "timeout scenario"})

        failure = _failure_events(_persisted_trace())[0]
        assert failure.metadata is not None
        assert failure.metadata["exception_type"] == "TimeoutError"
        assert "timed out" in failure.metadata["exception_message"]

    def test_timeout_failure_identifies_the_step(self, client: TestClient):
        """The failed step is identifiable from metadata (planning fires first)."""
        with gemini_raising(TimeoutError("boom")):
            client.post("/api/run", json={"query": "timeout scenario"})

        failure = _failure_events(_persisted_trace())[0]
        assert failure.event_type == EventType.LLM_CALL
        assert failure.metadata is not None
        assert failure.metadata.get("step") == "planning"

    def test_timeout_stops_pipeline_before_final_response(self, client: TestClient):
        """A timeout at step 1 must not record a FINAL_RESPONSE event."""
        with gemini_raising(TimeoutError("boom")):
            client.post("/api/run", json={"query": "timeout scenario"})

        trace = _persisted_trace()
        assert not any(e.event_type == EventType.FINAL_RESPONSE for e in trace.events), (
            "Pipeline should have stopped before the final-response step"
        )

    def test_timeout_visible_through_trace_api(self, client: TestClient):
        """The trace endpoint must expose the timeout as failure_event_ids."""
        with gemini_raising(TimeoutError("boom")):
            client.post("/api/run", json={"query": "timeout scenario"})

        trace = _persisted_trace()
        failure = _failure_events(trace)[0]

        resp = client.get(f"/api/traces/{trace.trace_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failure"
        assert body["failure_count"] == 1
        assert body["failure_event_ids"] == [str(failure.event_id)]


# ===========================================================================
# 2. Invalid-input scenarios
# ===========================================================================


class TestInvalidInputScenarios:
    """Bad requests are rejected without creating or corrupting traces."""

    def test_blank_query_rejected_without_trace(self, client: TestClient):
        resp = client.post("/api/run", json={"query": "   "})
        assert resp.status_code == 422
        assert len(trace_store._traces) == 0, "Validation failure must not create a trace"

    def test_empty_query_rejected_without_trace(self, client: TestClient):
        resp = client.post("/api/run", json={"query": ""})
        assert resp.status_code == 422
        assert len(trace_store._traces) == 0

    def test_missing_query_rejected_without_trace(self, client: TestClient):
        resp = client.post("/api/run", json={})
        assert resp.status_code == 422
        assert len(trace_store._traces) == 0

    def test_malformed_trace_id_returns_400(self, client: TestClient):
        resp = client.get("/api/traces/not-a-uuid")
        assert resp.status_code == 400
        assert "Invalid trace_id" in resp.json()["detail"]

    def test_unknown_trace_returns_404(self, client: TestClient):
        resp = client.get(f"/api/traces/{uuid4()}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_invalid_status_filter_returns_400(self, client: TestClient):
        tid, _ = _save_synthetic_trace([EventStatus.SUCCESS])
        resp = client.get(f"/api/traces/{tid}?status=definitely_not_valid")
        assert resp.status_code == 400
        assert "Invalid status filter" in resp.json()["detail"]

    def test_invalid_event_type_filter_returns_400(self, client: TestClient):
        tid, _ = _save_synthetic_trace([EventStatus.SUCCESS])
        resp = client.get(f"/api/traces/{tid}?event_type=not_a_type")
        assert resp.status_code == 400
        assert "Invalid event_type filter" in resp.json()["detail"]


# ===========================================================================
# 3. Tool-failure scenarios
# ===========================================================================


class TestToolFailureScenarios:
    """A crashing tool records a TOOL_CALL failure and halts the pipeline."""

    async def test_tool_failure_records_tool_call_failure_event(self):
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="tool-agent")

        with pytest.raises(ConnectionError):
            async with col.capture_tool_call(
                "web_search",
                input_data={"query": "x"},
                metadata={"step": "search"},
            ):
                raise ConnectionError("search backend down")

        await col.end_trace()
        events = list(event_store._events.values())
        assert len(events) == 1
        failure = events[0]
        assert failure.event_type == EventType.TOOL_CALL
        assert failure.status == EventStatus.FAILURE

    async def test_tool_failure_carries_exception_details(self):
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="tool-agent")

        with pytest.raises(ConnectionError):
            async with col.capture_tool_call("web_search", metadata={"step": "search"}):
                raise ConnectionError("search backend down")

        await col.end_trace()
        failure = list(event_store._events.values())[0]
        assert failure.metadata is not None
        assert failure.metadata["exception_type"] == "ConnectionError"
        assert "search backend down" in failure.metadata["exception_message"]
        assert failure.metadata.get("tool") == "web_search"

    async def test_tool_failure_reraises_original_exception(self):
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="tool-agent")

        with pytest.raises(ConnectionError, match="search backend down"):
            async with col.capture_tool_call("web_search"):
                raise ConnectionError("search backend down")

        await col.end_trace()

    async def test_failed_tool_step_inherits_parent(self):
        """A failed tool call must keep the parent context for the UI breadcrumb."""
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="tool-agent")

        parent = await col.record_event(
            event_type=EventType.DECISION,
            output_data={"plan": "do the thing"},
            metadata={"step": "planning"},
        )
        col.push_parent(parent.event_id)
        with pytest.raises(RuntimeError):
            async with col.capture_tool_call("web_search", metadata={"step": "search"}):
                raise RuntimeError("tool exploded")
        col.pop_parent()
        await col.end_trace()

        failure = next(e for e in event_store._events.values() if e.status == EventStatus.FAILURE)
        assert failure.parent_id == parent.event_id

    async def test_pipeline_stops_after_tool_failure(self):
        """No step recorded after the failing tool call — no fake successes."""
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="tool-agent")
        reached: list[str] = []

        async def pipeline() -> None:
            async with col.capture_step(EventType.RETRIEVAL, metadata={"step": "retrieval"}):
                pass
            reached.append("retrieval")
            async with col.capture_tool_call("web_search", metadata={"step": "search"}):
                raise ConnectionError("search backend down")
            reached.append("search")  # must never run
            async with col.capture_step(EventType.FINAL_RESPONSE, metadata={"step": "final"}):
                pass
            reached.append("final")

        with pytest.raises(ConnectionError):
            await pipeline()

        await col.end_trace()
        assert reached == ["retrieval"], "Pipeline must stop at the failing tool call"

        statuses = [e.status for e in event_store._events.values()]
        assert statuses.count(EventStatus.FAILURE) == 1


# ===========================================================================
# 4. Retry behaviour
# ===========================================================================


class TestRetryBehavior:
    """Retries are linked to the failed attempt via retry_of / attempt."""

    async def test_retry_links_to_failed_event(self):
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="retry-agent")

        with pytest.raises(ConnectionError):
            async with col.capture_tool_call("flaky_tool", metadata={"step": "tool"}):
                raise ConnectionError("first attempt failed")
        failed = list(event_store._events.values())[0]

        async with col.capture_tool_call(
            "flaky_tool",
            metadata={"step": "tool"},
            retry_of=failed.event_id,
            attempt=2,
        ) as result:
            result[0] = {"ok": True}

        await col.end_trace()
        retry_event = [
            e for e in event_store._events.values() if e.status == EventStatus.SUCCESS
        ][0]
        assert retry_event.metadata is not None
        assert retry_event.metadata["retry_of"] == str(failed.event_id)
        assert retry_event.metadata["attempt"] == 2

    async def test_first_attempt_has_no_retry_metadata(self):
        col = EventCollector(storage_backend=event_store)
        await col.start_trace(agent_id="retry-agent")

        async with col.capture_step(EventType.RETRIEVAL, metadata={"step": "retrieval"}):
            pass

        await col.end_trace()
        event = list(event_store._events.values())[0]
        assert "retry_of" not in (event.metadata or {})

    def test_retry_chain_is_visible_through_api(self, client: TestClient):
        """The API exposes retry_of so the UI can render retry breadcrumbs."""
        tid, eids = _save_synthetic_trace(
            statuses=[EventStatus.FAILURE, EventStatus.SUCCESS],
            event_types=[EventType.TOOL_CALL, EventType.TOOL_CALL],
            metadatas=[
                {"step": "search", "exception_type": "ConnectionError"},
                {"step": "search", "retry_of": None, "attempt": 2},
            ],
        )
        # Point the retry at the failed attempt so the link is real.
        async def _fix() -> None:
            retry_event = await event_store.get_event(UUID(eids[1]))
            retry_event.metadata["retry_of"] = eids[0]
            await event_store.save_event(retry_event)

        asyncio.run(_fix())

        resp = client.get(f"/api/traces/{tid}")
        assert resp.status_code == 200
        events = {e["event_id"]: e for e in resp.json()["events"]}
        retry_event = events[eids[1]]
        assert retry_event["metadata"]["retry_of"] == eids[0]
        assert retry_event["metadata"]["attempt"] == 2


# ===========================================================================
# 5. Exact failure identification
# ===========================================================================


class TestExactFailureIdentification:
    """The exact failed event(s) must be reported and identifiable by step."""

    def test_single_failure_is_named_exactly(self, client: TestClient):
        tid, eids = _save_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE, EventStatus.SUCCESS],
            metadatas=[{}, {"step": "synthesis"}, {}],
        )
        body = client.get(f"/api/traces/{tid}").json()
        assert body["failure_event_ids"] == [eids[1]]
        assert body["failure_count"] == 1

    def test_multiple_failures_are_all_named(self, client: TestClient):
        tid, eids = _save_synthetic_trace(
            statuses=[EventStatus.FAILURE, EventStatus.SUCCESS, EventStatus.FAILURE],
        )
        body = client.get(f"/api/traces/{tid}").json()
        assert set(body["failure_event_ids"]) == {eids[0], eids[2]}
        assert body["failure_count"] == 2

    def test_failed_step_identifiable_by_metadata(self, client: TestClient):
        tid, eids = _save_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE],
            event_types=[EventType.TOOL_CALL, EventType.LLM_CALL],
            metadatas=[{"step": "search"}, {"step": "synthesis", "exception_type": "RuntimeError"}],
        )
        body = client.get(f"/api/traces/{tid}").json()
        events = {e["event_id"]: e for e in body["events"]}
        failed = events[eids[1]]
        assert failed["status"] == "failure"
        assert failed["metadata"]["step"] == "synthesis"
        assert failed["metadata"]["exception_type"] == "RuntimeError"

    def test_summary_counts_are_consistent(self, client: TestClient):
        tid, _ = _save_synthetic_trace(
            statuses=[EventStatus.SUCCESS, EventStatus.FAILURE, EventStatus.SUCCESS],
        )
        body = client.get(f"/api/traces/{tid}").json()
        assert body["success_count"] == 2
        assert body["failure_count"] == 1
        assert len(body["failure_event_ids"]) == 1

    def test_clean_run_has_no_failure_ids(self, client: TestClient):
        resp = client.post("/api/run", json={"query": "offline success"})
        assert resp.status_code == 200
        body = client.get(f"/api/traces/{resp.json()['trace_id']}").json()
        assert body["failure_count"] == 0
        assert body["failure_event_ids"] == []
