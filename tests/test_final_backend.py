"""
Tests for the Phase 12 backend hardening.

Covers the three backend changes that make the local demo work:

1. A run that fails *inside* the pipeline is not an HTTP error — the failed
   trace is returned (``status="failure"``) so the dashboard can show the
   causal analysis of the failure instead of a bare "something went wrong".
2. ``RunResponse`` carries the Phase 11 causal graph and root-cause diagnosis
   (additive, fail-open).
3. The live pipeline records exactly one event per step (the four duplicate
   step events are gone) and the events form a connected parent chain.

Plus the SDK-level building blocks the runner relies on: ``capture_step``
records the step's output (like ``capture_llm_call`` / ``capture_tool_call``
already did) and ``EventCollector.last_event`` exposes the latest event.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentlens.events import EventStatus, EventType
from agentlens.sdk_collector import EventCollector
from agentlens.web_api import app, event_store, trace_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stores(monkeypatch):
    """Reset the module-level stores and force deterministic offline runs."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemini_client(fail_after: int = 0, message: str = "Gemini quota exceeded"):
    """
    Build a fake Gemini client.

    Args:
        fail_after: Number of calls that succeed before the client raises.
            ``0`` fails on the very first call, ``1`` lets planning succeed and
            breaks the synthesis step.
        message: The error message raised by the failing call.
    """
    calls = {"n": 0}

    def _generate_content(**_kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_after:
            return SimpleNamespace(text="1. search 2. retrieve 3. synthesise")
        raise RuntimeError(message)

    fake = MagicMock()
    fake.models.generate_content.side_effect = _generate_content
    return fake


@contextlib.contextmanager
def _patched_gemini(fake_client):
    """Force the runner to use *fake_client* instead of a real Gemini client."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        with patch(
            "agentlens.web_api.os.getenv",
            side_effect=lambda k, *a: {
                "GEMINI_API_KEY": "fake-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "CORS_ORIGINS": "*",
            }.get(k, a[0] if a else None),
        ):
            with patch("google.genai.Client", return_value=fake_client):
                yield


def _key(event: dict) -> str:
    """Return a stable ``event_type:step`` identifier for an event."""
    step = (event.get("metadata") or {}).get("step")
    return f"{event['event_type']}:{step}"


def _by_key(body: dict) -> dict[str, dict]:
    """Index a run response's events by their ``event_type:step`` key."""
    return {_key(e): e for e in body["events"]}


# ===========================================================================
# 1. A failing live run returns the failed trace
# ===========================================================================


class TestFailedRunReturnsTheTrace:
    """A run that fails inside the pipeline must not throw the trace away."""

    def test_failed_run_returns_the_failed_trace_not_a_502(self, client):
        """The failed trace is returned like any other run."""
        with _patched_gemini(_gemini_client()):
            resp = client.post("/api/run", json={"query": "trigger gemini failure"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failure"
        assert body["trace_id"]
        assert body["events"], "the failed trace must carry its events"

    def test_failed_run_reports_the_error(self, client):
        """The error message names the exception and its cause."""
        with _patched_gemini(_gemini_client(message="Gemini quota exceeded")):
            resp = client.post("/api/run", json={"query": "boom"})

        error = resp.json()["error"]
        assert error is not None
        assert "RuntimeError" in error
        assert "Gemini quota exceeded" in error

    def test_failed_trace_is_retrievable_afterwards(self, client):
        """The failed trace is persisted and can be fetched by id."""
        with _patched_gemini(_gemini_client()):
            body = client.post("/api/run", json={"query": "boom"}).json()

        trace = client.get(f"/api/traces/{body['trace_id']}")
        assert trace.status_code == 200
        assert trace.json()["status"] == "failure"
        assert trace.json()["failure_count"] == 1

    def test_failed_run_carries_the_causal_analysis(self, client):
        """The failure comes with the graph and the root-cause diagnosis."""
        with _patched_gemini(_gemini_client()):
            body = client.post("/api/run", json={"query": "boom"}).json()

        assert body["causal_graph"] is not None
        assert body["root_cause"] is not None
        assert body["root_cause"]["verdict"] == "root_cause"

        # The diagnosis points at the event that actually failed.
        failed_ids = {e["event_id"] for e in body["events"] if e["status"] == "failure"}
        assert body["root_cause"]["root_cause_event_id"] in failed_ids
        assert "Gemini quota exceeded" in body["root_cause"]["explanation"]

    def test_failure_at_a_later_step_keeps_the_earlier_chain(self, client):
        """When synthesis fails, the completed steps are still reported."""
        with _patched_gemini(_gemini_client(fail_after=1)):
            body = client.post("/api/run", json={"query": "boom later"}).json()

        assert body["status"] == "failure"
        indexed = _by_key(body)
        assert "tool_call:search" in indexed
        assert "retrieval:retrieval" in indexed
        assert "memory_write:memory-write" in indexed

        failures = [e for e in body["events"] if e["status"] == "failure"]
        assert len(failures) == 1
        assert failures[0]["event_type"] == "llm_call"
        assert body["root_cause"]["root_cause_event_id"] == failures[0]["event_id"]

    def test_run_machinery_failure_is_still_a_502(self, client):
        """
        A failure of the run machinery itself has no trace to return.

        Only failures *inside* a traced step are turned into a failed trace;
        everything else stays a 502.
        """
        with patch(
            "agentlens.web_api._AgentRunner.run",
            side_effect=OSError("storage offline"),
        ):
            resp = client.post("/api/run", json={"query": "boom"})

        assert resp.status_code == 502
        assert "Agent execution failed" in resp.json()["detail"]


# ===========================================================================
# 2. RunResponse carries the causal analysis
# ===========================================================================


class TestRunResponseCarriesCausalAnalysis:
    """``RunResponse`` exposes the Phase 11 analysis (additive, fail-open)."""

    def test_successful_run_includes_the_causal_graph(self, client):
        """A successful run returns the graph of its own trace."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        graph = body["causal_graph"]
        assert graph is not None
        assert len(graph["nodes"]) == len(body["events"])
        assert graph["edges"], "the pipeline must produce parent relations"
        assert {e["relation"] for e in graph["edges"]} <= {
            "parent_of",
            "retry_of",
            "depends_on",
            "influenced",
        }

    def test_successful_run_includes_the_root_cause(self, client):
        """A healthy run is diagnosed as ``no_failure``."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        root_cause = body["root_cause"]
        assert root_cause is not None
        assert root_cause["verdict"] == "no_failure"
        assert root_cause["confidence"] == 0.0

    def test_graph_matches_the_trace_endpoint(self, client):
        """The run's graph is the same graph the trace endpoint returns."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        graph = client.get(f"/api/traces/{body['trace_id']}/causal-graph").json()
        assert len(graph["nodes"]) == len(body["causal_graph"]["nodes"])
        assert len(graph["edges"]) == len(body["causal_graph"]["edges"])

    def test_run_response_stays_backwards_compatible(self, client):
        """The pre-existing response fields are untouched."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        for key in (
            "trace_id",
            "response",
            "events",
            "total_latency_ms",
            "status",
            "observation_summary",
        ):
            assert key in body
        assert body["status"] == "success"
        assert body["response"]


# ===========================================================================
# 3. No duplicate step events
# ===========================================================================


class TestNoDuplicateEvents:
    """Every pipeline step is recorded exactly once."""

    def test_live_run_records_one_event_per_step(self, client):
        """The pipeline produces seven events — one per step."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        assert len(body["events"]) == 7

    def test_no_two_events_share_type_and_step(self, client):
        """No step is recorded twice (the old duplicate-event bug)."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        keys = [_key(e) for e in body["events"]]
        assert len(keys) == len(set(keys)), f"duplicate step events: {keys}"

    def test_each_step_records_its_own_output(self, client):
        """The output of a step is recorded on that step's event."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()
        indexed = _by_key(body)

        assert indexed["retrieval:retrieval"]["output_data"]["documents"]
        assert indexed["final_response:final-response"]["output_data"]["answer"]
        assert indexed["tool_call:search"]["output_data"]["hits"]

    def test_document_count_is_the_real_document_count(self, client):
        """The memory write counts the documents it actually stored."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()
        indexed = _by_key(body)

        documents = indexed["retrieval:retrieval"]["output_data"]["documents"]
        count = indexed["memory_write:memory-write"]["output_data"]["document_count"]
        assert count == len(documents)

    def test_events_carry_their_real_latency(self, client):
        """The recorded events keep the timing of the actual step."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        assert all(e["latency_ms"] is not None for e in body["events"])
        assert body["total_latency_ms"] > 0


# ===========================================================================
# 4. The parent chain is connected
# ===========================================================================


class TestParentChain:
    """The pipeline's events form one causal chain."""

    def test_pipeline_events_form_one_chain(self, client):
        """Each step is a child of the step it depends on."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()
        indexed = _by_key(body)

        chain = [
            ("decision:planning", "tool_call:search"),
            ("tool_call:search", "retrieval:retrieval"),
            ("retrieval:retrieval", "memory_write:memory-write"),
            ("memory_write:memory-write", "llm_call:synthesis"),
            ("llm_call:synthesis", "final_response:final-response"),
        ]
        for parent_key, child_key in chain:
            assert indexed[child_key]["parent_id"] == indexed[parent_key]["event_id"], (
                f"{child_key} should be a child of {parent_key}"
            )

    def test_only_the_roots_have_no_parent(self, client):
        """Exactly two events are roots: the planning call and the decision."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        roots = [e for e in body["events"] if e["parent_id"] is None]
        assert len(roots) == 2
        assert {e["event_type"] for e in roots} == {"llm_call", "decision"}

    def test_every_parent_exists_in_the_trace(self, client):
        """A parent id always refers to an event of the same trace."""
        body = client.post("/api/run", json={"query": "How are agents debugged?"}).json()

        ids = {e["event_id"] for e in body["events"]}
        for event in body["events"]:
            if event["parent_id"] is not None:
                assert event["parent_id"] in ids


# ===========================================================================
# 5. capture_step records the step's output
# ===========================================================================


class TestCaptureStepRecordsOutput:
    """``capture_step`` behaves like its two sibling capture helpers."""

    async def test_capture_step_records_the_output(self):
        """Assigning ``result[0]`` records the output on the event."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        async with collector.capture_step(
            EventType.RETRIEVAL,
            input_data={"query": "x"},
        ) as result:
            result[0] = {"documents": ["a", "b"]}

        await collector.end_trace()

        event = collector.last_event
        assert event.event_type == EventType.RETRIEVAL
        assert event.status == EventStatus.SUCCESS
        assert event.input_data == {"query": "x"}
        assert event.output_data == {"documents": ["a", "b"]}

    async def test_capture_step_without_a_holder_still_works(self):
        """The holder is optional — the old call style keeps working."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        async with collector.capture_step(EventType.TOOL_CALL):
            pass

        await collector.end_trace()

        assert collector.last_event.status == EventStatus.SUCCESS
        assert collector.last_event.output_data is None

    async def test_capture_step_failure_records_the_exception(self):
        """A failing step still records the exception, but no output."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        with pytest.raises(ValueError, match="boom"):
            async with collector.capture_step(EventType.LLM_CALL) as result:
                result[0] = "partial"
                raise ValueError("boom")

        await collector.end_trace()

        event = collector.last_event
        assert event.status == EventStatus.FAILURE
        assert event.output_data is None
        assert event.metadata["exception_type"] == "ValueError"
        assert event.metadata["exception_message"] == "boom"

    async def test_capture_step_holder_starts_empty(self):
        """The holder is a one-element list so ``result[0]`` is assignable."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        async with collector.capture_step(EventType.DECISION) as result:
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] is None

        await collector.end_trace()


# ===========================================================================
# 6. EventCollector.last_event
# ===========================================================================


class TestLastEvent:
    """``last_event`` exposes the most recently recorded event."""

    async def test_last_event_is_set_by_record_event(self):
        """After recording, ``last_event`` is that event."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")
        assert collector.last_event is None

        event = await collector.record_event(
            event_type=EventType.DECISION,
            output_data={"plan": "x"},
        )

        assert collector.last_event is event
        await collector.end_trace()

    async def test_last_event_tracks_the_latest_event(self):
        """Each new recording replaces the previous one."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        first = await collector.record_event(event_type=EventType.DECISION)
        second = await collector.record_event(event_type=EventType.TOOL_CALL)

        assert collector.last_event is second
        assert collector.last_event is not first
        await collector.end_trace()

    async def test_capture_wrappers_update_last_event(self):
        """The capture helpers publish the event they recorded."""
        collector = EventCollector()
        await collector.start_trace(agent_id="test-agent")

        async with collector.capture_tool_call(
            "web_search",
            input_data={"query": "x"},
        ) as result:
            result[0] = {"hits": ["a"]}

        event = collector.last_event
        assert event.event_type == EventType.TOOL_CALL
        assert event.output_data == {"hits": ["a"]}
        await collector.end_trace()
