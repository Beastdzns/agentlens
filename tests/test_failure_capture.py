"""
Tests for automatic failure capture — collector wrappers and pipeline routing.

Coverage
--------
- CaptureStep             — SUCCESS path, FAILURE path (exception_type + message), re-raise
- CaptureLlmCall          — SUCCESS path (result populated), FAILURE path
- CaptureToolCall         — SUCCESS path (result populated), FAILURE path
- RetryLinkage            — retry_of + attempt stored in metadata
- PipelineFailureCapture  — killing a single step (synthesis) still persists
                            exactly one FAILURE event; other steps keep SUCCESS.

Test conventions
----------------
- asyncio_mode = "auto" is set in pyproject.toml.
- Mocks are used for the storage backend so tests never touch disk or network.
- The ``_AgentRunner`` is exercised through the FastAPI test client so the full
  HTTP stack is covered.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentlens.events import AgentEvent, EventStatus, EventType
from agentlens.sdk_collector import EventCollector
from agentlens.web_api import app, event_store, trace_store


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_stores(monkeypatch):
    """Reset in-memory stores and remove any Gemini API key by default."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()
    yield
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()


@pytest.fixture
def collector() -> EventCollector:
    """Fresh EventCollector with no storage backend."""
    return EventCollector()


@pytest.fixture
def client() -> TestClient:
    """Synchronous FastAPI test client."""
    return TestClient(app)


# ===========================================================================
# 1. capture_step — generic per-step context manager
# ===========================================================================


class TestCaptureStep:
    """Verify capture_step records SUCCESS / FAILURE correctly."""

    async def test_success_records_success_event(
        self, collector: EventCollector
    ) -> None:
        """Body completes cleanly -> SUCCESS event recorded in storage."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_step(
            EventType.RETRIEVAL,
            input_data={"query": "test"},
            metadata={"step": "retrieval"},
        ):
            pass  # clean exit

        await col.end_trace()

        assert backend.save.call_count == 1
        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.SUCCESS
        assert saved.event_type == EventType.RETRIEVAL

    async def test_failure_records_failure_event_and_reraises(
        self, collector: EventCollector
    ) -> None:
        """Exception in body -> FAILURE event recorded, exception re-raised."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        with pytest.raises(ValueError, match="retrieval broke"):
            async with col.capture_step(
                EventType.RETRIEVAL,
                input_data={"query": "test"},
                metadata={"step": "retrieval"},
            ):
                raise ValueError("retrieval broke")

        await col.end_trace()

        assert backend.save.call_count == 1
        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.FAILURE
        assert saved.event_type == EventType.RETRIEVAL
        assert saved.metadata is not None
        assert saved.metadata["exception_type"] == "ValueError"
        assert saved.metadata["exception_message"] == "retrieval broke"

    async def test_failure_event_has_non_negative_latency(self) -> None:
        """FAILURE event latency_ms must be non-negative."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        with pytest.raises(RuntimeError):
            async with col.capture_step(EventType.MEMORY_WRITE):
                raise RuntimeError("write failed")

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.latency_ms is not None
        assert saved.latency_ms >= 0.0

    async def test_success_event_carries_metadata(self) -> None:
        """SUCCESS event must carry the metadata passed to capture_step."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_step(
            EventType.MEMORY_WRITE,
            metadata={"step": "memory-write", "extra": "value"},
        ):
            pass

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.SUCCESS
        assert saved.metadata is not None
        assert saved.metadata["step"] == "memory-write"
        assert saved.metadata["extra"] == "value"

    async def test_step_accepts_string_event_type(self) -> None:
        """capture_step must accept a string event_type value."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_step("retrieval"):
            pass

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.event_type == EventType.RETRIEVAL


# ===========================================================================
# 2. capture_llm_call
# ===========================================================================


class TestCaptureLlmCall:
    """Verify capture_llm_call records events correctly."""

    async def test_success_records_llm_call_with_response(self) -> None:
        """SUCCESS path: response stored in output_data."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_llm_call(
            prompt="What is X?",
            model="gemini-2.5-flash",
            metadata={"step": "synthesis"},
        ) as result:
            result[0] = "X is a variable."

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.SUCCESS
        assert saved.event_type == EventType.LLM_CALL
        assert saved.input_data == {"prompt": "What is X?", "model": "gemini-2.5-flash"}
        assert saved.output_data == {"response": "X is a variable."}
        assert saved.metadata is not None
        assert saved.metadata["model"] == "gemini-2.5-flash"
        assert saved.metadata["step"] == "synthesis"

    async def test_failure_records_failure_event_with_exception_details(self) -> None:
        """FAILURE path: exception_type + exception_message in metadata."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        with pytest.raises(ConnectionError, match="quota exceeded"):
            async with col.capture_llm_call(
                prompt="Generate something",
                model="gemini-2.5-flash",
            ) as result:
                raise ConnectionError("quota exceeded")

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.FAILURE
        assert saved.event_type == EventType.LLM_CALL
        assert saved.output_data is None
        assert saved.metadata is not None
        assert saved.metadata["exception_type"] == "ConnectionError"
        assert saved.metadata["exception_message"] == "quota exceeded"

    async def test_failure_reraises_original_exception(self) -> None:
        """The original exception object must propagate out unchanged."""
        col = EventCollector()
        await col.start_trace(agent_id="agent-1")

        sentinel = RuntimeError("sentinel error")
        with pytest.raises(RuntimeError) as exc_info:
            async with col.capture_llm_call(prompt="test") as result:
                raise sentinel

        await col.end_trace()
        assert exc_info.value is sentinel

    async def test_result_list_is_empty_string_by_default(self) -> None:
        """The yielded result list must have a single empty-string element."""
        col = EventCollector()
        await col.start_trace(agent_id="agent-1")
        captured: list[list[str]] = []

        async with col.capture_llm_call(prompt="test") as result:
            captured.append(list(result))  # copy before mutation

        await col.end_trace()
        assert captured[0] == [""]


# ===========================================================================
# 3. capture_tool_call
# ===========================================================================


class TestCaptureToolCall:
    """Verify capture_tool_call records events correctly."""

    async def test_success_records_tool_call_with_output(self) -> None:
        """SUCCESS path: output stored in output_data."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_tool_call(
            "web_search",
            input_data={"query": "AI agents"},
            metadata={"step": "search"},
        ) as result:
            result[0] = {"hits": ["paper-1", "paper-2"]}

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.SUCCESS
        assert saved.event_type == EventType.TOOL_CALL
        assert saved.output_data == {"hits": ["paper-1", "paper-2"]}
        assert saved.metadata is not None
        assert saved.metadata["tool"] == "web_search"
        assert saved.metadata["step"] == "search"

    async def test_failure_records_failure_event_with_exception_details(self) -> None:
        """FAILURE path: exception_type + exception_message in metadata."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        with pytest.raises(TimeoutError, match="search timed out"):
            async with col.capture_tool_call(
                "web_search",
                input_data={"query": "test"},
            ) as result:
                raise TimeoutError("search timed out")

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.FAILURE
        assert saved.event_type == EventType.TOOL_CALL
        assert saved.metadata is not None
        assert saved.metadata["exception_type"] == "TimeoutError"
        assert saved.metadata["exception_message"] == "search timed out"

    async def test_non_dict_output_is_wrapped(self) -> None:
        """A non-dict result must be wrapped in {\"output\": value}."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_tool_call("tool") as result:
            result[0] = ["item1", "item2"]

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.output_data == {"output": ["item1", "item2"]}


# ===========================================================================
# 4. Retry linkage
# ===========================================================================


class TestRetryLinkage:
    """Verify retry_of + attempt are stored in event metadata."""

    async def test_capture_step_retry_linkage_on_success(self) -> None:
        """retry_of and attempt must appear in a retried (successful) step's metadata."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        failed_event_id = uuid4()

        async with col.capture_step(
            EventType.RETRIEVAL,
            metadata={"step": "retrieval"},
            retry_of=failed_event_id,
            attempt=2,
        ):
            pass

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.SUCCESS
        assert saved.metadata is not None
        assert saved.metadata["retry_of"] == str(failed_event_id)
        assert saved.metadata["attempt"] == 2

    async def test_capture_llm_call_retry_linkage_on_failure(self) -> None:
        """On failure, retry_of + attempt must also appear in the FAILURE event."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        failed_event_id = uuid4()

        with pytest.raises(RuntimeError):
            async with col.capture_llm_call(
                prompt="retry prompt",
                retry_of=failed_event_id,
                attempt=2,
            ) as result:
                raise RuntimeError("still failing")

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        assert saved.status == EventStatus.FAILURE
        assert saved.metadata is not None
        assert saved.metadata["retry_of"] == str(failed_event_id)
        assert saved.metadata["attempt"] == 2

    async def test_no_retry_metadata_when_not_a_retry(self) -> None:
        """When retry_of is None, retry_of/attempt must NOT appear in metadata."""
        backend = AsyncMock()
        backend.save = AsyncMock()
        col = EventCollector(storage_backend=backend)
        await col.start_trace(agent_id="agent-1")

        async with col.capture_step(
            EventType.RETRIEVAL,
            metadata={"step": "retrieval"},
            # no retry_of, no attempt override
        ):
            pass

        await col.end_trace()

        saved: AgentEvent = backend.save.call_args[0][0]
        meta = saved.metadata or {}
        assert "retry_of" not in meta
        assert "attempt" not in meta


# ===========================================================================
# 5. Pipeline failure capture — integration via HTTP
# ===========================================================================


class TestPipelineFailureCapture:
    """
    Test that killing a single step (synthesis Gemini call) persists exactly one
    FAILURE event in the trace with the correct exception type and message,
    while all previously completed steps keep their real statuses.
    """

    def test_gemini_failure_persists_failure_trace(self, client: TestClient) -> None:
        """
        When Gemini raises during synthesis, the stored trace must contain
        exactly one FAILURE event carrying exception_type + exception_message.
        """
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError(
            "Gemini quota exceeded"
        )

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
                    response = client.post(
                        "/api/run",
                        json={"query": "trigger gemini failure"},
                    )

        # HTTP layer must return 502.
        assert response.status_code == 502
        assert "Agent execution failed" in response.json()["detail"]

        # The trace must have been persisted as FAILURE.
        assert len(trace_store._traces) == 1
        persisted_trace = next(iter(trace_store._traces.values()))
        assert persisted_trace.status == EventStatus.FAILURE

        # Exactly one FAILURE event must exist in the trace.
        failure_events = [
            e for e in persisted_trace.events
            if e.status == EventStatus.FAILURE
        ]
        assert len(failure_events) == 1, (
            f"Expected exactly 1 FAILURE event, got {len(failure_events)}"
        )

        # The FAILURE event must carry exception_type + exception_message.
        failure_event = failure_events[0]
        assert failure_event.metadata is not None
        assert failure_event.metadata.get("exception_type") == "RuntimeError"
        assert "Gemini quota exceeded" in (
            failure_event.metadata.get("exception_message", "")
        )

    def test_gemini_failure_event_is_llm_call_type(
        self, client: TestClient
    ) -> None:
        """
        The FAILURE event created by capture_llm_call must have type LLM_CALL
        (not ERROR) because the wrapper uses the step's own event_type.
        """
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("crash")

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
                    client.post("/api/run", json={"query": "test"})

        persisted_trace = next(iter(trace_store._traces.values()))
        failure_events = [
            e for e in persisted_trace.events
            if e.status == EventStatus.FAILURE
        ]
        assert len(failure_events) == 1
        # The FAILURE must be on an LLM_CALL event (from capture_llm_call).
        assert failure_events[0].event_type == EventType.LLM_CALL

    def test_offline_run_has_no_failure_events(self, client: TestClient) -> None:
        """Successful offline run (no Gemini key) must produce zero FAILURE events."""
        response = client.post("/api/run", json={"query": "offline query"})
        assert response.status_code == 200

        from uuid import UUID
        trace_id = UUID(response.json()["trace_id"])
        persisted_trace = trace_store._traces[trace_id]

        failure_events = [
            e for e in persisted_trace.events
            if e.status == EventStatus.FAILURE
        ]
        assert len(failure_events) == 0
