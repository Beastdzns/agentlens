"""
Unit tests for the sdk_collector package.

Coverage
--------
- EventCreation         — auto-ID generation, required fields, AgentEvent validation
- TraceContextManagement — start_trace, end_trace, get_current_trace_id/parent_id
- ParentChildHierarchy  — push_parent / pop_parent, nested event parent_id propagation
- AsyncContextIsolation — two concurrent coroutines maintain independent contexts
- TracedDecorator       — async def wrapping, exception propagation, context cleanup
- EventNormalizerFromDict — field aliasing, timestamp formats, missing-field defaults

Test conventions
----------------
- asyncio_mode = "auto" is set in pyproject.toml so ``async def`` test methods
  run automatically without a ``@pytest.mark.asyncio`` decorator.
- Mocks are used for the storage_backend so tests never touch disk or network.
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agentlens.events import AgentEvent, EventStatus, EventType
from agentlens.sdk_collector import ContextManager, EventCollector, EventNormalizer


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def collector() -> EventCollector:
    """Return a fresh EventCollector with no storage backend."""
    return EventCollector()


@pytest.fixture()
def collector_with_storage() -> tuple[EventCollector, AsyncMock]:
    """Return a collector wired to a mock async storage backend."""
    backend = AsyncMock()
    backend.save = AsyncMock()
    return EventCollector(storage_backend=backend), backend


@pytest.fixture()
def normalizer() -> EventNormalizer:
    """Return a fresh EventNormalizer."""
    return EventNormalizer()


# ===========================================================================
# 1. Event Creation
# ===========================================================================


class TestEventCreation:
    """Verify that record_event auto-generates IDs and timestamps."""

    async def test_record_event_returns_agent_event(self, collector: EventCollector) -> None:
        """record_event must return a validated AgentEvent."""
        await collector.start_trace(agent_id="agent-1")
        event = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "hello"},
            output_data={"response": "world"},
            status=EventStatus.SUCCESS,
            latency_ms=55.0,
            metadata={"model": "gemini-2.0-flash"},
        )
        await collector.end_trace()

        assert isinstance(event, AgentEvent)
        assert isinstance(event.event_id, UUID)
        assert event.event_type == EventType.LLM_CALL
        assert event.status == EventStatus.SUCCESS
        assert event.latency_ms == 55.0

    async def test_event_id_is_auto_generated_and_unique(
        self, collector: EventCollector
    ) -> None:
        """Each call to record_event must produce a distinct event_id."""
        await collector.start_trace(agent_id="agent-1")
        e1 = await collector.record_event(event_type=EventType.LLM_CALL)
        e2 = await collector.record_event(event_type=EventType.TOOL_CALL)
        await collector.end_trace()

        assert e1.event_id != e2.event_id

    async def test_trace_id_is_auto_set_from_context(
        self, collector: EventCollector
    ) -> None:
        """The recorded event must carry the trace_id set by start_trace."""
        fixed_trace_id = uuid4()
        await collector.start_trace(agent_id="agent-1", trace_id=fixed_trace_id)
        event = await collector.record_event(event_type=EventType.LLM_CALL)
        await collector.end_trace()

        assert event.trace_id == fixed_trace_id

    async def test_record_event_timestamp_is_utc(
        self, collector: EventCollector
    ) -> None:
        """Auto-generated timestamp must be UTC-aware."""
        await collector.start_trace(agent_id="agent-1")
        event = await collector.record_event(event_type=EventType.LLM_CALL)
        await collector.end_trace()

        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc

    async def test_record_event_without_active_trace_raises(
        self, collector: EventCollector
    ) -> None:
        """record_event must raise RuntimeError when no trace is active."""
        with pytest.raises(RuntimeError, match="No active trace"):
            await collector.record_event(event_type=EventType.LLM_CALL)

    async def test_record_event_accepts_string_event_type(
        self, collector: EventCollector
    ) -> None:
        """record_event must accept string values for event_type."""
        await collector.start_trace(agent_id="agent-1")
        event = await collector.record_event(event_type="tool_call")
        await collector.end_trace()

        assert event.event_type == EventType.TOOL_CALL

    async def test_record_event_accepts_string_status(
        self, collector: EventCollector
    ) -> None:
        """record_event must accept string values for status."""
        await collector.start_trace(agent_id="agent-1")
        event = await collector.record_event(
            event_type=EventType.LLM_CALL, status="failure"
        )
        await collector.end_trace()

        assert event.status == EventStatus.FAILURE


# ===========================================================================
# 2. Trace Context Management
# ===========================================================================


class TestTraceContextManagement:
    """Verify start_trace, end_trace, and the context accessor helpers."""

    async def test_start_trace_sets_trace_id(self, collector: EventCollector) -> None:
        """start_trace must populate get_current_trace_id."""
        assert collector.get_current_trace_id() is None
        tid = await collector.start_trace(agent_id="agent-1")
        assert collector.get_current_trace_id() == tid
        await collector.end_trace()

    async def test_start_trace_with_explicit_trace_id(
        self, collector: EventCollector
    ) -> None:
        """start_trace should use the caller-supplied trace_id."""
        fixed_id = uuid4()
        returned = await collector.start_trace(agent_id="agent-1", trace_id=fixed_id)
        assert returned == fixed_id
        assert collector.get_current_trace_id() == fixed_id
        await collector.end_trace()

    async def test_end_trace_clears_trace_id(self, collector: EventCollector) -> None:
        """end_trace must reset get_current_trace_id to None."""
        await collector.start_trace(agent_id="agent-1")
        await collector.end_trace()
        assert collector.get_current_trace_id() is None

    async def test_end_trace_with_no_active_trace_is_safe(
        self, collector: EventCollector
    ) -> None:
        """Calling end_trace with no active trace must not raise."""
        await collector.end_trace()  # should not raise

    async def test_get_current_parent_id_default_is_none(
        self, collector: EventCollector
    ) -> None:
        """Before any push_parent, get_current_parent_id should return None."""
        await collector.start_trace(agent_id="agent-1")
        assert collector.get_current_parent_id() is None
        await collector.end_trace()


# ===========================================================================
# 3. Parent–Child Hierarchy
# ===========================================================================


class TestParentChildHierarchy:
    """Verify push_parent / pop_parent and parent propagation into events."""

    async def test_push_parent_sets_current_parent_id(
        self, collector: EventCollector
    ) -> None:
        """After push_parent, get_current_parent_id must return pushed UUID."""
        await collector.start_trace(agent_id="agent-1")
        parent_id = uuid4()
        collector._ctx.push_parent(parent_id)
        assert collector.get_current_parent_id() == parent_id
        await collector.end_trace()

    async def test_pop_parent_restores_previous_parent(
        self, collector: EventCollector
    ) -> None:
        """Popping the parent stack must expose the previous level."""
        await collector.start_trace(agent_id="agent-1")
        outer_id = uuid4()
        inner_id = uuid4()

        collector._ctx.push_parent(outer_id)
        collector._ctx.push_parent(inner_id)
        assert collector.get_current_parent_id() == inner_id

        popped = collector._ctx.pop_parent()
        assert popped == inner_id
        assert collector.get_current_parent_id() == outer_id

        collector._ctx.pop_parent()
        assert collector.get_current_parent_id() is None
        await collector.end_trace()

    async def test_recorded_event_inherits_parent_id(
        self, collector: EventCollector
    ) -> None:
        """Events recorded after push_parent must carry that parent_id."""
        await collector.start_trace(agent_id="agent-1")

        # Record the parent event.
        parent_event = await collector.record_event(event_type=EventType.DECISION)

        # Push it as the current parent.
        collector._ctx.push_parent(parent_event.event_id)

        # Child event must inherit the parent_id.
        child_event = await collector.record_event(event_type=EventType.LLM_CALL)
        assert child_event.parent_id == parent_event.event_id

        collector._ctx.pop_parent()
        await collector.end_trace()

    async def test_deeply_nested_hierarchy(self, collector: EventCollector) -> None:
        """Test three levels of parent nesting."""
        await collector.start_trace(agent_id="agent-1")

        lvl0 = await collector.record_event(event_type=EventType.DECISION)
        collector._ctx.push_parent(lvl0.event_id)

        lvl1 = await collector.record_event(event_type=EventType.LLM_CALL)
        assert lvl1.parent_id == lvl0.event_id
        collector._ctx.push_parent(lvl1.event_id)

        lvl2 = await collector.record_event(event_type=EventType.TOOL_CALL)
        assert lvl2.parent_id == lvl1.event_id

        collector._ctx.pop_parent()
        collector._ctx.pop_parent()
        await collector.end_trace()

    async def test_pop_on_empty_stack_returns_none(
        self, collector: EventCollector
    ) -> None:
        """pop_parent on an empty stack must return None safely."""
        result = collector._ctx.pop_parent()
        assert result is None


# ===========================================================================
# 4. Async Context Isolation (Concurrent Traces)
# ===========================================================================


class TestAsyncContextIsolation:
    """
    Verify that two concurrently running coroutines each maintain their own
    independent trace context — the core guarantee of contextvars.
    """

    async def test_two_concurrent_traces_are_independent(self) -> None:
        """
        Spawn two tasks that each start a trace with a different trace_id.
        When they both finish, the IDs they recorded must be distinct and
        must match their own start_trace call — not each other's.
        """
        c1 = EventCollector()
        c2 = EventCollector()

        results: dict[str, UUID | None] = {}

        async def run_trace_a() -> None:
            tid_a = uuid4()
            await c1.start_trace(agent_id="agent-a", trace_id=tid_a)
            # Yield to allow task-b to run.
            await asyncio.sleep(0)
            results["a"] = c1.get_current_trace_id()
            await c1.end_trace()

        async def run_trace_b() -> None:
            tid_b = uuid4()
            await c2.start_trace(agent_id="agent-b", trace_id=tid_b)
            await asyncio.sleep(0)
            results["b"] = c2.get_current_trace_id()
            await c2.end_trace()

        await asyncio.gather(run_trace_a(), run_trace_b())

        assert results["a"] is not None
        assert results["b"] is not None
        assert results["a"] != results["b"]

    async def test_parent_stack_isolation_across_tasks(self) -> None:
        """
        Child tasks must *copy* the parent stack at spawn time and not inherit
        mutations made by the parent *after* the child is spawned.
        """
        ctx = ContextManager()
        outer_id = uuid4()
        ctx.push_parent(outer_id)

        captured_from_child: list[UUID | None] = []

        async def child_task() -> None:
            # Immediately read — should see outer_id.
            captured_from_child.append(ctx.get_parent_id())
            # Push something locally — must not affect parent task.
            ctx.push_parent(uuid4())

        task = asyncio.create_task(child_task())
        # Push a new parent in the parent task *after* spawning the child.
        inner_id = uuid4()
        ctx.push_parent(inner_id)

        await task

        # Parent task should see both outer_id and inner_id on its stack.
        assert ctx.get_parent_id() == inner_id

        # Child saw the stack *as it was at spawn time* (outer_id only).
        assert captured_from_child[0] == outer_id


# ===========================================================================
# 5. @traced Decorator
# ===========================================================================


class TestTracedDecorator:
    """Verify the @collector.traced(...) decorator for async and sync functions."""

    async def test_traced_starts_and_ends_trace_for_async_func(
        self, collector: EventCollector
    ) -> None:
        """Trace must be active inside the decorated coroutine."""
        trace_id_inside: list[UUID | None] = []

        @collector.traced(agent_id="agent-1")
        async def my_coroutine() -> str:
            trace_id_inside.append(collector.get_current_trace_id())
            return "done"

        result = await my_coroutine()
        assert result == "done"
        assert trace_id_inside[0] is not None  # was active inside
        assert collector.get_current_trace_id() is None  # cleared after exit

    async def test_traced_cleans_up_on_exception(
        self, collector: EventCollector
    ) -> None:
        """end_trace must be called even when the decorated function raises."""
        @collector.traced(agent_id="agent-err")
        async def failing() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await failing()

        # Context must be cleared even after exception.
        assert collector.get_current_trace_id() is None

    async def test_traced_as_async_context_manager(
        self, collector: EventCollector
    ) -> None:
        """traced() must work as an async context manager (async with)."""
        async with collector.traced(agent_id="ctx-agent"):
            assert collector.get_current_trace_id() is not None

        assert collector.get_current_trace_id() is None

    async def test_traced_records_event_with_correct_trace_id(
        self, collector: EventCollector
    ) -> None:
        """Events recorded inside @traced must carry the auto-generated trace_id."""
        recorded_events: list[AgentEvent] = []

        @collector.traced(agent_id="pipeline")
        async def pipeline() -> None:
            event = await collector.record_event(event_type=EventType.LLM_CALL)
            recorded_events.append(event)

        await pipeline()

        assert len(recorded_events) == 1
        assert recorded_events[0].trace_id is not None
        assert recorded_events[0].agent_id == "pipeline"

    async def test_traced_propagates_return_value(
        self, collector: EventCollector
    ) -> None:
        """@traced must transparently return the wrapped function's return value."""
        @collector.traced(agent_id="agent-1")
        async def compute() -> int:
            return 42

        assert await compute() == 42


# ===========================================================================
# 6. Storage Backend Integration
# ===========================================================================


class TestStorageBackend:
    """Verify storage backend calls and fail-open behaviour."""

    async def test_storage_backend_is_called_on_record_event(
        self, collector_with_storage: tuple[EventCollector, AsyncMock]
    ) -> None:
        """save() must be called once per recorded event."""
        collector, backend = collector_with_storage
        await collector.start_trace(agent_id="agent-1")
        await collector.record_event(event_type=EventType.LLM_CALL)
        await collector.end_trace()

        backend.save.assert_called_once()
        saved_event: AgentEvent = backend.save.call_args[0][0]
        assert isinstance(saved_event, AgentEvent)

    async def test_storage_failure_does_not_propagate(self) -> None:
        """A crashing storage backend must not raise to the caller."""
        bad_backend = AsyncMock()
        bad_backend.save = AsyncMock(side_effect=OSError("disk full"))
        collector = EventCollector(storage_backend=bad_backend)

        await collector.start_trace(agent_id="agent-1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            event = await collector.record_event(event_type=EventType.LLM_CALL)

        await collector.end_trace()

        # Event is still returned despite storage failure.
        assert isinstance(event, AgentEvent)
        # A RuntimeWarning is emitted.
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# ===========================================================================
# 7. EventNormalizer — from_dict
# ===========================================================================


class TestEventNormalizerFromDict:
    """Verify EventNormalizer.from_dict handles all supported input shapes."""

    def test_minimal_dict_produces_valid_agent_event(
        self, normalizer: EventNormalizer
    ) -> None:
        """A dict with only agent_id and event_type is sufficient."""
        event = normalizer.from_dict(
            {"agent_id": "bot", "event_type": "llm_call"}
        )
        assert isinstance(event, AgentEvent)
        assert event.agent_id == "bot"
        assert event.event_type == EventType.LLM_CALL

    def test_field_alias_response_maps_to_output_data(
        self, normalizer: EventNormalizer
    ) -> None:
        """'response' key must be mapped to output_data."""
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "response": {"text": "hello"},
            }
        )
        assert event.output_data == {"text": "hello"}

    def test_field_alias_prompt_maps_to_input_data(
        self, normalizer: EventNormalizer
    ) -> None:
        """'prompt' key must be mapped to input_data (wrapped if scalar)."""
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "prompt": "What is 2+2?",
            }
        )
        assert event.input_data == {"value": "What is 2+2?"}

    def test_field_alias_prompt_dict_is_used_directly(
        self, normalizer: EventNormalizer
    ) -> None:
        """'prompt' that is already a dict must not be double-wrapped."""
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "prompt": {"messages": ["Hello"]},
            }
        )
        assert event.input_data == {"messages": ["Hello"]}

    def test_iso_timestamp_string_is_normalized_to_utc(
        self, normalizer: EventNormalizer
    ) -> None:
        """ISO string timestamps must be parsed and made UTC-aware."""
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "timestamp": "2026-01-15T10:30:00+05:30",
            }
        )
        assert event.timestamp.tzinfo == timezone.utc
        assert event.timestamp.hour == 5  # 10:30 IST → 05:00 UTC

    def test_unix_epoch_timestamp_is_normalized(
        self, normalizer: EventNormalizer
    ) -> None:
        """Unix epoch float timestamps must become UTC datetimes."""
        epoch = 1_700_000_000.0  # 2023-11-14 22:13:20 UTC
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "timestamp": epoch,
            }
        )
        expected = datetime.fromtimestamp(epoch, tz=timezone.utc)
        assert event.timestamp == expected

    def test_naive_datetime_assumed_utc(
        self, normalizer: EventNormalizer
    ) -> None:
        """Naïve datetime objects must be treated as UTC."""
        naive_dt = datetime(2026, 6, 1, 12, 0, 0)  # no tzinfo
        event = normalizer.from_dict(
            {
                "agent_id": "bot",
                "event_type": "llm_call",
                "timestamp": naive_dt,
            }
        )
        assert event.timestamp.tzinfo == timezone.utc
        assert event.timestamp.hour == 12

    def test_trace_id_string_is_coerced_to_uuid(
        self, normalizer: EventNormalizer
    ) -> None:
        """A trace_id given as a string must be coerced to UUID."""
        raw_id = "550e8400-e29b-41d4-a716-446655440000"
        event = normalizer.from_dict(
            {"agent_id": "bot", "event_type": "llm_call", "trace_id": raw_id}
        )
        assert isinstance(event.trace_id, UUID)
        assert str(event.trace_id) == raw_id

    def test_missing_trace_id_gets_auto_generated(
        self, normalizer: EventNormalizer
    ) -> None:
        """trace_id must be auto-generated when absent."""
        event = normalizer.from_dict({"agent_id": "bot", "event_type": "llm_call"})
        assert isinstance(event.trace_id, UUID)

    def test_unknown_event_type_defaults_to_llm_call_with_warning(
        self, normalizer: EventNormalizer
    ) -> None:
        """An unrecognised event_type string must default to LLM_CALL and warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            event = normalizer.from_dict(
                {"agent_id": "bot", "event_type": "nonexistent_type"}
            )
        assert event.event_type == EventType.LLM_CALL
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_missing_agent_id_defaults_to_unknown(
        self, normalizer: EventNormalizer
    ) -> None:
        """Missing agent_id must fall back to 'unknown'."""
        event = normalizer.from_dict({"event_type": "llm_call"})
        assert event.agent_id == "unknown"

    def test_from_dict_does_not_mutate_input(
        self, normalizer: EventNormalizer
    ) -> None:
        """from_dict must not modify the caller's dictionary."""
        original = {
            "agent_id": "bot",
            "event_type": "llm_call",
            "prompt": "hello",
        }
        copy_before = dict(original)
        normalizer.from_dict(original)
        assert original == copy_before

    def test_full_dict_round_trip(self, normalizer: EventNormalizer) -> None:
        """A fully-specified dict must produce a matching AgentEvent."""
        tid = uuid4()
        pid = uuid4()
        event = normalizer.from_dict(
            {
                "trace_id": str(tid),
                "parent_id": str(pid),
                "agent_id": "research-bot",
                "event_type": "tool_call",
                "status": "failure",
                "latency_ms": 99.9,
                "input_data": {"query": "ai"},
                "output_data": {"error": "timeout"},
                "metadata": {"tool": "search"},
            }
        )
        assert event.trace_id == tid
        assert event.parent_id == pid
        assert event.agent_id == "research-bot"
        assert event.event_type == EventType.TOOL_CALL
        assert event.status == EventStatus.FAILURE
        assert event.latency_ms == 99.9
        assert event.input_data == {"query": "ai"}
        assert event.output_data == {"error": "timeout"}

    def test_from_langchain_event_raises_not_implemented(
        self, normalizer: EventNormalizer
    ) -> None:
        """from_langchain_event stub must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            normalizer.from_langchain_event(object())

    def test_from_llamaindex_event_raises_not_implemented(
        self, normalizer: EventNormalizer
    ) -> None:
        """from_llamaindex_event stub must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            normalizer.from_llamaindex_event(object())
