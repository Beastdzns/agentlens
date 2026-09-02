"""
Shared pytest fixtures for AgentLens test suite.

This conftest is loaded automatically by pytest for all tests under ``tests/``.

Fixtures provided
-----------------
- ``sample_event``      — factory fixture that returns a callable creating AgentEvent
- ``sample_trace``      — factory fixture that returns a callable creating Trace
- ``memory_event_store`` — fresh InMemoryEventStore per test
- ``memory_trace_store`` — fresh InMemoryTraceStore per test

Usage in test files::

    def test_something(sample_event, memory_event_store):
        event = sample_event()
        await memory_event_store.save_event(event)
        ...
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
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


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_event():
    """
    Factory fixture that returns a callable for creating :class:`AgentEvent` instances.

    The factory accepts keyword overrides so each test can customise only the
    fields it cares about, keeping boilerplate minimal.

    Usage::

        def test_foo(sample_event):
            event = sample_event(event_type=EventType.TOOL_CALL, latency_ms=55.0)
            assert event.event_type == EventType.TOOL_CALL
    """

    def _factory(
        trace_id: Optional[UUID] = None,
        agent_id: str = "test-agent",
        event_type: EventType = EventType.LLM_CALL,
        status: EventStatus = EventStatus.SUCCESS,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        latency_ms: Optional[float] = 100.0,
        parent_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        importance_score: float = 0.5,
        observation_level: ObservationLevel = ObservationLevel.FULL,
    ) -> AgentEvent:
        return create_event(
            trace_id=trace_id or uuid4(),
            agent_id=agent_id,
            event_type=event_type,
            status=status,
            input_data=input_data or {"key": "value"},
            output_data=output_data or {"result": "ok"},
            latency_ms=latency_ms,
            parent_id=parent_id,
            metadata=metadata,
            importance_score=importance_score,
            observation_level=observation_level,
        )

    return _factory


@pytest.fixture
def sample_trace():
    """
    Factory fixture that returns a callable for creating :class:`Trace` instances.

    Usage::

        def test_bar(sample_trace):
            trace = sample_trace(agent_id="research-bot")
            assert trace.agent_id == "research-bot"
    """

    def _factory(
        agent_id: str = "test-agent",
        status: EventStatus = EventStatus.PENDING,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Trace:
        return create_trace(agent_id=agent_id, status=status, metadata=metadata)

    return _factory


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_event_store() -> InMemoryEventStore:
    """
    Return a **fresh** :class:`InMemoryEventStore` for each test.

    Isolation: every test function gets its own store with zero events so
    state never leaks between tests.

    Usage::

        async def test_save(memory_event_store, sample_event):
            event = sample_event()
            await memory_event_store.save_event(event)
            found = await memory_event_store.get_event(event.event_id)
            assert found == event
    """
    return InMemoryEventStore()


@pytest.fixture
def memory_trace_store() -> InMemoryTraceStore:
    """
    Return a **fresh** :class:`InMemoryTraceStore` for each test.

    Isolation: every test function gets its own store with zero traces.

    Usage::

        async def test_save_trace(memory_trace_store, sample_trace):
            trace = sample_trace()
            await memory_trace_store.save_trace(trace)
            loaded = await memory_trace_store.get_trace(trace.trace_id)
            assert loaded.trace_id == trace.trace_id
    """
    return InMemoryTraceStore()
