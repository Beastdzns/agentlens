"""
Unit tests for AgentLens event models.

Tests cover:
- Event creation and validation
- Trace management and event hierarchies
- Serialization and deserialization
- Field validation and constraints
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    Trace,
    create_event,
    create_trace,
)

# ============================================================================
# Tests for EventType Enum
# ============================================================================


class TestEventType:
    """Test EventType enum."""

    def test_event_type_values(self) -> None:
        """Test that all expected event types exist."""
        expected_types = {
            "LLM_CALL",
            "TOOL_CALL",
            "RETRIEVAL",
            "MEMORY_READ",
            "MEMORY_WRITE",
            "SUBAGENT_START",
            "SUBAGENT_END",
            "DECISION",
            "ERROR",
            "RETRY",
            "FINAL_RESPONSE",
        }
        actual_types = {e.name for e in EventType}
        assert expected_types == actual_types

    def test_event_type_string_values(self) -> None:
        """Test string representation of event types."""
        assert EventType.LLM_CALL.value == "llm_call"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.ERROR.value == "error"


class TestEventStatus:
    """Test EventStatus enum."""

    def test_event_status_values(self) -> None:
        """Test that all expected statuses exist."""
        expected_statuses = {"PENDING", "SUCCESS", "FAILURE", "PARTIAL"}
        actual_statuses = {s.name for s in EventStatus}
        assert expected_statuses == actual_statuses

    def test_event_status_string_values(self) -> None:
        """Test string representation of statuses."""
        assert EventStatus.SUCCESS.value == "success"
        assert EventStatus.FAILURE.value == "failure"


class TestObservationLevel:
    """Test ObservationLevel enum."""

    def test_observation_level_values(self) -> None:
        """Test that all expected observation levels exist."""
        expected_levels = {"FULL", "COMPRESSED", "METADATA_ONLY", "DROP"}
        actual_levels = {level.name for level in ObservationLevel}
        assert expected_levels == actual_levels


# ============================================================================
# Tests for AgentEvent Model
# ============================================================================


class TestAgentEventCreation:
    """Test AgentEvent creation and field assignment."""

    def test_create_event_with_required_fields(self) -> None:
        """Test creating an event with only required fields."""
        trace_id = uuid4()
        event = AgentEvent(
            trace_id=trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )

        assert event.trace_id == trace_id
        assert event.agent_id == "agent-1"
        assert event.event_type == EventType.LLM_CALL
        assert event.status == EventStatus.SUCCESS
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.timestamp, datetime)

    def test_event_auto_generates_id_and_timestamp(self) -> None:
        """Test that event_id and timestamp are auto-generated."""
        event1 = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )
        event2 = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )

        # IDs should be unique
        assert event1.event_id != event2.event_id
        # Timestamps should be close but likely different
        assert isinstance(event1.timestamp, datetime)
        assert isinstance(event2.timestamp, datetime)

    def test_create_event_with_all_fields(self) -> None:
        """Test creating an event with all optional fields."""
        trace_id = uuid4()
        parent_id = uuid4()
        input_data = {"prompt": "test"}
        output_data = {"response": "answer"}
        metadata = {"key": "value"}

        event = AgentEvent(
            trace_id=trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data=input_data,
            output_data=output_data,
            latency_ms=1234.5,
            parent_id=parent_id,
            metadata=metadata,
            importance_score=0.8,
            observation_level=ObservationLevel.FULL,
        )

        assert event.input_data == input_data
        assert event.output_data == output_data
        assert event.latency_ms == 1234.5
        assert event.parent_id == parent_id
        assert event.metadata == metadata
        assert event.importance_score == 0.8
        assert event.observation_level == ObservationLevel.FULL

    def test_event_default_observation_level(self) -> None:
        """Test that observation_level defaults to FULL."""
        event = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )
        assert event.observation_level == ObservationLevel.FULL

    def test_event_timestamp_is_utc(self) -> None:
        """Test that auto-generated timestamp is in UTC."""
        event = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )
        # Timestamp should have UTC timezone
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc


class TestAgentEventValidation:
    """Test validation of AgentEvent fields."""

    def test_latency_must_be_non_negative(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEvent(
                trace_id=uuid4(),
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                latency_ms=-100.0,
            )
        assert "latency_ms" in str(exc_info.value).lower()

    def test_importance_score_must_be_between_0_and_1(self) -> None:
        """Test that importance_score is constrained to [0, 1]."""
        # Score too high
        with pytest.raises(ValidationError):
            AgentEvent(
                trace_id=uuid4(),
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                importance_score=1.5,
            )

        # Score too low
        with pytest.raises(ValidationError):
            AgentEvent(
                trace_id=uuid4(),
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                importance_score=-0.1,
            )

    def test_importance_score_accepts_valid_range(self) -> None:
        """Test that importance_score accepts values in [0, 1]."""
        for score in [0.0, 0.5, 1.0]:
            event = AgentEvent(
                trace_id=uuid4(),
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                importance_score=score,
            )
            assert event.importance_score == score

    def test_event_is_immutable(self) -> None:
        """Test that events are frozen (immutable)."""
        event = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )

        with pytest.raises(ValidationError):  # Pydantic frozen models raise ValidationError
            event.agent_id = "agent-2"  # type: ignore


class TestAgentEventSerialization:
    """Test serialization and deserialization of AgentEvent."""

    def test_to_dict_converts_to_json_serializable(self) -> None:
        """Test that to_dict produces JSON-serializable output."""
        event = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data={"key": "value"},
            metadata={"user": "alice"},
        )

        event_dict = event.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(event_dict)
        assert json_str

        # UUIDs should be strings
        assert isinstance(event_dict["event_id"], str)
        assert isinstance(event_dict["trace_id"], str)

        # Datetime should be string
        assert isinstance(event_dict["timestamp"], str)

    def test_from_dict_reconstructs_event(self) -> None:
        """Test that from_dict reconstructs event from dictionary."""
        trace_id = uuid4()
        original = AgentEvent(
            trace_id=trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data={"prompt": "test"},
            latency_ms=500.0,
        )

        event_dict = original.to_dict()
        reconstructed = AgentEvent.from_dict(event_dict)

        # Should match original
        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.agent_id == original.agent_id
        assert reconstructed.event_type == original.event_type
        assert reconstructed.status == original.status
        assert reconstructed.input_data == original.input_data
        assert reconstructed.latency_ms == original.latency_ms

    def test_round_trip_serialization(self) -> None:
        """Test full round-trip: event -> dict -> JSON -> dict -> event."""
        original = AgentEvent(
            trace_id=uuid4(),
            agent_id="agent-1",
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            input_data={"tool": "search", "query": "ai"},
            output_data={"error": "rate limited"},
            latency_ms=100.0,
            metadata={"retry_count": 2},
            observation_level=ObservationLevel.COMPRESSED,
        )

        # Round-trip
        dict1 = original.to_dict()
        json_str = json.dumps(dict1)
        dict2 = json.loads(json_str)
        reconstructed = AgentEvent.from_dict(dict2)

        # Critical fields should match
        assert reconstructed.event_id == original.event_id
        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.agent_id == original.agent_id
        assert reconstructed.event_type == original.event_type
        assert reconstructed.status == original.status
        assert reconstructed.input_data == original.input_data
        assert reconstructed.output_data == original.output_data


# ============================================================================
# Tests for Trace Model
# ============================================================================


class TestTraceCreation:
    """Test Trace creation and field assignment."""

    def test_create_trace_with_required_fields(self) -> None:
        """Test creating a trace with only required fields."""
        trace = Trace(agent_id="agent-1")

        assert trace.agent_id == "agent-1"
        assert isinstance(trace.trace_id, UUID)
        assert isinstance(trace.start_time, datetime)
        assert trace.status == EventStatus.PENDING
        assert trace.event_count == 0
        assert trace.total_latency_ms == 0.0
        assert trace.events == []

    def test_trace_auto_generates_id_and_start_time(self) -> None:
        """Test that trace_id and start_time are auto-generated."""
        trace1 = Trace(agent_id="agent-1")
        trace2 = Trace(agent_id="agent-1")

        assert trace1.trace_id != trace2.trace_id
        assert trace1.start_time != trace2.start_time or trace1.trace_id != trace2.trace_id

    def test_create_trace_with_all_fields(self) -> None:
        """Test creating trace with all optional fields."""
        metadata = {"user_id": "user-123"}
        trace = Trace(
            agent_id="agent-1",
            status=EventStatus.SUCCESS,
            metadata=metadata,
        )

        assert trace.status == EventStatus.SUCCESS
        assert trace.metadata == metadata

    def test_trace_start_time_is_utc(self) -> None:
        """Test that trace start_time is in UTC."""
        trace = Trace(agent_id="agent-1")
        assert trace.start_time.tzinfo is not None
        assert trace.start_time.tzinfo == timezone.utc


class TestTraceEventManagement:
    """Test Trace methods for managing events."""

    def test_add_event_to_trace(self) -> None:
        """Test adding an event to a trace."""
        trace = Trace(agent_id="agent-1")
        event = AgentEvent(
            trace_id=trace.trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            latency_ms=100.0,
        )

        trace.add_event(event)

        assert len(trace.events) == 1
        assert trace.event_count == 1
        assert trace.total_latency_ms == 100.0

    def test_add_multiple_events(self) -> None:
        """Test adding multiple events accumulates statistics."""
        trace = Trace(agent_id="agent-1")

        events = [
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                latency_ms=100.0,
            ),
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.TOOL_CALL,
                status=EventStatus.SUCCESS,
                latency_ms=50.0,
            ),
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                latency_ms=75.0,
            ),
        ]

        for event in events:
            trace.add_event(event)

        assert trace.event_count == 3
        assert trace.total_latency_ms == 225.0

    def test_add_event_with_mismatched_trace_id_fails(self) -> None:
        """Test that adding event with wrong trace_id raises error."""
        trace = Trace(agent_id="agent-1")
        wrong_trace_id = uuid4()

        event = AgentEvent(
            trace_id=wrong_trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
        )

        with pytest.raises(ValueError, match="trace_id"):
            trace.add_event(event)

    def test_add_event_without_latency(self) -> None:
        """Test that adding event without latency doesn't break stats."""
        trace = Trace(agent_id="agent-1")
        event = AgentEvent(
            trace_id=trace.trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            latency_ms=None,
        )

        trace.add_event(event)
        assert trace.event_count == 1
        assert trace.total_latency_ms == 0.0

    def test_get_events_by_type(self) -> None:
        """Test filtering events by type."""
        trace = Trace(agent_id="agent-1")

        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
            )
        )
        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.TOOL_CALL,
                status=EventStatus.SUCCESS,
            )
        )
        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
            )
        )

        llm_calls = trace.get_events_by_type(EventType.LLM_CALL)
        assert len(llm_calls) == 2
        assert all(e.event_type == EventType.LLM_CALL for e in llm_calls)

        tool_calls = trace.get_events_by_type(EventType.TOOL_CALL)
        assert len(tool_calls) == 1

    def test_get_events_by_status(self) -> None:
        """Test filtering events by status."""
        trace = Trace(agent_id="agent-1")

        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
            )
        )
        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.TOOL_CALL,
                status=EventStatus.FAILURE,
            )
        )
        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.RETRY,
                status=EventStatus.SUCCESS,
            )
        )

        successes = trace.get_events_by_status(EventStatus.SUCCESS)
        assert len(successes) == 2
        assert all(e.status == EventStatus.SUCCESS for e in successes)

        failures = trace.get_events_by_status(EventStatus.FAILURE)
        assert len(failures) == 1


class TestTraceFinalization:
    """Test Trace finalization."""

    def test_finalize_sets_end_time(self) -> None:
        """Test that finalize sets end_time."""
        trace = Trace(agent_id="agent-1")
        assert trace.end_time is None

        trace.finalize()

        assert trace.end_time is not None
        assert isinstance(trace.end_time, datetime)

    def test_finalize_can_update_status(self) -> None:
        """Test that finalize can update status."""
        trace = Trace(agent_id="agent-1", status=EventStatus.PENDING)
        assert trace.status == EventStatus.PENDING

        trace.finalize(EventStatus.SUCCESS)

        assert trace.status == EventStatus.SUCCESS

    def test_finalize_preserves_status_if_not_provided(self) -> None:
        """Test that finalize preserves status if not provided."""
        trace = Trace(agent_id="agent-1", status=EventStatus.SUCCESS)

        trace.finalize()

        assert trace.status == EventStatus.SUCCESS


class TestTraceSerialization:
    """Test serialization of Trace."""

    def test_trace_to_dict(self) -> None:
        """Test converting trace to dict."""
        trace = Trace(agent_id="agent-1", status=EventStatus.SUCCESS)
        trace.add_event(
            AgentEvent(
                trace_id=trace.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
            )
        )

        trace_dict = trace.to_dict()

        assert isinstance(trace_dict, dict)
        assert trace_dict["agent_id"] == "agent-1"
        assert trace_dict["event_count"] == 1
        assert isinstance(trace_dict["events"], list)

    def test_trace_from_dict(self) -> None:
        """Test reconstructing trace from dict."""
        original = Trace(agent_id="agent-1", status=EventStatus.SUCCESS)
        original.add_event(
            AgentEvent(
                trace_id=original.trace_id,
                agent_id="agent-1",
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
            )
        )

        trace_dict = original.to_dict()
        reconstructed = Trace.from_dict(trace_dict)

        assert reconstructed.agent_id == original.agent_id
        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.event_count == original.event_count
        assert len(reconstructed.events) == 1


# ============================================================================
# Tests for Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Test create_event and create_trace factory functions."""

    def test_create_event_factory(self) -> None:
        """Test create_event factory function."""
        trace_id = uuid4()
        event = create_event(
            trace_id=trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data={"prompt": "test"},
            output_data={"response": "answer"},
            latency_ms=1000.0,
        )

        assert isinstance(event, AgentEvent)
        assert event.trace_id == trace_id
        assert event.agent_id == "agent-1"
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.timestamp, datetime)

    def test_create_event_with_minimal_args(self) -> None:
        """Test create_event with only required arguments."""
        trace_id = uuid4()
        event = create_event(
            trace_id=trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
        )

        assert event.status == EventStatus.SUCCESS  # Default
        assert event.observation_level == ObservationLevel.FULL  # Default

    def test_create_trace_factory(self) -> None:
        """Test create_trace factory function."""
        trace = create_trace(agent_id="agent-1")

        assert isinstance(trace, Trace)
        assert trace.agent_id == "agent-1"
        assert isinstance(trace.trace_id, UUID)
        assert isinstance(trace.start_time, datetime)

    def test_create_trace_with_metadata(self) -> None:
        """Test create_trace with metadata."""
        metadata = {"user": "alice", "session": "sess-123"}
        trace = create_trace(agent_id="agent-1", metadata=metadata)

        assert trace.metadata == metadata


# ============================================================================
# Integration Tests
# ============================================================================


class TestEventTraceIntegration:
    """Integration tests for events and traces working together."""

    def test_build_trace_with_hierarchy(self) -> None:
        """Test building a trace with parent-child event hierarchy."""
        trace = create_trace(agent_id="agent-1")

        # Parent event
        parent_event = create_event(
            trace_id=trace.trace_id,
            agent_id="agent-1",
            event_type=EventType.DECISION,
            status=EventStatus.SUCCESS,
            input_data={"question": "what to do?"},
            latency_ms=100.0,
        )
        trace.add_event(parent_event)

        # Child events (nested under parent)
        child1 = create_event(
            trace_id=trace.trace_id,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            parent_id=parent_event.event_id,
            latency_ms=50.0,
        )
        trace.add_event(child1)

        child2 = create_event(
            trace_id=trace.trace_id,
            agent_id="agent-1",
            event_type=EventType.TOOL_CALL,
            status=EventStatus.SUCCESS,
            parent_id=parent_event.event_id,
            latency_ms=30.0,
        )
        trace.add_event(child2)

        # Verify structure
        assert trace.event_count == 3
        assert trace.total_latency_ms == 180.0
        assert child1.parent_id == parent_event.event_id
        assert child2.parent_id == parent_event.event_id

    def test_failure_diagnosis_scenario(self) -> None:
        """Test a realistic scenario with failure and recovery."""
        trace = create_trace(agent_id="research-bot")

        # Initial search
        search_event = create_event(
            trace_id=trace.trace_id,
            agent_id="research-bot",
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            input_data={"query": "ai agents 2026"},
            output_data={"error": "no results"},
            latency_ms=100.0,
            metadata={"tool": "search"},
        )
        trace.add_event(search_event)

        # Retry with modified query
        retry_event = create_event(
            trace_id=trace.trace_id,
            agent_id="research-bot",
            event_type=EventType.RETRY,
            status=EventStatus.SUCCESS,
            parent_id=search_event.event_id,
            input_data={"query": "autonomous agents"},
            latency_ms=50.0,
        )
        trace.add_event(retry_event)

        # LLM processes results
        llm_event = create_event(
            trace_id=trace.trace_id,
            agent_id="research-bot",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data={"results": ["..."]},
            output_data={"summary": "..."},
            latency_ms=200.0,
        )
        trace.add_event(llm_event)

        # Final response
        final_event = create_event(
            trace_id=trace.trace_id,
            agent_id="research-bot",
            event_type=EventType.FINAL_RESPONSE,
            status=EventStatus.SUCCESS,
            output_data={"answer": "..."},
            latency_ms=0.0,
        )
        trace.add_event(final_event)

        trace.finalize(EventStatus.SUCCESS)

        # Verify the trace structure for diagnosis
        failures = trace.get_events_by_status(EventStatus.FAILURE)
        assert len(failures) == 1
        assert failures[0].event_type == EventType.TOOL_CALL

        # Retries should be child of failures
        retries = trace.get_events_by_type(EventType.RETRY)
        assert len(retries) == 1
        assert retries[0].parent_id == failures[0].event_id

        # Overall trace should be success
        assert trace.status == EventStatus.SUCCESS
        assert trace.event_count == 4