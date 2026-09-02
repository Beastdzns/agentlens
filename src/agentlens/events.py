"""Event models and schemas for AgentLens."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    """Enumeration of supported agent event types."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"
    DECISION = "decision"
    ERROR = "error"
    RETRY = "retry"
    FINAL_RESPONSE = "final_response"


class EventStatus(str, Enum):
    """Status of an agent event."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class ObservationLevel(str, Enum):
    """Adaptive observation levels for events."""

    FULL = "full"
    COMPRESSED = "compressed"
    METADATA_ONLY = "metadata_only"
    DROP = "drop"


class AgentEvent(BaseModel):
    """Structured representation of a single agent operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    parent_id: Optional[UUID] = None
    agent_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    status: EventStatus = EventStatus.SUCCESS
    latency_ms: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    importance_score: float = 0.5
    observation_level: ObservationLevel = ObservationLevel.FULL

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("latency_ms must be non-negative")
        return value

    @field_validator("importance_score")
    @classmethod
    def validate_importance_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("importance_score must be between 0 and 1")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "trace_id": str(self.trace_id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "agent_id": self.agent_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "observation_level": self.observation_level.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        payload = dict(data)
        if "event_id" in payload and payload["event_id"] is not None:
            payload["event_id"] = UUID(str(payload["event_id"]))
        if "trace_id" in payload and payload["trace_id"] is not None:
            payload["trace_id"] = UUID(str(payload["trace_id"]))
        if "parent_id" in payload and payload["parent_id"] is not None:
            payload["parent_id"] = UUID(str(payload["parent_id"]))
        if "timestamp" in payload and payload["timestamp"] is not None:
            payload["timestamp"] = datetime.fromisoformat(str(payload["timestamp"]))
        if "event_type" in payload and isinstance(payload["event_type"], str):
            payload["event_type"] = EventType(payload["event_type"])
        if "status" in payload and isinstance(payload["status"], str):
            payload["status"] = EventStatus(payload["status"])
        if "observation_level" in payload and isinstance(payload["observation_level"], str):
            payload["observation_level"] = ObservationLevel(payload["observation_level"])
        return cls(**payload)


class Trace(BaseModel):
    """Represents a complete agent execution."""

    trace_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: EventStatus = EventStatus.PENDING
    event_count: int = 0
    total_latency_ms: float = 0.0
    events: list[AgentEvent] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None

    def add_event(self, event: AgentEvent) -> None:
        if event.trace_id != self.trace_id:
            raise ValueError("Event trace_id does not match Trace trace_id")
        self.events.append(event)
        self.event_count = len(self.events)
        self.total_latency_ms = sum((e.latency_ms or 0.0) for e in self.events)

    def get_events_by_type(self, event_type: EventType) -> list[AgentEvent]:
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_status(self, status: EventStatus) -> list[AgentEvent]:
        return [event for event in self.events if event.status == status]

    def finalize(self, status: Optional[EventStatus] = None) -> None:
        if status is not None:
            self.status = status
        self.end_time = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": str(self.trace_id),
            "agent_id": self.agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "event_count": self.event_count,
            "total_latency_ms": self.total_latency_ms,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        payload = dict(data)
        if "trace_id" in payload and payload["trace_id"] is not None:
            payload["trace_id"] = UUID(str(payload["trace_id"]))
        if "start_time" in payload and payload["start_time"] is not None:
            payload["start_time"] = datetime.fromisoformat(str(payload["start_time"]))
        if "end_time" in payload and payload["end_time"] is not None:
            payload["end_time"] = datetime.fromisoformat(str(payload["end_time"]))
        if "status" in payload and isinstance(payload["status"], str):
            payload["status"] = EventStatus(payload["status"])
        if "events" in payload:
            payload["events"] = [
                AgentEvent.from_dict(event) if isinstance(event, dict) else event
                for event in payload["events"]
            ]
        return cls(**payload)


def create_event(
    trace_id: UUID,
    agent_id: str,
    event_type: EventType,
    status: EventStatus = EventStatus.SUCCESS,
    input_data: Optional[dict[str, Any]] = None,
    output_data: Optional[dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    parent_id: Optional[UUID] = None,
    metadata: Optional[dict[str, Any]] = None,
    importance_score: float = 0.5,
    observation_level: ObservationLevel = ObservationLevel.FULL,
) -> AgentEvent:
    return AgentEvent(
        trace_id=trace_id,
        agent_id=agent_id,
        event_type=event_type,
        status=status,
        input_data=input_data,
        output_data=output_data,
        latency_ms=latency_ms,
        parent_id=parent_id,
        metadata=metadata,
        importance_score=importance_score,
        observation_level=observation_level,
    )


def create_trace(
    agent_id: str,
    metadata: Optional[dict[str, Any]] = None,
    status: EventStatus = EventStatus.PENDING,
) -> Trace:
    return Trace(agent_id=agent_id, metadata=metadata, status=status)
