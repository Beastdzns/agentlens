"""
Event models and schemas for AgentLens.

This module defines the core data structures for representing AI agent execution events.

Key concepts:
- AgentEvent: Structured representation of a single agent operation
- Trace: Complete agent execution containing multiple events
- Event types: LLM_CALL, TOOL_CALL, RETRIEVAL, MEMORY_READ, MEMORY_WRITE, DECISION, ERROR, RETRY, etc.

PHASE 1 PLACEHOLDER:
This module currently contains only basic type definitions and interfaces.
Full event model implementation will be added in Phase 1.

Future responsibilities:
- Define Pydantic models for structured events
- Implement event validation
- Support parent-child event relationships
- Track trace_id and event_id uniqueness
- Store event metadata and status
- Support event serialization/deserialization
"""

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from uuid import UUID


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
    """
    Adaptive observation levels for events.
    
    PHASE 2 will implement logic to assign these dynamically.
    """
    
    FULL = "full"  # Capture detailed information
    COMPRESSED = "compressed"  # Store reduced information/summary
    METADATA_ONLY = "metadata_only"  # Store minimal information
    DROP = "drop"  # Do not retain event payload


class AgentEvent:
    """
    PLACEHOLDER: Core event representation.
    
    PHASE 1 will implement this as a full Pydantic model with:
    - event_id: Unique event identifier
    - trace_id: Identifier for the complete trace/execution
    - parent_id: Optional parent event for hierarchical relationships
    - agent_id: Identifier for the agent that generated this event
    - event_type: Type of agent operation (from EventType enum)
    - timestamp: When the event occurred
    - input: Input to the operation
    - output: Result of the operation
    - status: Success/failure status
    - latency: Duration of the operation
    - metadata: Additional contextual information
    - importance_score: (PHASE 2) How important is this event for diagnosis?
    - observation_level: (PHASE 2) How much detail should be recorded?
    
    Example usage (future):
    
        event = AgentEvent(
            event_id=uuid.uuid4(),
            trace_id=trace_uuid,
            agent_id="agent-1",
            event_type=EventType.LLM_CALL,
            timestamp=datetime.now(),
            input={"prompt": "..."},
            output={"response": "..."},
            status=EventStatus.SUCCESS,
            latency=1.234,
        )
    """
    
    pass


class Trace:
    """
    PLACEHOLDER: Represents a complete agent execution.
    
    PHASE 1 will implement this to:
    - Group events by trace_id
    - Track start and end times
    - Accumulate statistics (event count, total latency, etc.)
    - Provide event ordering and hierarchy
    
    Example usage (future):
    
        trace = Trace(
            trace_id=uuid.uuid4(),
            agent_id="agent-1",
            start_time=datetime.now(),
        )
        
        for event in events:
            trace.add_event(event)
    """
    
    pass
