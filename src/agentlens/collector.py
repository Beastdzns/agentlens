"""
Event collection and normalization for AgentLens.

This module provides the primary interface for instrumenting AI agents to emit
structured execution events to AgentLens.

Key responsibilities:
- Capture events from instrumented AI agents
- Normalize events into the AgentLens event schema
- Manage trace context (trace_id, parent_id relationships)
- Handle context propagation across async/concurrent operations
- Persist events to storage

PHASE 1 PLACEHOLDER:
This module currently contains only basic interface stubs.
Full collector implementation will be added in Phase 1.

Future responsibilities:
- EventCollector: Main SDK class for instrumenting agents
- Context manager for associating events with traces
- Async event handling and batch processing
- Storage adapter interface
"""

from typing import Any, Optional, Callable
from contextlib import asynccontextmanager
from uuid import UUID


class EventCollector:
    """
    PLACEHOLDER: Main event collection interface.
    
    Agents will use EventCollector to emit structured events.
    
    PHASE 1 will implement:
    - Initialization with storage backend
    - Event creation and validation
    - Trace lifecycle management
    - Context propagation
    - Error handling and fail-open semantics
    
    Example usage (future):
    
        collector = EventCollector(storage_backend=PostgresEventStore())
        
        with collector.trace("user-request-1") as trace:
            # Agent operations emit events
            result = llm.call("prompt")
            
            event = collector.record_event(
                event_type="llm_call",
                input={"prompt": "..."},
                output={"response": "..."},
                latency=1.234,
            )
    """
    
    def __init__(self, storage_backend: Optional[Any] = None) -> None:
        """
        Initialize the event collector.
        
        Args:
            storage_backend: Storage implementation (None = in-memory for now)
        """
        self.storage_backend = storage_backend
    
    async def record_event(
        self,
        event_type: str,
        input_data: Any,
        output_data: Any,
        latency: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Record a single agent event.
        
        PHASE 1 will implement full validation and persistence.
        
        Args:
            event_type: Type of agent operation
            input_data: Input to the operation
            output_data: Result of the operation
            latency: Duration in seconds
            metadata: Additional context
            
        Returns:
            Recorded event with assigned IDs
        """
        raise NotImplementedError("Phase 1 implementation required")
    
    @asynccontextmanager
    async def trace(self, trace_id: Optional[str] = None, agent_id: Optional[str] = None):
        """
        Context manager for a complete agent execution trace.
        
        PHASE 1 will implement proper trace lifecycle.
        
        Args:
            trace_id: Optional explicit trace ID (auto-generated if None)
            agent_id: Identifier for the agent
            
        Yields:
            Trace object for recording events within this context
            
        Example:
            async with collector.trace("user-request-1") as trace:
                # Events emitted here are associated with this trace
                pass
        """
        raise NotImplementedError("Phase 1 implementation required")


class EventNormalizer:
    """
    PLACEHOLDER: Normalize heterogeneous agent events to AgentLens schema.
    
    Different agent frameworks (LangChain, LlamaIndex, custom, etc.) emit
    events in different formats. EventNormalizer adapts them to the canonical
    AgentLens event schema.
    
    PHASE 1 will implement:
    - Adapter patterns for different frameworks
    - Schema validation and transformation
    - Timestamp normalization
    - Field mapping and defaults
    
    Example usage (future):
    
        normalizer = EventNormalizer()
        agentlens_event = normalizer.from_langchain_event(langchain_event)
        agentlens_event = normalizer.from_llamaindex_event(llamaindex_event)
    """
    
    @staticmethod
    def from_dict(data: dict) -> dict:
        """
        Normalize a generic dict event to AgentLens schema.
        
        PHASE 1 will implement validation and transformation.
        """
        raise NotImplementedError("Phase 1 implementation required")


class ContextManager:
    """
    PLACEHOLDER: Manage execution context for proper event association.
    
    When events are emitted from different threads/tasks, this manages
    the current trace_id and parent_id context so events are properly
    hierarchical and grouped.
    
    PHASE 1 will implement:
    - Thread-local context storage
    - Async context variables
    - Automatic parent-child relationship tracking
    - Context propagation across boundaries
    
    Example usage (future):
    
        ctx = ContextManager()
        ctx.set_trace_id(trace_uuid)
        ctx.push_event(event_id)  # Make this the parent for next event
        ...
        ctx.pop_event()  # Back to previous context
    """
    
    pass
