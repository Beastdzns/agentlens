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
"""

from typing import Any, Optional

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    Trace,
    create_event,
    create_trace,
)


class EventCollector:
    """
    Main event collection interface for capturing agent execution events.
    
    Agents use EventCollector to emit structured events for observability.
    Supports trace lifecycle management and event recording.
    
    Example usage:
    
        collector = EventCollector()
        trace = collector.start_trace(agent_id="agent-1")
        
        collector.record_llm_call(
            agent_id="agent-1",
            prompt="What is X?",
            response="X is...",
            latency_ms=1234.5
        )
        
        final_trace = collector.end_trace(EventStatus.SUCCESS)
    """
    
    def __init__(self) -> None:
        """Initialize the event collector."""
        self.current_trace: Optional[Trace] = None
    
    def start_trace(self, agent_id: str, metadata: Optional[dict] = None) -> Trace:
        """
        Start a new trace for agent execution.
        
        Args:
            agent_id: Identifier for the agent.
            metadata: Optional trace-level metadata.
            
        Returns:
            New Trace instance.
        """
        self.current_trace = create_trace(agent_id=agent_id, metadata=metadata)
        return self.current_trace
    
    def record_llm_call(
        self,
        agent_id: str,
        prompt: str,
        response: str,
        latency_ms: float,
        model: str = "gemini-2.0-flash",
        status: EventStatus = EventStatus.SUCCESS,
    ) -> AgentEvent:
        """
        Record an LLM call event.
        
        Args:
            agent_id: Identifier for the agent.
            prompt: The input prompt.
            response: The model response.
            latency_ms: Duration in milliseconds.
            model: Model identifier.
            status: Completion status.
            
        Returns:
            Recorded AgentEvent.
            
        Raises:
            RuntimeError: If no trace is active.
        """
        if not self.current_trace:
            raise RuntimeError("No trace started. Call start_trace() first.")
        
        event = create_event(
            trace_id=self.current_trace.trace_id,
            agent_id=agent_id,
            event_type=EventType.LLM_CALL,
            status=status,
            input_data={"prompt": prompt, "model": model},
            output_data={"response": response},
            latency_ms=latency_ms,
            metadata={"model": model},
        )
        self.current_trace.add_event(event)
        return event
    
    def record_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        input_data: dict,
        output_data: dict,
        latency_ms: float,
        status: EventStatus = EventStatus.SUCCESS,
    ) -> AgentEvent:
        """
        Record a tool call event.
        
        Args:
            agent_id: Identifier for the agent.
            tool_name: Name of the tool.
            input_data: Tool inputs.
            output_data: Tool outputs.
            latency_ms: Duration in milliseconds.
            status: Completion status.
            
        Returns:
            Recorded AgentEvent.
            
        Raises:
            RuntimeError: If no trace is active.
        """
        if not self.current_trace:
            raise RuntimeError("No trace started. Call start_trace() first.")
        
        event = create_event(
            trace_id=self.current_trace.trace_id,
            agent_id=agent_id,
            event_type=EventType.TOOL_CALL,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            metadata={"tool": tool_name},
        )
        self.current_trace.add_event(event)
        return event
    
    def record_event(
        self,
        agent_id: str,
        event_type: EventType,
        status: EventStatus,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> AgentEvent:
        """
        Record a generic agent event.
        
        Args:
            agent_id: Identifier for the agent.
            event_type: Type of agent operation.
            status: Completion status.
            input_data: Input to the operation.
            output_data: Result of the operation.
            latency_ms: Duration in milliseconds.
            metadata: Additional context.
            
        Returns:
            Recorded AgentEvent.
            
        Raises:
            RuntimeError: If no trace is active.
        """
        if not self.current_trace:
            raise RuntimeError("No trace started. Call start_trace() first.")
        
        event = create_event(
            trace_id=self.current_trace.trace_id,
            agent_id=agent_id,
            event_type=event_type,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        self.current_trace.add_event(event)
        return event
    
    def end_trace(self, status: EventStatus = EventStatus.SUCCESS) -> Trace:
        """
        Finalize the current trace.
        
        Args:
            status: Final status for the trace.
            
        Returns:
            Finalized Trace.
            
        Raises:
            RuntimeError: If no trace is active.
        """
        if not self.current_trace:
            raise RuntimeError("No trace to finalize.")
        
        self.current_trace.finalize(status)
        trace = self.current_trace
        self.current_trace = None
        return trace
    
    def get_current_trace(self) -> Optional[Trace]:
        """
        Get the currently active trace.
        
        Returns:
            Current Trace or None if no trace is active.
        """
        return self.current_trace
